import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import tensorflow as tf
from multiprocessing import Pool


def apply(func, args=None, kwds=None):
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def combined_cavity_u(repeat, num_train, lr=5e-4, pde_weight=0.0, data_weight=0.3, model_restore_path="model_u/model"):
    import deepxde as dde

    def pde(x, y):
        # Placeholder: cavity FT_Phys not present in repo; set pde_weight=0 for purely observational training
        return tf.zeros_like(y)

    class CombinedData(dde.data.data.Data):
        def __init__(self, X_train, y_train, X_test, y_test, num_train_obs):
            self.train_x = X_train
            self.train_y = y_train
            self.test_x = X_test
            self.test_y = y_test
            self.num_obs = num_train_obs
            self.train_sampler = dde.data.sampler.BatchSampler(len(self.train_y) - self.num_obs, shuffle=True)

        def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
            y_pred = outputs
            x_trunk = inputs[1]
            residual = pde(x_trunk, y_pred)
            pde_loss = dde.losses.mean_squared_error(residual, tf.zeros_like(residual))
            if self.num_obs > 0:
                data_pred = y_pred[-self.num_obs:]
                data_true = targets[-self.num_obs:]
                data_loss = dde.losses.mean_squared_error(data_pred, data_true)
            else:
                data_loss = tf.constant(0.0, dtype=tf.float32)
            return pde_weight * pde_loss + data_weight * data_loss

        def train_next_batch(self, batch_size=None):
            if batch_size is None:
                return self.train_x, self.train_y
            indices = self.train_sampler.get_next(batch_size)
            branch = np.concatenate([self.train_x[0][indices], self.train_x[0][-self.num_obs:]], axis=0)
            trunk = np.concatenate([self.train_x[1][indices], self.train_x[1][-self.num_obs:]], axis=0)
            y = np.concatenate([self.train_y[indices], self.train_y[-self.num_obs:]], axis=0)
            return (branch, trunk), y

        def test(self):
            return self.test_x, self.test_y

    data_test = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_train_u_addition = np.tile(sensor_value, [num_train, 1])
    X_train_addition = data_test['x_trunk_test_select'][repeat]
    y_train_addition = data_test['u_test_select'][repeat]

    train_data = np.load("../../../data/cavity_flow/cavity_train.npz")
    X_train = (np.concatenate([train_data["branch_train"][-5*10201:], X_train_u_addition], axis=0),
               np.concatenate([train_data["trunk_train"][-5*10201:], X_train_addition], axis=0))
    y_train = np.concatenate([train_data["u_train"][-5*10201:], y_train_addition], axis=0)

    X_test0 = np.tile(sensor_value, [10201, 1])
    X_test1 = data_test['x_trunk_test'][repeat]
    X_test = (X_test0, X_test1)
    y_test = data_test['u_test'][repeat]

    data = CombinedData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_train_obs=num_train)

    net = dde.nn.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    net.build()

    model = dde.Model(data, net)
    loss_func = lambda y_true, y_pred: dde.losses.mean_squared_error(y_true[:-num_train], y_pred[:-num_train]) + 0.3 * dde.losses.mean_squared_error(y_true[-num_train:], y_pred[-num_train:])
    iterations = 3000
    model.compile("adam", lr=lr, loss=loss_func, metrics=["l2 relative error"], decay=("inverse time", iterations // 5, 0.8))
    model.train(iterations=iterations, display_every=100, model_restore_path=model_restore_path)

    return model.predict(X_test).reshape(1, 101*101)


if __name__ == "__main__":
    num_train = 100
    output_list = []
    for repeat in range(100):
        output = apply(combined_cavity_u, [repeat, num_train])
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_phys_combined_cavity_u.npy", output_list)
