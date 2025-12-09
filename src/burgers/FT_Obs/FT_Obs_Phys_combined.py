import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import tensorflow.compat.v1 as tf
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


def periodic(x):
    return tf.concat([tf.math.cos(x[:, 0:1] * 2 * np.pi), tf.math.sin(x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(2 * x[:, 0:1] * 2 * np.pi), tf.math.sin(2 * x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(3 * x[:, 0:1] * 2 * np.pi), tf.math.sin(3 * x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(4 * x[:, 0:1] * 2 * np.pi), tf.math.sin(4 * x[:, 0:1] * 2 * np.pi), x[:, 1:2]], 1)


def combined_burgers(repeat, ls_test, num_train, lr=1e-3, pde_weight=1.0, data_weight=0.3,
                     model_restore_path="model/model", trainable_branch=True, trainable_trunk=[True, True, True]):
    import deepxde as dde

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.1 * dy_xx

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

    data_test = np.load(f"../../../data/burgers/burgers_test_ls_{ls_test}_num_{num_train}.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_train_branch_addition = np.tile(sensor_value, [num_train, 1])
    X_train_trunk_addition = data_test['x_trunk_test_select'][repeat]
    y_train_addition = data_test['y_test_select'][repeat]

    data_train = np.load(f"../../../data/burgers/burgers_train_ls_1.0_101_101.npz")
    X_train_branch_origin = np.repeat(data_train["X_train0"], 101 * 101, axis=0)
    X_train_trunk_origin = np.tile(data_train["X_train1"], (1000, 1))
    y_train_origin = data_train["y_train"].reshape(-1, 1)

    X_train = (np.concatenate([X_train_branch_origin, X_train_branch_addition], axis=0),
               np.concatenate([X_train_trunk_origin, X_train_trunk_addition], axis=0))
    y_train = np.concatenate([y_train_origin, y_train_addition], axis=0)

    X_test = (np.tile(sensor_value, [101*101, 1]),
              np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)]))
    y_test = data_test['y_test'][repeat]

    data = CombinedData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_train_obs=num_train)

    net = dde.nn.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=trainable_branch,
        trainable_trunk=trainable_trunk,
    )
    net.apply_feature_transform(periodic)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    def loss_func(y_true, y_pred):
        return dde.losses.mean_squared_error(y_true, y_pred)

    iterations = 3000
    model.compile("adam", lr=lr, loss=loss_func, metrics=["l2 relative error"], decay=("inverse time", iterations // 5, 0.8))
    model.train(iterations=iterations, display_every=100, batch_size=10000, model_restore_path=model_restore_path)

    return model.predict(X_test).reshape(1, 101*101)


if __name__ == "__main__":
    ls_test = 0.6
    num_train = 100
    output_list = []
    for repeat in range(100):
        output = apply(combined_burgers, (repeat, ls_test, num_train))
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_phys_combined_burgers.npy", output_list)
