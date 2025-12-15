import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from scipy.interpolate import griddata
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


def combined_train(repeat, ls_test, num_train, lr=1e-3, pde_weight=1.0, data_weight=0,
                   model_restore_path="model/model"):
    """
    Combined方法：同时使用物理约束(PDE)和数据约束(观测数据)

    核心思路（参考FT_Phys + FT_Obs_T）：
    1. 使用TimePDE处理PDE残差（来自FT_Phys）
    2. 在batch中同时包含PDE配点和观测点（来自FT_Obs_T的思路）
    3. 对PDE配点应用PDE loss，对观测点应用data loss
    """
    import deepxde as dde

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def dirichlet(inputs, outputs):
        x_trunk = inputs[1]
        x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
        return 4 * x * t * outputs + tf.sin(np.pi * x) + tf.sin(0.5 * np.pi * t)

    # 加载观测数据
    data_test = np.load(f"../../../data/diffusion_reaction/dr_test_ls_{ls_test}_num_{num_train}.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_obs_trunk = data_test['x_trunk_test_select'][repeat]  # (num_train, 2)
    y_obs = data_test['y_test_select'][repeat].reshape(-1, 1)  # (num_train, 1)
    X_obs_branch = np.tile(sensor_value, [num_train, 1])
    xt = np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])

    # PDE定义（物理约束，来自FT_Phys）
    def pde(x, y):
        """
        advection方程的PDE残差
        dy_t + u dy_x
        """
        dy = tf.gradients(y, x)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:]
        # u(x) interpolated from branch (sensor)
        u = tfp.math.batch_interp_regular_1d_grid(x[:, :1], 0, 1, tf.cast(sensor_value, tf.float32))
        return dy_t + u * dy_x

    def func(x_input):
        # 用于生成训练数据的函数（虽然我们主要用观测数据，但TimePDE需要这个）
        uu = data_test['y_test'][repeat].reshape(-1, 1)
        return griddata(xt, uu, x_input, method="cubic")

    # 创建TimePDE（会自动处理PDE梯度计算）
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data_pde = dde.data.TimePDE(geomtime, pde, [], num_domain=1000, solution=func, num_test=10000)

    # ===== 关键修改：修改 train_next_batch 而不是 train_points =====
    # 这样可以确保每个 batch 都包含所有观测点（参考 FT_Obs_T 的实现）

    # 保存原始方法
    original_losses = data_pde.losses
    original_train_next_batch = data_pde.train_next_batch

    # 修改 train_next_batch：每个 batch 从配点中采样 + 添加所有观测点
    def combined_train_next_batch(batch_size=None):
        """
        每个 batch 的组成：
        - 从 1000 个 PDE 配点中采样 batch_size 个
        - 添加所有 100 个观测点
        - 总共：batch_size + 100 个点

        返回：(train_x, train_y, train_aux_vars) tuple
        """
        # 获取配点的 batch - 返回 (X, y, aux) tuple
        colloc_batch_tuple = original_train_next_batch(batch_size)
        colloc_x, colloc_y, colloc_aux = colloc_batch_tuple

        # 添加所有观测点
        if data_weight > 0:
            # 合并 trunk coordinates
            combined_x = np.vstack([colloc_x, X_obs_trunk])
            # 合并 y（观测点的 y 已知）
            combined_y = np.vstack([colloc_y, y_obs]) if colloc_y is not None else None
            # aux_vars 保持 None
            combined_aux = None

            return combined_x, combined_y, combined_aux
        else:
            # 如果 data_weight=0，不添加观测点，保持与原始 TimePDE 完全一致
            return colloc_batch_tuple

    # 替换方法
    data_pde.train_next_batch = combined_train_next_batch

    y_obs_tensor = tf.convert_to_tensor(y_obs, dtype=tf.float32)

    def combined_losses(targets, outputs, loss_fn, inputs, model, aux=None):
        """
        关键修改：返回 list 而不是标量，与原始 TimePDE.losses 的格式一致

        原始 TimePDE.losses 返回：
        - [pde_loss_1, pde_loss_2, ..., bc_loss_1, bc_loss_2, ...]

        我们的 combined 返回：
        - 当 data_weight=0: [pde_loss]  (与原始一致)
        - 当 data_weight>0: [pde_loss, data_loss]
        """

        # ===== 计算 loss =====
        total = tf.shape(outputs)[0]

        if data_weight == 0.0:
            # 纯 PDE 模式：完全模仿原始 TimePDE.losses 的行为
            # 1. 调用 pde 函数获取残差（inputs 只包含 trunk，因为是普通 PDE）
            pde_residual = pde(inputs, outputs)

            # 2. 使用 loss_fn 计算 loss（与原始一致）
            # 原始代码：loss_fn[i](bkd.zeros_like(error), error)
            if isinstance(loss_fn, (list, tuple)):
                pde_loss = loss_fn[0](tf.zeros_like(pde_residual), pde_residual)
            else:
                pde_loss = loss_fn(tf.zeros_like(pde_residual), pde_residual)

            # 3. 返回 list（与原始格式一致）
            return [pde_loss]

        else:
            # 组合模式
            num_obs = tf.constant(num_train, dtype=tf.int32)
            colloc_count = total - num_obs

            # 1. PDE loss：只对前面的配点计算
            # 关键问题：PDE 需要用完整的 inputs 来保持梯度连接
            # 我们不能切片 inputs，而是在 PDE residual 中只使用前面的部分

            # 调用 PDE 在完整 inputs 上（保持梯度连接）
            pde_residual_all = pde(inputs, outputs)

            # 然后只对前面的配点部分计算 loss
            pde_residual = pde_residual_all[:colloc_count]

            if isinstance(loss_fn, (list, tuple)):
                pde_loss = loss_fn[0](tf.zeros_like(pde_residual), pde_residual)
            else:
                pde_loss = loss_fn(tf.zeros_like(pde_residual), pde_residual)

            # 2. Data loss：只对后面的观测点计算
            outputs_obs = outputs[colloc_count:]
            data_residual = outputs_obs - y_obs_tensor

            if isinstance(loss_fn, (list, tuple)) and len(loss_fn) > 1:
                data_loss = loss_fn[1](tf.zeros_like(data_residual), data_residual)
            else:
                # 如果没有第二个 loss_fn，使用第一个或默认的
                if isinstance(loss_fn, (list, tuple)):
                    data_loss = loss_fn[0](tf.zeros_like(data_residual), data_residual)
                else:
                    data_loss = loss_fn(tf.zeros_like(data_residual), data_residual)

            # 3. 应用权重并返回 list
            weighted_pde_loss = pde_weight * pde_loss
            weighted_data_loss = data_weight * data_loss

            return [weighted_pde_loss, weighted_data_loss]

    # ===== 启用 combined_losses =====
    # 替换方法（注意：只替换 losses，train_next_batch 已经在前面替换了）
    data_pde.losses = combined_losses
    print("\n✓ 已启用 combined_losses (返回 list 格式)\n")

    # 构建DeepONet
    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=True,
        trainable_trunk=[True, True, True],
    )

    net.apply_output_transform(dirichlet)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data_pde, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"])

    # ============ 关键修改：使用 FT_Phys 的方式 ============
    # 1. 第二次 compile（与 FT_Phys 保持一致）
    model.compile("adam", lr=lr, metrics=["l2 relative error"])

    # 2. 训练时 restore（而不是提前 restore）
    iterations = 200  # 改为与 FT_Phys 一致的 200 步
    print(f"\n开始Combined训练 (PDE权重={pde_weight}, 数据权重={data_weight}, 迭代={iterations})...")
    print("✓ 使用 FT_Phys 风格：双 compile + train 中 restore\n")

    losshistory, train_state = model.train(
        epochs=iterations,
        display_every=50,
        model_restore_path=model_restore_path  # 在这里 restore
    )

    # 测试
    X_test = (np.tile(sensor_value, [101*101, 1]),
              np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)]))
    y_test = data_test['y_test'][repeat]
    y_pred = model.predict(X_test)

    error = dde.metrics.l2_relative_error(y_test, y_pred)
    print(f"\n最终L2相对误差: {error:.6f}")

    return error


if __name__ == "__main__":
    repeat = 0
    ls_test = 0.2
    num_train = 100
    model_path = "model/model"

    print("=" * 80)
    print("Combined方法完整测试")
    print("=" * 80)
    print(f"样本: {repeat}, 测试长度尺度: {ls_test}, 观测点数: {num_train}")
    print("=" * 80)

    # 测试1: 纯 PDE 模式 (data_weight=0)
    print("\n[测试1] 纯 PDE 模式 (pde_weight=1.0, data_weight=0.0)")
    print("-" * 80)
    error_pde = apply(combined_train, (repeat, ls_test, num_train, 1e-3, 1.0, 0.0, model_path))
    print(f"✓ 纯 PDE 最终误差: {error_pde:.6f}")

    # 测试2: 组合模式 (data_weight=0.3)
    print("\n[测试2] 组合模式 (pde_weight=1.0, data_weight=0.3)")
    print("-" * 80)
    error_combined = apply(combined_train, (repeat, ls_test, num_train, 1e-3, 1.0, 0.3, model_path))
    print(f"✓ 组合模式最终误差: {error_combined:.6f}")

    # 总结
    print("\n" + "=" * 80)
    print("总结对比")
    print("=" * 80)
    print(f"纯 PDE 模式:  {error_pde:.6f}")
    print(f"组合模式:     {error_combined:.6f}")
    print(f"FT_Phys 基线: 0.008883 (参考)")
    print("=" * 80)
