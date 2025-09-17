import scipy.io as sio
import numpy as np
import MUMT as MU
from memory import MemoryDNN
import time
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_gain_comparison(results, param_names, window_size=60):
    """绘制多参数对比图"""
    plt.figure(figsize=(12, 8))

    # 创建主图
    ax = plt.gca()

    for i, (gain_his_ratio, label) in enumerate(results):
        df = pd.DataFrame(gain_his_ratio)
        df_roll = df.rolling(window_size, min_periods=1).mean()
        plt.plot(np.arange(len(gain_his_ratio)) + 1, df_roll, label=label)
        plt.fill_between(
            np.arange(len(gain_his_ratio)) + 1,
            df.rolling(window_size, min_periods=1).min()[0],
            df.rolling(window_size, min_periods=1).max()[0],
            alpha=0.2
        )

    plt.ylabel('Gain ratio')
    plt.xlabel('Learning steps')
    plt.title('Gain Ratio Comparison with Different Parameters')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # 添加放大小图 - 在右上角留白处
    axins = inset_axes(ax, width=2.5, height=2, loc='upper right')

    # 定义放大区域：200-1000步
    zoom_start, zoom_end = 200, 1000

    # 在小图中绘制放大部分
    for i, (gain_his_ratio, label) in enumerate(results):
        df = pd.DataFrame(gain_his_ratio)
        df_roll = df.rolling(window_size, min_periods=1).mean()

        # 获取放大区域的数据
        steps = np.arange(len(gain_his_ratio)) + 1
        zoom_mask = (steps >= zoom_start) & (steps <= zoom_end)

        if np.any(zoom_mask):  # 确保有数据在放大区域内
            axins.plot(steps[zoom_mask], df_roll[zoom_mask],
                       label=label if i == 0 else "")  # 只为第一条线添加标签避免重复

    # 设置放大图属性
    axins.set_xlim(zoom_start, zoom_end)

    # 自动计算y轴范围，让放大效果更明显
    y_data_in_range = []
    for gain_his_ratio, _ in results:
        df = pd.DataFrame(gain_his_ratio)
        df_roll = df.rolling(window_size, min_periods=1).mean()
        steps = np.arange(len(gain_his_ratio)) + 1
        zoom_mask = (steps >= zoom_start) & (steps <= zoom_end)
        if np.any(zoom_mask):
            y_data_in_range.extend(df_roll[zoom_mask].values.flatten())

    if y_data_in_range:
        y_min, y_max = min(y_data_in_range), max(y_data_in_range)
        y_range = y_max - y_min
        axins.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    axins.set_title('200-1000 Steps (Zoomed)', fontsize=10)
    axins.grid(True, alpha=0.3)

    # 在主图上用半透明区域标记放大范围
    ax.axvspan(zoom_start, zoom_end, alpha=0.15, color='gray', zorder=0)

    plt.tight_layout()
    plt.show()


def run_experiment(param, N, split_idx, num_test, task_size, gain):
    """运行单个参数配置的实验"""
    lr, batch_size, training_interval = param
    mem = MemoryDNN(
        net=[9, 120, 80, 9],  # 3x3任务固定结构
        net_num=3,
        learning_rate=lr,
        training_interval=training_interval,
        batch_size=batch_size,
        memory_size=1024
    )

    env = MU.MUMT(3, 3, rand_seed=1)
    gain_his_ratio = []

    for i in range(N):
        if i % (N // 100) == 0:
            print(f"参数 {param} 进度: {i / N:.2f}")

        # 数据索引处理
        if i < N - num_test:
            i_idx = i % split_idx
        else:
            i_idx = i - N + num_test + split_idx
        t1 = task_size[i_idx, :]
        t = t1 * 10 - 200  # 预处理

        # 决策与训练
        m_list = mem.decode(t)
        r_list = [env.compute_Q(t1, m) for m in m_list]
        mem.encode(t, m_list[np.argmin(r_list)])

        # 记录性能
        gain_his_ratio.append(gain[0][i_idx] / np.min(r_list))

    return gain_his_ratio


if __name__ == "__main__":
    # 配置参数
    N = 10000  # 任务数量（减少以加快对比实验）
    WD_num = 3
    task_num = 3

    # 加载数据
    data = sio.loadmat('./data/MUMT_data_3x3')
    task_size = data['task_size']
    gain = data['gain_min']

    # 数据划分
    split_idx = int(.8 * len(task_size))
    num_test = min(len(task_size) - split_idx, N - int(.8 * N))

    # 待对比的参数组合 (学习率, 批大小, 训练间隔)
    params = [
        (0.01, 128, 10),  # 基准参数
        (0.001, 128, 10),  # 更小学习率
        (0.01, 64, 10),  # 更小批大小
        (0.01, 128, 20)  # 更长训练间隔
    ]
    param_labels = [
        "lr=0.01, batch=128, interval=10",
        "lr=0.001, batch=128, interval=10",
        "lr=0.01, batch=64, interval=10",
        "lr=0.01, batch=128, interval=20"
    ]

    # 运行所有实验
    results = []
    for i, param in enumerate(params):
        print(f"开始运行参数组合 {i + 1}/{len(params)}: {param}")
        start_time = time.time()
        gain_his_ratio = run_experiment(param, N, split_idx, num_test, task_size, gain)
        total_time = time.time() - start_time
        print(f"参数 {param} 耗时: {total_time:.2f}秒")
        results.append((gain_his_ratio, param_labels[i]))

    # 绘制对比图
    plot_gain_comparison(results, param_labels)

    # 打印测试集性能
    for i, (gain_his_ratio, label) in enumerate(results):
        test_ratio = sum(gain_his_ratio[-num_test: -1]) / num_test
        print(f"{label} 测试集平均Gain ratio: {test_ratio:.4f}")