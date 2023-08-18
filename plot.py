import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def plot_1():
    yolov7    = [0.50, 0.406]
    yolov7_RL = [0.523, 0.415]
    index     = ['precision', 'recall']

    sns.set_style("darkgrid")
    pd.DataFrame({'yolov7': yolov7, 'yolov7_RL': yolov7_RL}, index=index) \
                        .plot(rot=0, kind='bar', color=["#5975a4", "#cc8963"], width=0.3)
    plt.ylim(0, 1)
    plt.show()

def plot_2(csv_data):
    # 讀取 CSV 檔案
    data = pd.read_csv(csv_data)

    # 提取需要繪製直方圖的欄位
    column_name = 'score_gap' 
    data = data[data[column_name] != 0]
    column_data = data[column_name]

    # 算平均值
    mean_value = column_data.mean()
    print("Average value: ", mean_value)

    # 算標準差
    std_dev = column_data.std()
    print("Standard deviation: ", std_dev)

    # 繪製直方圖
    plt.hist(column_data, bins=100)  # 可以根據需要調整 bins 的數量
    plt.xlabel("score difference")
    plt.ylabel('Count')

    # 顯示圖形
    plt.show()

def plot_3(csv_data, polyorders):
    # 读取 CSV 文件
    data = pd.read_csv(csv_data)

    # 提取需要绘制折线图的列名
    column_name = 'reward'  # 请将 'column_name' 替换为您想要绘制折线图的列名
    # data = data[data[column_name] != 0]

    # 获取需要绘制的数据
    column_data = data[column_name]

    # 计算每个 epoch 的范围
    per_epoch = 20000
    epoch_range = np.arange(per_epoch, len(column_data)+1, per_epoch)

    # 创建包含多个子图的图像
    if len(polyorders) == 1:
        smoothed_data = savgol_filter(column_data.values, window_length=1000, polyorder=polyorders[0])
        plt.plot(smoothed_data)
        plt.xlabel('Epoch')
        plt.ylabel(column_name)

    else:
        fig, ax = plt.subplots(1, len(polyorders), figsize=(18, 5))

        # 绘制多个折线图
        for i, polyorder in enumerate(polyorders):
            # 平滑数据
            smoothed_data = savgol_filter(column_data.values, window_length=2000, polyorder=polyorder)

            # 绘制折线图
            ax[i].plot(smoothed_data, label='Polyorder {}'.format(polyorder))
            ax[i].set_xlabel('Index')
            ax[i].set_ylabel(column_name)
            ax[i].set_title('Smoothed Line Charts of {}'.format(column_name))
            ax[i].legend()

            # 设置 x 轴刻度和标签
            ax[i].set_xticks(epoch_range)
            ax[i].set_xticklabels(np.arange(1, len(epoch_range)+1))

    # 显示图形
    plt.show()

def plot_4(csv_data):
    # Read CSV 
    data = pd.read_csv(csv_data)
    column_data = data[['brightness', 'saturation', 'contrast', 'sharpness']]

    # Create a grid of subplots. Adjust the size (15, 10) and spacing as needed.
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axs = axs.ravel()   # Flatten the axis array to make iteration easier

    # Use pandas describe() function to get the statistical summary 
    # that includes mean, standard deviation, and other metrics.
    summary = column_data.describe()

    # To make the printout easier to read, we use a formatted print statement 
    # and iterate through each column's statistics.
    for col in summary.columns:
        print(f"Statistics for {col}:\n"
            f"  Average value: {summary.loc['mean', col]:.2f}\n"
            f"  Standard deviation: {summary.loc['std', col]:.2f}\n")

    # Iterate over the column names
    for index, column in enumerate(column_data.columns):
        axs[index].hist(column_data[column], bins=50, ec='black')
        axs[index].set_title(f'Histogram of {column}')
        axs[index].set_xlabel(column + " factor")
        axs[index].set_ylabel('Count')

    # Display the figure with all histograms
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    csv_data = "runs\score\exp\metric.csv"
    plot_4(csv_data)