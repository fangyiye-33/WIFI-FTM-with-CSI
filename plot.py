import matplotlib.pyplot as plt
import numpy as np

# ==================== 可修改的参数 ====================

# X轴标签（实际距离）- 4个标签
X_LABELS = ['2.76m L=3m', '2.76m L=4.5m', '2.76m NLOS L=4.5m', '4.5m L=4.5m']

# Y轴标签
Y_LABEL = 'Computed distance (m)'
X_LABEL = 'GROUP'

# 图例标签（修改这些文字）- 3个柱子
LEGEND_LABELS = [
    'Real',  # 第1个：深蓝色
    'FUSIC',  # 第2个：浅黄色（带斜线）
    'FTM'  # 第3个：红色
]

# 图表标题（子图标题）
SUBPLOT_TITLE = '(a) Raw results'

# 数据：每个X位置有3个柱状图的数据
# 格式：[[位置1的3个值], [位置2的3个值], ...]
DATA = [
    [2.76, 3.02, 9.05],
    [2.76, 3.28, 10.5],
    [2.76, 3.65, 12.285],
    [4.5, 4.67, 10.665]
]

# 误差条（可选，设为None则不显示）
ERRORS = [
    [0.3, 0.5, 0.8],
    [0.4, 0.6, 1.0],
    [0.5, 0.8, 1.2],
    [0.6, 1.0, 1.5]
]

# 颜色配置 - 3个颜色
COLORS = ['navy', 'lightyellow', 'red']

# 柱状图宽度
BAR_WIDTH = 0.2


# ==================== 绘图代码 ====================

def plot_bar_chart():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(X_LABELS))  # X轴位置

    # 绘制每个组的柱状图（3个柱子）
    for i in range(3):
        # 计算每个柱子的位置（3个柱子：-1, 0, 1）
        positions = x + (i - 1) * BAR_WIDTH

        # 提取该组的数据
        values = [DATA[j][i] for j in range(len(X_LABELS))]

        # 提取误差（如果有）
        yerr = [ERRORS[j][i] for j in range(len(X_LABELS))] if ERRORS else None

        # 设置填充样式
        if i == 1:  # 第二个：斜线填充
            hatch = '//'
            edgecolor = 'black'
        else:
            hatch = None
            edgecolor = 'black' if i == 1 else COLORS[i]

        # 绘制柱状图
        bars = ax.bar(positions, values, BAR_WIDTH,
                      label=LEGEND_LABELS[i],
                      color=COLORS[i],
                      edgecolor=edgecolor,
                      hatch=hatch,
                      linewidth=1.5)

        # 添加误差条
        if yerr:
            ax.errorbar(positions, values, yerr=yerr,
                        fmt='none', color='black', capsize=3, linewidth=1)

    # 设置X轴
    ax.set_xlabel(X_LABEL, fontsize=14)
    ax.set_ylabel(Y_LABEL, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=12)
    ax.set_ylim(0, 15)

    # 添加图例（1行3列布局）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=9, frameon=True, fancybox=True)

    # 添加网格线
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 添加子图标题
    fig.text(0.5, 0.02, SUBPLOT_TITLE, ha='center', fontsize=16, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # 保存图片
    plt.savefig('bar_chart_results.png', dpi=300, bbox_inches='tight')
    print("图片已保存: bar_chart_results.png")

    plt.show()


if __name__ == '__main__':
    plot_bar_chart()
