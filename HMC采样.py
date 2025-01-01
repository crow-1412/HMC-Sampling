import numpy as np
from scipy.stats import norm, entropy
import matplotlib.pyplot as plt

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("未找到 Microsoft YaHei 字体，尝试其他中文字体")
    # 备选方案
    chinese_fonts = ['SimHei', 'KaiTi', 'FangSong', 'SimSun', 'NSimSun', 'STXihei']
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            break
        except:
            continue

# 保持原有的混合高斯分布参数
weights = [0.3, 0.7]
means = [4, 7]
variances = [0.3, 2]

def target_log_prob(x):
    """目标分布的对数概率密度"""
    return np.log(weights[0] * norm.pdf(x, loc=means[0], scale=np.sqrt(variances[0])) + 
                  weights[1] * norm.pdf(x, loc=means[1], scale=np.sqrt(variances[1])))

def target_grad(x):
    """目标分布的梯度"""
    comp1 = weights[0] * norm.pdf(x, means[0], np.sqrt(variances[0])) * (means[0] - x) / variances[0]
    comp2 = weights[1] * norm.pdf(x, means[1], np.sqrt(variances[1])) * (means[1] - x) / variances[1]
    return (comp1 + comp2) / (weights[0] * norm.pdf(x, means[0], np.sqrt(variances[0])) + 
                             weights[1] * norm.pdf(x, means[1], np.sqrt(variances[1])))

def hamiltonian_monte_carlo(n_samples, epsilon=0.05, L=20, init_position=5.5):
    """
    改进的HMC采样器
    n_samples: 采样数量
    epsilon: 步长（调小以提高精确度）
    L: 每次采样的leapfrog步数（增加以更好地探索空间）
    init_position: 初始位置（设置为更接近理论均值）
    """
    samples = np.zeros(n_samples)
    x = init_position  # 使用更好的初始位置
    
    accept_count = 0  # 记录接受率
    
    for i in range(n_samples):
        x_current = x
        p_current = np.random.normal(0, 1)  # 动量初始化
        
        # 当前哈密顿量
        current_H = -target_log_prob(x_current) + 0.5 * p_current**2
        
        # Leapfrog积分
        x_proposed = x_current
        p_proposed = p_current
        
        for j in range(L):
            p_proposed = p_proposed + 0.5 * epsilon * target_grad(x_proposed)
            x_proposed = x_proposed + epsilon * p_proposed
            p_proposed = p_proposed + 0.5 * epsilon * target_grad(x_proposed)
        
        # 提议哈密顿量
        proposed_H = -target_log_prob(x_proposed) + 0.5 * p_proposed**2
        
        # Metropolis接受/拒绝
        if np.random.random() < np.exp(current_H - proposed_H):
            x = x_proposed
            accept_count += 1
        
        samples[i] = x
    
    acceptance_rate = accept_count / n_samples
    return samples, acceptance_rate

def calculate_metrics(samples, x_vals, target_vals):
    """计算评估指标"""
    hist, bin_edges = np.histogram(samples, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    epsilon = 1e-10
    hist_normalized = hist + epsilon
    target_at_bins = weights[0] * norm.pdf(bin_centers, means[0], np.sqrt(variances[0])) + \
                     weights[1] * norm.pdf(bin_centers, means[1], np.sqrt(variances[1])) + epsilon
    kl_div = entropy(hist_normalized, target_at_bins)
    
    mad = np.mean(np.abs(hist - target_at_bins))
    
    theoretical_mean = weights[0] * means[0] + weights[1] * means[1]
    theoretical_var = weights[0] * (variances[0] + means[0]**2) + \
                     weights[1] * (variances[1] + means[1]**2) - theoretical_mean**2
    
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    return {
        'kl_divergence': kl_div,
        'mad': mad,
        'theoretical_mean': theoretical_mean,
        'sample_mean': sample_mean,
        'theoretical_var': theoretical_var,
        'sample_var': sample_var,
        'mean_error': abs(theoretical_mean - sample_mean),
        'var_error': abs(theoretical_var - sample_var)
    }

# 设置随机种子以确保可重复性
np.random.seed(42)

# 增加采样数量
num_samples = 20000  # 增加采样数量以提高精确度

# 使用调优后的参数进行采样
hmc_samples, acceptance_rate = hamiltonian_monte_carlo(
    n_samples=num_samples,
    epsilon=0.05,  # 较小的步长以提高精确度
    L=20,          # 增加leapfrog步数以better探索
    init_position=5.5  # 初始位置接近理论均值
)

# 计算目标分布
x_vals = np.linspace(0, 10, 1000)
target_vals = weights[0] * norm.pdf(x_vals, means[0], np.sqrt(variances[0])) + \
             weights[1] * norm.pdf(x_vals, means[1], np.sqrt(variances[1]))

# 计算评估指标
metrics = calculate_metrics(hmc_samples, x_vals, target_vals)

# 创建图形和子图，调整布局
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.5

# 调整图形大小和子图之间的间距
fig = plt.figure(figsize=(10, 12))  # 调整整体高度
gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1.2], hspace=0.4)

# 第一个子图：分布对比
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_vals, target_vals, color='#FF6B6B', label='目标分布', linewidth=2)
ax1.hist(hmc_samples, bins=50, density=True, alpha=0.6, label='HMC采样', 
         color='#4ECDC4', edgecolor='white')
ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('密度', fontsize=12, fontweight='bold')
ax1.set_title('改进后的HMC采样 vs 目标分布', fontsize=14, pad=15, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.8)
ax1.grid(True, alpha=0.3, color='white')
ax1.set_facecolor('#f8f9fa')

# 第二个子图：采样轨迹
ax2 = fig.add_subplot(gs[1])
ax2.plot(np.arange(1000), hmc_samples[:1000], color='#45B7D1', alpha=0.8, linewidth=1)
ax2.set_xlabel('采样步数', fontsize=12, fontweight='bold')
ax2.set_ylabel('采样值', fontsize=12, fontweight='bold')
ax2.set_title('采样轨迹（前1000步）', fontsize=14, pad=15, fontweight='bold')
ax2.grid(True, alpha=0.3, color='white')
ax2.set_facecolor('#f8f9fa')

# 第三个子图：评估指标
ax3 = fig.add_subplot(gs[2])
ax3.set_title('评估指标', fontsize=14, pad=15, fontweight='bold')

# 将评估指标分为左右两列
left_metrics = (
    f"KL散度:           {metrics['kl_divergence']:.4f}\n"
    f"平均绝对偏差(MAD): {metrics['mad']:.4f}\n"
    f"理论均值:         {metrics['theoretical_mean']:.4f}\n"
    f"采样均值:         {metrics['sample_mean']:.4f}\n"
    f"均值误差:         {metrics['mean_error']:.4f}"
)

right_metrics = (
    f"理论方差: {metrics['theoretical_var']:.4f}\n"
    f"采样方差: {metrics['sample_var']:.4f}\n"
    f"方差误差: {metrics['var_error']:.4f}\n"
    f"接受率:   {acceptance_rate:.2%}\n"
)

# 创建左侧文本框
left_box = ax3.text(
    0.25, 0.5,  # 左侧位置
    left_metrics,
    transform=ax3.transAxes,
    fontsize=11,
    family='Microsoft YaHei',
    ha='center',
    va='center',
    bbox=dict(
        facecolor='#f8f9fa',
        edgecolor='#ddd',
        boxstyle='round,pad=1',
        alpha=1.0
    ),
    linespacing=2.0
)

# 创建右侧文本框
right_box = ax3.text(
    0.75, 0.5,  # 右侧位置
    right_metrics,
    transform=ax3.transAxes,
    fontsize=11,
    family='Microsoft YaHei',
    ha='center',
    va='center',
    bbox=dict(
        facecolor='#f8f9fa',
        edgecolor='#ddd',
        boxstyle='round,pad=1',
        alpha=1.0
    ),
    linespacing=2.0
)

# 关闭坐标轴
ax3.set_axis_off()

# 调整布局前先设置合适的边距
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05)

# 保存图片
plt.savefig('HMC采样结果.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            pad_inches=0.5)

plt.show()