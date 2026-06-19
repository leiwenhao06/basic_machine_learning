# -*- coding: utf-8 -*-
"""
01_python_basics — Python编程练习

包含三部分:
  1. 学生成绩分析：读取成绩文件, 计算统计量, 查找重名学生
  2. 成绩相关性分析：Pearson/Spearman 相关系数, 散点图+回归线
  3. 成绩分布分析：直方图+正态分布曲线, Q-Q图, Shapiro-Wilk 检验
"""

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

# ============================================================
# 中文字体设置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 辅助：生成示例成绩文件
# ============================================================
def generate_score_file(filepath: str):
    """生成示例成绩文件，包含30名学生，3门课程"""
    np.random.seed(42)
    names_pool = [
        "张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十",
        "钱十一", "陈十二", "张三", "李四", "刘十三", "黄十四", "林十五",
        "何十六", "郭十七", "马十八", "罗十九", "梁二十", "宋二一", "唐二二",
        "韩二三", "冯二四", "董二五", "萧二六", "程二七", "曹二八", "袁二九", "邓三十",
    ]
    records = []
    for i, name in enumerate(names_pool):
        # 三科成绩：语文, 数学, 英语
        s1 = np.clip(np.random.normal(78, 12), 30, 100)
        s2 = np.clip(np.random.normal(75, 15), 30, 100)
        s3 = np.clip(np.random.normal(72, 14), 30, 100)
        records.append(f"{name},{i+1:03d},{s1:.1f},{s2:.1f},{s3:.1f}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("姓名,学号,语文,数学,英语\n")
        f.write("\n".join(records))
    print(f"[INFO] 已生成成绩文件: {filepath}")


# ============================================================
# Part 1: 学生成绩分析
# ============================================================
def load_scores(filepath: str):
    """读取成绩文件，返回 names, ids, scores_array (numpy)"""
    names, ids = [], []
    scores_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            names.append(parts[0])
            ids.append(parts[1])
            scores_list.append([float(p) for p in parts[2:]])
    return names, ids, np.array(scores_list)


def analyze_scores(filepath: str):
    """学生成绩分析主流程"""
    print("=" * 60)
    print("Part 1: 学生成绩分析")
    print("=" * 60)

    names, ids, scores = load_scores(filepath)
    n_students, n_courses = scores.shape
    print(f"学生总数: {n_students}, 课程数: {n_courses}")

    # 计算每位学生的平均分
    avg_scores = np.mean(scores, axis=1)

    # 找出平均分 >= 90 的学生
    excellent_mask = avg_scores >= 90
    excellent_students = np.where(excellent_mask)[0]
    print(f"\n--- 平均分 >= 90 的学生 ({len(excellent_students)} 人) ---")
    for idx in excellent_students:
        print(f"  {names[idx]} (学号: {ids[idx]}), 平均分: {avg_scores[idx]:.2f}")

    # 各课程统计量: 最高分, 最低分, 及格率, 平均分, 标准差
    print("\n--- 各课程统计 ---")
    course_names = ["语文", "数学", "英语"]
    for j in range(n_courses):
        col = scores[:, j]
        max_v = np.max(col)
        min_v = np.min(col)
        mean_v = np.mean(col)
        std_v = np.std(col, ddof=1)
        pass_rate = np.sum(col >= 60) / n_students * 100
        print(f"  {course_names[j]}: 最高={max_v:.1f}, 最低={min_v:.1f}, "
              f"平均={mean_v:.2f}, 标准差={std_v:.2f}, 及格率={pass_rate:.1f}%")

    # 查找重名学生
    print("\n--- 重名检测 ---")
    name_counts = Counter(names)
    duplicates = {k: v for k, v in name_counts.items() if v > 1}
    if duplicates:
        for name, count in duplicates.items():
            indices = [i for i, n in enumerate(names) if n == name]
            print(f"  姓名 '{name}' 出现 {count} 次, 索引: {indices}")
    else:
        print("  未发现重名学生")

    return names, ids, scores, avg_scores


# ============================================================
# Part 2: 成绩相关性分析
# ============================================================
def correlation_analysis(scores: np.ndarray):
    """成绩相关性分析：Pearson/Spearman 相关系数, 散点图+回归线"""
    print("\n" + "=" * 60)
    print("Part 2: 成绩相关性分析 (语文 vs 数学)")
    print("=" * 60)

    chinese = scores[:, 0]
    math = scores[:, 1]

    # Pearson 相关系数
    r_pearson, p_pearson = stats.pearsonr(chinese, math)
    # Spearman 相关系数
    r_spearman, p_spearman = stats.spearmanr(chinese, math)

    print(f"  Pearson  r = {r_pearson:.4f}, p-value = {p_pearson:.4f}")
    print(f"  Spearman r = {r_spearman:.4f}, p-value = {p_spearman:.4f}")

    # 线性回归 (最小二乘)
    slope, intercept, r_value, p_value, std_err = stats.linregress(chinese, math)

    # 散点图 + 回归线
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(chinese, math, color="steelblue", alpha=0.7, edgecolors="k", label="数据点")
    x_line = np.linspace(chinese.min() - 5, chinese.max() + 5, 100)
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=2,
            label=f"回归线 (R²={r_value**2:.4f})")
    ax.set_xlabel("语文成绩")
    ax.set_ylabel("数学成绩")
    ax.set_title("语文 vs 数学 成绩相关性")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "correlation_scatter.png"), dpi=150)
    plt.show()


# ============================================================
# Part 3: 成绩分布分析
# ============================================================
def distribution_analysis(scores: np.ndarray):
    """成绩分布分析：直方图+正态分布曲线, Q-Q图, Shapiro-Wilk检验"""
    print("\n" + "=" * 60)
    print("Part 3: 成绩分布分析 (数学)")
    print("=" * 60)

    math = scores[:, 1]

    # Shapiro-Wilk 正态性检验
    stat_sw, p_sw = stats.shapiro(math)
    print(f"  Shapiro-Wilk 统计量 = {stat_sw:.4f}, p-value = {p_sw:.4f}")
    if p_sw > 0.05:
        print("  结论: 不能拒绝正态分布假设 (alpha=0.05)")
    else:
        print("  结论: 拒绝正态分布假设 (alpha=0.05)")

    # 偏度与峰度
    skewness = stats.skew(math)
    kurtosis = stats.kurtosis(math)
    print(f"  偏度 (skewness) = {skewness:.4f}")
    print(f"  峰度 (kurtosis) = {kurtosis:.4f}")

    # 绘制: 直方图 + 正态分布曲线 + Q-Q图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 直方图 + 正态分布拟合曲线
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(math, bins=12, density=True, alpha=0.6,
                                      color="steelblue", edgecolor="k", label="样本分布")
    mu, sigma = np.mean(math), np.std(math, ddof=1)
    x_norm = np.linspace(math.min() - 5, math.max() + 5, 200)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    ax1.plot(x_norm, y_norm, "r-", linewidth=2,
             label=f"正态拟合 (μ={mu:.1f}, σ={sigma:.1f})")
    ax1.set_xlabel("数学成绩")
    ax1.set_ylabel("概率密度")
    ax1.set_title("数学成绩分布直方图 + 正态曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右: Q-Q 图
    ax2 = axes[1]
    stats.probplot(math, dist="norm", plot=ax2)
    ax2.set_title("数学成绩 Q-Q 图")
    ax2.get_lines()[0].set_markerfacecolor("steelblue")
    ax2.get_lines()[0].set_markeredgecolor("k")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "distribution_qq.png"), dpi=150)
    plt.show()


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    score_file = os.path.join(os.path.dirname(__file__), "scores.csv")

    # 生成示例成绩文件（如果不存在）
    if not os.path.exists(score_file):
        generate_score_file(score_file)

    # Part 1: 学生成绩分析
    names, ids, scores, avg_scores = analyze_scores(score_file)

    # Part 2: 成绩相关性分析
    correlation_analysis(scores)

    # Part 3: 成绩分布分析
    distribution_analysis(scores)
