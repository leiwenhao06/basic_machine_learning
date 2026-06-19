"""
支持向量机原理 (SVM Principles)
===============================
Task 1: 简化SVM — 几何中心法, 中垂线作为分类边界, 散点图+决策边界
Task 2: 近似最优间隔SVM — 角度扫描投影, 最大间隔方向, 支持向量, H1/H2
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================== 中文 / 字体设置 ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子以保证可复现
np.random.seed(42)


def generate_samples():
    """生成正负样本的各 10 个点"""
    # 正类 (label=+1): x1∈[3.5, 9.4], x2∈[7.7, 16.8]
    pos_x1 = np.random.uniform(3.5, 9.4, 10)
    pos_x2 = np.random.uniform(7.7, 16.8, 10)
    X_pos = np.column_stack([pos_x1, pos_x2])

    # 负类 (label=-1): x1∈[7.1, 19.0], x2∈[2.7, 10.3]
    neg_x1 = np.random.uniform(7.1, 19.0, 10)
    neg_x2 = np.random.uniform(2.7, 10.3, 10)
    X_neg = np.column_stack([neg_x1, neg_x2])

    return X_pos, X_neg


# ================================================================
#              Task 1: 简化SVM — 几何中心法 (中垂线)
# ================================================================
def task1_geometric_center_svm():
    """
    几何中心法:
    1) 计算各类中心 (均值)
    2) 中垂线 (perpendicular bisector of centers) 作为决策边界
    3) 分类两个测试点
    """
    print("=" * 60)
    print("Task 1: 简化SVM — 几何中心法 (中垂线)")
    print("=" * 60)

    X_pos, X_neg = generate_samples()

    # 各类中心
    center_pos = np.mean(X_pos, axis=0)
    center_neg = np.mean(X_neg, axis=0)
    print(f"正类中心: ({center_pos[0]:.4f}, {center_pos[1]:.4f})")
    print(f"负类中心: ({center_neg[0]:.4f}, {center_neg[1]:.4f})")

    # 中点
    midpoint = (center_pos + center_neg) / 2.0
    print(f"中点: ({midpoint[0]:.4f}, {midpoint[1]:.4f})")

    # 法向量 (从负类中心指向正类中心) — 这就是超平面的法向量方向
    w = center_pos - center_neg
    # 决策边界: w · (x - midpoint) = 0  =>  w·x - w·midpoint = 0
    b = -np.dot(w, midpoint)
    print(f"法向量 w: ({w[0]:.4f}, {w[1]:.4f})")
    print(f"偏置 b: {b:.4f}")

    # 分类函数
    def classify(x):
        return np.sign(np.dot(w, x) + b)

    # 测试点
    test_points = {
        'A (9.6, 11.5)': np.array([9.6, 11.5]),
        'B (17.3, 3.8)': np.array([17.3, 3.8]),
    }

    print("\n测试点分类结果:")
    for name, pt in test_points.items():
        decision = np.dot(w, pt) + b
        label = '正类' if decision > 0 else '负类'
        print(f"  {name}: 决策值={decision:.4f}, 分类={label}")

    # ---- 可视化 ----
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', marker='o',
               s=80, edgecolors='k', label='正类 (10 样本)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='red', marker='s',
               s=80, edgecolors='k', label='负类 (10 样本)')

    ax.scatter(*center_pos, c='darkblue', marker='D', s=200,
               edgecolors='k', linewidths=2, label='正类中心')
    ax.scatter(*center_neg, c='darkred', marker='D', s=200,
               edgecolors='k', linewidths=2, label='负类中心')
    ax.scatter(*midpoint, c='green', marker='X', s=150,
               edgecolors='k', linewidths=2, label='中点')

    # 连线
    ax.plot([center_neg[0], center_pos[0]], [center_neg[1], center_pos[1]],
            'k--', linewidth=1.5, alpha=0.6, label='中心连线')

    # 决策边界 (中垂线)
    x_vals = np.linspace(0, 22, 300)
    if abs(w[1]) > 1e-9:
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'g-', linewidth=2.5, label='决策边界 (中垂线)')
    else:
        x_boundary = -b / w[0]
        ax.axvline(x=x_boundary, color='g', linewidth=2.5, label='决策边界 (中垂线)')

    # 测试点
    for name, pt in test_points.items():
        ax.scatter(*pt, c='purple', marker='*', s=250,
                   edgecolors='k', linewidths=2, zorder=5)
        ax.annotate(name, pt, textcoords='offset points',
                    xytext=(10, -15), fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('简化SVM — 几何中心法 (中垂线)', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('07_svm_principles/task1_geometric_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 07_svm_principles/task1_geometric_svm.png\n")


# ================================================================
#          Task 2: 近似最优间隔SVM — 角度扫描投影法
# ================================================================
def task2_optimal_margin_svm():
    """
    角度扫描法找近似最优间隔:
    1) 对每个角度 θ∈[0,π), 定义方向向量 d=(cosθ, sinθ)
    2) 将所有样本投影到垂直于 d 的方向上 (即投影到法向量方向)
    3) 对每个 θ 计算间隔 (正负类投影区间之间的距离)
    4) 取最大间隔对应的 θ 为最优方向
    5) 找出支持向量, 绘制 H1/H2/决策边界
    """
    print("=" * 60)
    print("Task 2: 近似最优间隔SVM — 角度扫描投影法")
    print("=" * 60)

    X_pos, X_neg = generate_samples()
    X_all = np.vstack([X_pos, X_neg])
    y_all = np.hstack([np.ones(10), -np.ones(10)])

    # 角度扫描
    n_angles = 1000
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    margins = []
    best_margin = -np.inf
    best_angle = 0.0
    best_w = None

    for theta in angles:
        # 法向量方向 (超平面法向量)
        w = np.array([np.cos(theta), np.sin(theta)])
        w_norm = w / np.linalg.norm(w)

        # 投影到法向量
        projections = np.dot(X_all, w_norm)

        pos_proj = projections[y_all == 1]
        neg_proj = projections[y_all == -1]

        # 间隔 = 正类最小投影 - 负类最大投影
        margin = np.min(pos_proj) - np.max(neg_proj)
        margins.append(margin)

        if margin > best_margin:
            best_margin = margin
            best_angle = theta
            best_w = w_norm.copy()

    print(f"扫描角度数: {n_angles}")
    print(f"最优角度 θ: {best_angle:.4f} rad ({np.degrees(best_angle):.2f}°)")
    print(f"最优法向量 w: ({best_w[0]:.4f}, {best_w[1]:.4f})")
    print(f"最大间隔 (几何): {best_margin:.4f}")

    # 投影到最优方向
    projections = np.dot(X_all, best_w)
    pos_proj = projections[y_all == 1]
    neg_proj = projections[y_all == -1]

    # 支持向量
    sv_pos_idx = np.where((y_all == 1) & (projections == np.min(pos_proj)))[0]
    sv_neg_idx = np.where((y_all == -1) & (projections == np.max(neg_proj)))[0]

    sv_pos = X_all[sv_pos_idx]
    sv_neg = X_all[sv_neg_idx]
    all_sv = np.vstack([sv_pos, sv_neg])

    print(f"\n正类支持向量数: {len(sv_pos_idx)}")
    for i, idx in enumerate(sv_pos_idx):
        print(f"  SV{i+1}: ({X_all[idx, 0]:.4f}, {X_all[idx, 1]:.4f})")
    print(f"负类支持向量数: {len(sv_neg_idx)}")
    for i, idx in enumerate(sv_neg_idx):
        print(f"  SV{i+1}: ({X_all[idx, 0]:.4f}, {X_all[idx, 1]:.4f})")

    # 最优超平面: w·x + b = 0
    # b = -(min_pos + max_neg) / 2
    b = -(np.min(pos_proj) + np.max(neg_proj)) / 2.0
    print(f"\n最优超平面: w·x + b = 0,  b = {b:.4f}")

    # ---- 分类测试点 ----
    test_points = {
        'A (8.4, 10.6)': np.array([8.4, 10.6]),
        'B (15.9, 4.2)': np.array([15.9, 4.2]),
    }

    print("\n测试点分类结果:")
    for name, pt in test_points.items():
        decision = np.dot(best_w, pt) + b
        label = '正类' if decision > 0 else '负类'
        print(f"  {name}: 决策值={decision:.4f}, 分类={label}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 子图1: 角度扫描 — 间隔 vs 角度
    ax0 = axes[0]
    ax0.plot(np.degrees(angles), margins, 'b-', linewidth=1.5)
    ax0.axvline(np.degrees(best_angle), color='red', linestyle='--',
                linewidth=1.5, label=f'最优角度={np.degrees(best_angle):.2f}°')
    ax0.scatter(np.degrees(best_angle), best_margin, c='red', s=100,
                zorder=5, label=f'最大间隔={best_margin:.4f}')
    ax0.set_xlabel('角度 (度)', fontsize=12)
    ax0.set_ylabel('间隔', fontsize=12)
    ax0.set_title('角度扫描 — 间隔 vs 角度', fontsize=13)
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    # 子图2: 散点图 + 决策边界 + H1/H2
    ax1 = axes[1]
    ax1.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', marker='o',
                s=80, edgecolors='k', label='正类')
    ax1.scatter(X_neg[:, 0], X_neg[:, 1], c='red', marker='s',
                s=80, edgecolors='k', label='负类')

    # 支持向量
    ax1.scatter(all_sv[:, 0], all_sv[:, 1], c='yellow', marker='o',
                s=200, edgecolors='k', linewidths=2.5, zorder=5,
                label=f'支持向量 ({len(all_sv)})')
    # 为支持向量画圈
    for sv in all_sv:
        ax1.add_patch(plt.Circle(sv, 0.8, fill=False, color='black',
                                  linewidth=2.5, linestyle='-'))

    # 决策边界: w·x + b = 0  =>  x2 = -(w0*x1 + b) / w1
    x_vals = np.linspace(0, 22, 500)
    if abs(best_w[1]) > 1e-9:
        y_boundary = -(best_w[0] * x_vals + b) / best_w[1]
        y_h1 = -(best_w[0] * x_vals + b - best_margin / 2) / best_w[1]
        y_h2 = -(best_w[0] * x_vals + b + best_margin / 2) / best_w[1]

        ax1.plot(x_vals, y_boundary, 'g-', linewidth=3, label='决策边界 H')
        ax1.plot(x_vals, y_h1, 'k--', linewidth=2, label='H1 (正类边界)')
        ax1.plot(x_vals, y_h2, 'k-.', linewidth=2, label='H2 (负类边界)')

    # 测试点
    for name, pt in test_points.items():
        ax1.scatter(*pt, c='purple', marker='*', s=250,
                    edgecolors='k', linewidths=2, zorder=10)
        ax1.annotate(name, pt, textcoords='offset points',
                      xytext=(10, -15), fontsize=10, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='gray'))

    ax1.set_xlabel('x1', fontsize=12)
    ax1.set_ylabel('x2', fontsize=12)
    ax1.set_title(f'最优间隔SVM (间隔={best_margin:.4f})', fontsize=13)
    ax1.legend(loc='best', fontsize=8)
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, 20)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('07_svm_principles/task2_optimal_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 07_svm_principles/task2_optimal_svm.png\n")


# ================================================================
if __name__ == "__main__":
    import os
    os.makedirs('07_svm_principles', exist_ok=True)

    task1_geometric_center_svm()
    task2_optimal_margin_svm()
