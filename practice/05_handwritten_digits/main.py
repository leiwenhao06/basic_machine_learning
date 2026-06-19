"""
手写数字识别 — Handwritten Digit Recognition
============================================
使用sklearn digits数据集 (8x8, 10类 0-9)，分别使用KNN、逻辑回归（Softmax）、
决策树进行手写数字识别，并对决策树模型实施对抗攻击实验。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load digits dataset once at module level
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# ===================================================================
# Task 1: Data Exploration
# ===================================================================
def task1_data_exploration():
    """Explore the digits dataset: keys, class distribution, sample visualisation."""
    print("=" * 60)
    print("Task 1: 数据探索 Data Exploration")
    print("=" * 60)

    # Dataset keys
    print(f"\n数据集键 Dataset keys: {list(digits.keys())}")
    print(f"特征矩阵形状: {digits.data.shape}")
    print(f"标签形状:      {digits.target.shape}")
    print(f"图像形状:      {digits.images.shape}")
    print(f"类别标签:      {np.unique(digits.target)}")

    # Class distribution statistics
    print(f"\n类别分布 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  数字 {u}: {c} ({100 * c / len(y):.2f}%)")
    print(f"  总计: {len(y)}, 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

    # Randomly select 5 samples
    rng = np.random.default_rng(123)
    random_indices = rng.choice(len(digits.images), size=5, replace=False)

    fig, axes = plt.subplots(5, 3, figsize=(8, 14))
    axes[0, 0].set_title('2D Matrix (8x8)', fontsize=10)
    axes[0, 1].set_title('1D Vector (64,)', fontsize=10)
    axes[0, 2].set_title('Image (gray_r)', fontsize=10)

    for row_idx, sample_idx in enumerate(random_indices):
        img_2d = digits.images[sample_idx]
        vec_1d = digits.data[sample_idx]
        label = digits.target[sample_idx]

        # 2D matrix as text
        ax_mat = axes[row_idx, 0]
        ax_mat.axis('off')
        mat_str = '\n'.join(
            ' '.join(f'{v:2.0f}' for v in row) for row in img_2d
        )
        ax_mat.text(0.5, 0.5, mat_str, transform=ax_mat.transAxes,
                    fontsize=6, family='monospace', ha='center', va='center')
        ax_mat.set_title(f'Label={label} (2D)', fontsize=9)

        # 1D vector as text
        ax_vec = axes[row_idx, 1]
        ax_vec.axis('off')
        vec_str = '[' + ', '.join(f'{v:.0f}' for v in vec_1d[:16]) + ',\n'
        vec_str += ', '.join(f'{v:.0f}' for v in vec_1d[16:32]) + ',\n'
        vec_str += ', '.join(f'{v:.0f}' for v in vec_1d[32:48]) + ',\n'
        vec_str += ', '.join(f'{v:.0f}' for v in vec_1d[48:]) + ']'
        ax_vec.text(0.5, 0.5, vec_str, transform=ax_vec.transAxes,
                    fontsize=5.5, family='monospace', ha='center', va='center')
        ax_vec.set_title(f'Label={label} (1D)', fontsize=9)

        # Image
        axes[row_idx, 2].imshow(img_2d, cmap='gray_r')
        axes[row_idx, 2].set_title(f'Label={label}', fontsize=9)
        axes[row_idx, 2].axis('off')

    plt.suptitle('随机5个样本 Random 5 Samples', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/01_random_samples.png', dpi=150)
    plt.close()
    print("[图表已保存] 01_random_samples.png")

    # First sample with label 7
    idx_7 = np.where(digits.target == 7)[0][0]
    img_7 = digits.images[idx_7]
    vec_7 = digits.data[idx_7]
    print(f"\n第一个标签为7的样本 (索引 {idx_7}):")
    print("  2D 矩阵:")
    for row in img_7:
        print('    ' + ' '.join(f'{v:2.0f}' for v in row))
    print(f"  1D 向量 (前32位): [{', '.join(f'{v:.0f}' for v in vec_7[:32])} ...]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    ax1.imshow(img_7, cmap='gray_r')
    ax1.set_title(f'第一个Label=7的样本\n索引 Index={idx_7}')
    ax1.axis('off')

    # Pixel value heatmap
    im = ax2.imshow(img_7, cmap='coolwarm', vmin=0, vmax=16)
    ax2.set_title('像素值热力图 Pixel Heatmap')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/02_first_label7.png', dpi=150)
    plt.close()
    print("[图表已保存] 02_first_label7.png")


# ===================================================================
# Task 2: KNN with GridSearchCV
# ===================================================================
def task2_knn():
    """KNN classifier with GridSearchCV over k values."""
    print("\n" + "=" * 60)
    print("Task 2: KNN分类器与网格搜索 KNN with GridSearchCV")
    print("=" * 60)

    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"\n最佳 k: {grid.best_params_['n_neighbors']}")
    print(f"最佳交叉验证准确率: {grid.best_score_:.4f}")

    # CV results
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    ks = param_grid['n_neighbors']

    print("\nk值 vs 准确率:")
    for k, mean, std in zip(ks, means, stds):
        print(f"  k={k:2d}: accuracy={mean:.4f} +/- {std:.4f}")

    # Plot k vs accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ks, means, yerr=stds, marker='o', linewidth=2,
                markersize=8, capsize=5, color='#2c3e50')
    ax.axvline(x=grid.best_params_['n_neighbors'], color='red',
               linestyle='--', alpha=0.6, label=f"Best k={grid.best_params_['n_neighbors']}")
    ax.set_xlabel('k (邻居数量 Number of Neighbors)')
    ax.set_ylabel('准确率 Accuracy (5-fold CV)')
    ax.set_title('KNN: k值与准确率关系\nKNN: k vs Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/03_knn_k_vs_accuracy.png', dpi=150)
    plt.close()
    print("[图表已保存] 03_knn_k_vs_accuracy.png")

    # Train best model
    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率 Test Accuracy: {test_acc:.4f}")
    print("\n分类报告 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Misclassified samples
    mis_idx = np.where(y_pred != y_test)[0]
    print(f"\n误分类样本数: {len(mis_idx)} / {len(y_test)}")
    print(f"误分类率: {len(mis_idx) / len(y_test):.4f}")

    # Display first 10 misclassified images
    n_show = min(10, len(mis_idx))
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes

    for i in range(n_show):
        test_idx = mis_idx[i]
        img = X_test[test_idx].reshape(8, 8)
        axes[i].imshow(img, cmap='gray_r')
        axes[i].set_title(f'True: {y_test[test_idx]}  Pred: {y_pred[test_idx]}',
                          fontsize=9)
        axes[i].axis('off')

    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('KNN误分类样本 Misclassified Samples (True vs Predicted)', fontsize=13)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/04_knn_misclassified.png', dpi=150)
    plt.close()
    print("[图表已保存] 04_knn_misclassified.png")


# ===================================================================
# Task 3: Logistic Regression (Softmax)
# ===================================================================
def task3_logistic_regression():
    """Multinomial logistic regression with per-class metrics and weight heatmaps."""
    print("\n" + "=" * 60)
    print("Task 3: 逻辑回归（Softmax） Logistic Regression")
    print("=" * 60)

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                            max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率 Test Accuracy: {test_acc:.4f}")

    # Per-class precision, recall, F1
    print("\n各类别详细指标 Per-class Metrics:")
    prec = precision_score(y_test, y_pred, average=None)
    rec = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    print(f"{'Digit':>6s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
    print("-" * 42)
    for d in range(10):
        print(f"{d:6d}  {prec[d]:10.4f}  {rec[d]:10.4f}  {f1[d]:10.4f}")

    print(f"\nMacro Avg  - Precision: {prec.mean():.4f}, "
          f"Recall: {rec.mean():.4f}, F1: {f1.mean():.4f}")

    print("\n完整分类报告 Full Classification Report:")
    print(classification_report(y_test, y_pred))

    # Weight heatmaps for digits 0 and 1
    # LogisticRegression coef_ shape: (n_classes, n_features) = (10, 64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, digit in enumerate([0, 1]):
        weights = lr.coef_[digit].reshape(8, 8)
        im = axes[i].imshow(weights, cmap='coolwarm', center=0)
        axes[i].set_title(f'数字 {digit} 的权重热力图\nWeight Heatmap for Digit {digit}')
        axes[i].axis('off')
        # Annotate each pixel with weight value
        for r in range(8):
            for c in range(8):
                axes[i].text(c, r, f'{weights[r, c]:.2f}',
                             ha='center', va='center', fontsize=6,
                             color='black' if abs(weights[r, c]) < 1.5 else 'white')
        plt.colorbar(im, ax=axes[i], shrink=0.75)

    plt.suptitle('逻辑回归权重可视化 (Softmax Weights)', fontsize=13)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/05_lr_weight_heatmaps.png', dpi=150)
    plt.close()
    print("[图表已保存] 05_lr_weight_heatmaps.png")


# ===================================================================
# Task 4: Decision Tree + Adversarial Attack
# ===================================================================
def task4_decision_tree():
    """Decision tree classifier and adversarial attack on important pixels."""
    print("\n" + "=" * 60)
    print("Task 4: 决策树与对抗攻击 Decision Tree & Adversarial Attack")
    print("=" * 60)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n决策树测试准确率 Decision Tree Test Accuracy: {test_acc:.4f}")

    # Top 3 most important pixel indices
    importances = dt.feature_importances_
    top3_idx = np.argsort(importances)[-3:][::-1]  # descending
    print(f"\n最重要的3个像素索引 Top 3 Pixel Indices:")
    for rank, idx in enumerate(top3_idx, 1):
        row, col = idx // 8, idx % 8
        print(f"  #{rank}: index={idx} (row={row}, col={col}), importance={importances[idx]:.6f}")

    # Find the first correctly predicted sample of digit 6 (or another digit if not found)
    # Get original indices from X_test
    correct_mask = (y_pred == y_test)
    digit6_mask = (y_test == 6)

    if np.any(correct_mask & digit6_mask):
        target_idx = np.where(correct_mask & digit6_mask)[0][0]
    else:
        # Fallback: any correctly predicted sample
        target_idx = np.where(correct_mask)[0][0]

    original_vec = X_test[target_idx].copy().astype(float)
    original_img = original_vec.reshape(8, 8)
    original_label = y_test[target_idx]
    original_pred = y_pred[target_idx]

    print(f"\n选择样本: X_test[{target_idx}], 真实标签={original_label}, "
          f"原始预测={original_pred}")

    # Adversarial attack: flip values at top 3 pixel positions
    attacked_vec = original_vec.copy()
    for idx in top3_idx:
        attacked_vec[idx] = 16.0 - attacked_vec[idx]

    attacked_img = attacked_vec.reshape(8, 8)
    attacked_pred = dt.predict(attacked_vec.reshape(1, -1))[0]
    print(f"攻击后预测: {attacked_pred}")
    print(f"攻击成功: {'是 YES' if attacked_pred != original_pred else '否 NO'}")

    # Highlight map: mask showing top 3 pixels
    highlight = np.zeros((8, 8))
    for idx in top3_idx:
        r, c = idx // 8, idx % 8
        highlight[r, c] = 1

    # Plot: original + highlighted, attacked + highlighted
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Original image
    axes[0, 0].imshow(original_img, cmap='gray_r')
    axes[0, 0].set_title(f'原始图像 Original\nTrue={original_label}, Pred={original_pred}')
    axes[0, 0].axis('off')

    # Original with highlighted pixels
    axes[0, 1].imshow(original_img, cmap='gray_r')
    # Overlay red circles on important pixels
    for idx in top3_idx:
        r, c = idx // 8, idx % 8
        axes[0, 1].plot(c, r, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)
    axes[0, 1].set_title(f'原始图像 + 敏感像素位置\nTop 3 Important Pixels Highlighted')
    axes[0, 1].axis('off')

    # Attacked image
    axes[1, 0].imshow(attacked_img, cmap='gray_r')
    axes[1, 0].set_title(f'对抗攻击后图像 Attacked\nPred={attacked_pred}' +
                         (' (CHANGED!)' if attacked_pred != original_pred else ' (Unchanged)'))
    axes[1, 0].axis('off')

    # Attacked with highlighted pixels
    axes[1, 1].imshow(attacked_img, cmap='gray_r')
    for idx in top3_idx:
        r, c = idx // 8, idx % 8
        axes[1, 1].plot(c, r, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2)
    axes[1, 1].set_title(f'对抗样本 + 敏感像素位置\nAttacked + Highlighted Pixels')
    axes[1, 1].axis('off')

    plt.suptitle('决策树对抗攻击实验 Adversarial Attack on Decision Tree', fontsize=14)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/06_adversarial_attack.png', dpi=150)
    plt.close()
    print("[图表已保存] 06_adversarial_attack.png")

    # Feature importance bar chart (top 10)
    top10 = np.argsort(importances)[-10:][::-1]
    top10_imp = importances[top10]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), top10_imp, color='#e67e22')
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{i}' for i in top10])
    ax.set_xlabel('像素索引 Pixel Index')
    ax.set_ylabel('重要性 Importance')
    ax.set_title('决策树像素重要性 Top 10 Pixel Importances')
    for bar, val in zip(bars, top10_imp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig('practice/05_handwritten_digits/07_dt_feature_importance.png', dpi=150)
    plt.close()
    print("[图表已保存] 07_dt_feature_importance.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("手写数字识别 Handwritten Digit Recognition")
    print("=" * 60)
    print(f"数据集: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    task1_data_exploration()
    task2_knn()
    task3_logistic_regression()
    task4_decision_tree()

    print("\n" + "=" * 60)
    print("所有任务完成 All Tasks Completed")
    print("=" * 60)
