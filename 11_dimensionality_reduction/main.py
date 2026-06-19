"""
11_dimensionality_reduction — 降维算法
=========================================
Task 1: PCA from scratch AND sklearn on Iris
Task 2: PCA on seeds dataset (210 samples, 7 features, 3 classes)
Task 3: 3D PCA visualization on seeds dataset
"""

import os
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===================================================================
# PCA from scratch (eigendecomposition)
# ===================================================================
class PCA_from_scratch:
    """
    PCA 实现 (基于特征分解)

    Parameters
    ----------
    n_components : int or None
        保留的主成分数量。如果为 None，保留全部。

    Attributes
    ----------
    components_ : ndarray (n_components, n_features)
        主成分方向（特征向量，按特征值降序排列）
    eigenvalues_ : ndarray (n_components,)
        对应的特征值
    explained_variance_ratio_ : ndarray (n_components,)
        各主成分的解释方差比例
    cumsum_variance_ratio_ : ndarray (n_components,)
        累计解释方差比例
    mean_ : ndarray (n_features,)
        训练数据的均值
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.cumsum_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """
        拟合 PCA 模型。

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            输入数据。
        """
        n_samples, n_features = X.shape

        # 1. 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. 协方差矩阵
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. 特征分解
        eigenvalues, eigenvectors = la.eigh(cov_matrix)

        # 4. 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. 选择前 n_components 个
        if self.n_components is not None:
            components = eigenvectors[:, : self.n_components]
            eigenvalues = eigenvalues[: self.n_components]
        else:
            components = eigenvectors

        self.components_ = components.T  # (n_components, n_features)
        self.eigenvalues_ = eigenvalues
        total_var = np.sum(np.var(X_centered, axis=0, ddof=1))
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        self.cumsum_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X):
        """将数据投影到主成分空间。"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """拟合并转换数据。"""
        self.fit(X)
        return self.transform(X)


# ===================================================================
# Generate seeds dataset
# ===================================================================
def generate_seeds(n_samples=210):
    """
    生成模拟的小麦种子数据集 (类似 UCI seeds dataset).
    3 个品种 (Kama, Rosa, Canadian), 7 个特征, 共 210 个样本.
    """
    np.random.seed(42)
    n_per_class = n_samples // 3
    total = n_per_class * 3

    def gen_class(means, stds, n):
        data = np.column_stack([
            np.random.normal(m, s, n) for m, s in zip(means, stds)
        ])
        return data

    # Kama
    X1 = gen_class([15.0, 14.5, 0.87, 5.5, 3.2, 2.5, 5.0],
                   [1.5, 1.0, 0.03, 0.4, 0.3, 0.3, 0.5], n_per_class)
    y1 = np.zeros(n_per_class, dtype=int)

    # Rosa
    X2 = gen_class([18.5, 16.0, 0.88, 6.0, 3.5, 3.5, 5.5],
                   [1.5, 1.2, 0.03, 0.5, 0.3, 0.4, 0.5], n_per_class)
    y2 = np.ones(n_per_class, dtype=int)

    # Canadian
    X3 = gen_class([12.5, 15.5, 0.90, 5.0, 2.8, 4.5, 6.0],
                   [1.5, 1.0, 0.02, 0.4, 0.3, 0.4, 0.6], n_per_class)
    y3 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    feat_names = ["Area", "Perimeter", "Compactness",
                  "KernelLength", "KernelWidth",
                  "AsymmetryCoeff", "GrooveLength"]
    class_names = ["Kama", "Rosa", "Canadian"]

    df = pd.DataFrame(X, columns=feat_names)
    df["Class"] = y
    df["ClassName"] = [class_names[i] for i in y]

    return df, class_names, feat_names


# ===================================================================
# Task 1: PCA from scratch AND sklearn on Iris
# ===================================================================
def task1_pca_iris():
    print("\n" + "=" * 70)
    print("Task 1: PCA — 从零实现 与 sklearn 对比 (Iris)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA from scratch ---
    pca_scratch = PCA_from_scratch()
    pca_scratch.fit(X_scaled)

    # --- sklearn PCA ---
    pca_sklearn = SklearnPCA()
    pca_sklearn.fit(X_scaled)

    # Compare
    print("\n--- 特征值 & 解释方差 (Scratch vs sklearn) ---")
    print(f"{'PC':<5} {'Eigenvalue(scratch)':<22} {'EVR(scratch)':<14} "
          f"{'EVR(sklearn)':<14} {'Cumulative(sc)':<14} {'Cumulative(sk)':<14}")
    print("-" * 85)
    for i in range(4):
        ev_scratch = pca_scratch.eigenvalues_[i]
        evr_scratch = pca_scratch.explained_variance_ratio_[i]
        evr_sklearn = pca_sklearn.explained_variance_ratio_[i]
        cum_scratch = pca_scratch.cumsum_variance_ratio_[i]
        cum_sklearn = np.cumsum(pca_sklearn.explained_variance_ratio_)[i]
        print(f"PC{i + 1:<3}  {ev_scratch:<22.6f} {evr_scratch:<14.4f} "
              f"{evr_sklearn:<14.4f} {cum_scratch:<14.4f} {cum_sklearn:<14.4f}")

    # --- Determine optimal n_components ---
    cumsum = pca_scratch.cumsum_variance_ratio_
    eigenvalues = pca_scratch.eigenvalues_

    n_85 = np.searchsorted(cumsum, 0.85) + 1
    n_95 = np.searchsorted(cumsum, 0.95) + 1
    n_kaiser = np.sum(eigenvalues > 1.0)

    print(f"\n--- 最佳 n_components 选取 ---")
    print(f"85% 方差阈值:   n_components = {n_85}")
    print(f"95% 方差阈值:   n_components = {n_95}")
    print(f"Kaiser 准则 (>1): n_components = {n_kaiser}")

    # --- Plots: Scree plot & Cumulative variance ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_ticks = np.arange(1, len(eigenvalues) + 1)

    # Scree plot
    axes[0].plot(x_ticks, eigenvalues, "o-", color="#4C72B0", lw=2, ms=8)
    axes[0].axhline(y=1.0, color="red", linestyle="--", label="Kaiser 准则 (λ=1)")
    axes[0].set_xlabel("主成分 (Principal Component)")
    axes[0].set_ylabel("特征值 (Eigenvalue)")
    axes[0].set_title("碎石图 (Scree Plot)")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Cumulative variance
    axes[1].bar(x_ticks, pca_scratch.explained_variance_ratio_,
                color="#55A868", alpha=0.7, label="个体解释方差")
    axes[1].plot(x_ticks, cumsum, "o-", color="#C44E52", lw=2, ms=8,
                 label="累计解释方差")
    axes[1].axhline(y=0.85, color="orange", linestyle="--", label="85% 阈值")
    axes[1].axhline(y=0.95, color="purple", linestyle="--", label="95% 阈值")
    axes[1].set_xlabel("主成分")
    axes[1].set_ylabel("解释方差比例")
    axes[1].set_title("解释方差比例 & 累计方差")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task1_pca_iris.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # Also plot 2D PCA scatter
    X_pca = pca_scratch.transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, name in enumerate(iris.target_names):
        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                   c=colors[i], label=name, alpha=0.7, edgecolors="k", s=50)
    ax.set_xlabel(f"PC1 ({pca_scratch.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_scratch.explained_variance_ratio_[1]:.2%})")
    ax.set_title("Iris PCA 二维投影 (Scratch)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task1_pca_iris_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[Task 1 完成]")


# ===================================================================
# Task 2: PCA on seeds dataset
# ===================================================================
def task2_pca_seeds():
    print("\n" + "=" * 70)
    print("Task 2: PCA — 种子数据集分析")
    print("=" * 70)

    df, class_names, feat_names = generate_seeds(210)
    X = df.drop(["Class", "ClassName"], axis=1).values
    y = df["Class"].values

    # Z-score standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = SklearnPCA()
    X_pca = pca.fit_transform(X_scaled)

    # --- PC Loadings ---
    print("\n--- 主成分载荷 (PC Loadings) ---")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feat_names,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    )
    print(loadings_df.round(4))

    print("\n--- 解释方差 ---")
    for i, (evr, cum) in enumerate(zip(
        pca.explained_variance_ratio_,
        np.cumsum(pca.explained_variance_ratio_),
    )):
        print(f"PC{i + 1}: 个体 {evr:.4f}  累计 {cum:.4f}")

    # --- 2D scatter plot colored by class ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i], label=name, alpha=0.7, edgecolors="k", s=60)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title("种子数据集 PCA 二维投影")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task2_pca_seeds_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- Loadings heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(loadings_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(loadings_df.columns)))
    ax.set_xticklabels(loadings_df.columns)
    ax.set_yticks(range(len(loadings_df.index)))
    ax.set_yticklabels(loadings_df.index)
    ax.set_title("主成分载荷热力图 (PC Loadings Heatmap)")
    plt.colorbar(im, ax=ax, label="Loading")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task2_loadings_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[Task 2 完成]")


# ===================================================================
# Task 3: 3D PCA visualization
# ===================================================================
def task3_pca_3d():
    print("\n" + "=" * 70)
    print("Task 3: 3D PCA 可视化 (种子数据集)")
    print("=" * 70)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df, class_names, feat_names = generate_seeds(210)
    X = df.drop(["Class", "ClassName"], axis=1).values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = SklearnPCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    pc1, pc2, pc3 = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

    evr = pca.explained_variance_ratio_
    cum = np.sum(evr)
    print(f"\n前 3 个主成分累计解释方差: {cum:.4f} ({cum * 100:.2f}%)")

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    markers = ["o", "s", "D"]

    # --- Three viewing angles ---
    fig = plt.figure(figsize=(18, 6))

    viewing_angles = [
        (25, 45, "视角 1 (elev=25, azim=45)"),
        (10, 120, "视角 2 (elev=10, azim=120)"),
        (60, -60, "视角 3 (elev=60, azim=-60)"),
    ]

    for idx, (elev, azim, title) in enumerate(viewing_angles):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        for i, name in enumerate(class_names):
            mask = y == i
            ax.scatter(pc1[mask], pc2[mask], pc3[mask],
                       c=colors[i], marker=markers[i], label=name,
                       alpha=0.7, s=40)
        ax.set_xlabel(f"PC1 ({evr[0]:.2%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.2%})")
        ax.set_zlabel(f"PC3 ({evr[2]:.2%})")
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task3_pca_3d_views.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- Three 2D projection planes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    projections = [
        (pc1, pc2, "PC1", "PC2", evr[0], evr[1]),
        (pc1, pc3, "PC1", "PC3", evr[0], evr[2]),
        (pc2, pc3, "PC2", "PC3", evr[1], evr[2]),
    ]

    for ax, (x_var, y_var, x_label, y_label, x_evr, y_evr) in zip(axes, projections):
        for i, name in enumerate(class_names):
            mask = y == i
            ax.scatter(x_var[mask], y_var[mask],
                       c=colors[i], marker=markers[i], label=name,
                       alpha=0.7, edgecolors="k", s=45)
        ax.set_xlabel(f"{x_label} ({x_evr:.2%})")
        ax.set_ylabel(f"{y_label} ({y_evr:.2%})")
        ax.set_title(f"{x_label} vs {y_label}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "11_dimensionality_reduction",
                             "task3_pca_2d_projections.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- Detailed PC loading analysis ---
    print("\n--- 详细 PC 载荷分析 ---")
    loadings = pd.DataFrame(
        pca.components_,
        columns=feat_names,
        index=["PC1", "PC2", "PC3"],
    )
    print(loadings.round(4))

    print("\nPC1 主要载荷 (绝对值 > 0.4):")
    for feat, val in zip(feat_names, loadings.loc["PC1"]):
        if abs(val) > 0.4:
            print(f"  {feat}: {val:.4f}")

    print("\nPC2 主要载荷 (绝对值 > 0.4):")
    for feat, val in zip(feat_names, loadings.loc["PC2"]):
        if abs(val) > 0.4:
            print(f"  {feat}: {val:.4f}")

    print("\nPC3 主要载荷 (绝对值 > 0.4):")
    for feat, val in zip(feat_names, loadings.loc["PC3"]):
        if abs(val) > 0.4:
            print(f"  {feat}: {val:.4f}")

    print("\n[Task 3 完成]")


# ===================================================================
# main
# ===================================================================
if __name__ == "__main__":
    task1_pca_iris()
    task2_pca_seeds()
    task3_pca_3d()

    print("\n" + "=" * 70)
    print("11_dimensionality_reduction 全部任务完成!")
    print("=" * 70)
