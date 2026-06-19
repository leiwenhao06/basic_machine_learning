# -*- coding: utf-8 -*-
"""
04_knn — K近邻算法

包含:
  Task 1: KNN 分类 (手动实现) — 昆虫分类, k=1/3/5 预测
  Task 2: KNN 回归 — Bodyfat 数据集, sklearn + LOOCV, 手动 KNN 回归器
  Task 3: KNN 回归 — 运输成本预测, 标准化 + 算术平均 / 距离加权, LOOCV
"""

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 中文字体设置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 通用工具：距离计算
# ============================================================
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量之间的欧几里得距离"""
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算曼哈顿距离"""
    return np.sum(np.abs(a - b))


# ============================================================
# 手动 KNN 分类器
# ============================================================
class ManualKNNClassifier:
    """手动实现的 KNN 分类器"""

    def __init__(self, k: int = 3, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X.astype(float)
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X.astype(float))
        predictions = []
        for x in X:
            # 计算到所有训练样本的距离
            if self.metric == "euclidean":
                dists = np.array([euclidean_distance(x, xt) for xt in self.X_train])
            elif self.metric == "manhattan":
                dists = np.array([manhattan_distance(x, xt) for xt in self.X_train])
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # 找出 k 个最近邻
            nearest_idx = np.argsort(dists)[:self.k]
            nearest_labels = self.y_train[nearest_idx]

            # 多数投票
            label_counts = Counter(nearest_labels)
            most_common = label_counts.most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def predict_with_distances(self, X: np.ndarray):
        """预测并返回每个测试样本的 k 个最近邻的距离和标签"""
        X = np.atleast_2d(X.astype(float))
        results = []
        for x in X:
            dists = np.array([euclidean_distance(x, xt) for xt in self.X_train])
            nearest_idx = np.argsort(dists)[:self.k]
            results.append({
                "distances": dists[nearest_idx],
                "labels": self.y_train[nearest_idx],
                "indices": nearest_idx,
            })
        return results


# ============================================================
# Task 1: KNN 分类 — 昆虫分类
# ============================================================
def task1_knn_insect_classification():
    print("=" * 60)
    print("Task 1: KNN 分类 — 昆虫分类")
    print("=" * 60)

    # 生成训练数据
    np.random.seed(0)
    # A 类昆虫: 触角长度 1.1~1.3, 翅膀长度 1.8~2.3
    class_A = np.array([
        [1.10, 2.30], [1.15, 2.10], [1.20, 2.00], [1.25, 1.95],
        [1.28, 1.85], [1.30, 1.80],
    ])
    # B 类昆虫: 触角长度 1.3~1.5, 翅膀长度 1.5~1.7
    class_B = np.array([
        [1.30, 1.70], [1.35, 1.65], [1.40, 1.60], [1.45, 1.58],
        [1.48, 1.53], [1.50, 1.50],
    ])

    X_train = np.vstack([class_A, class_B])
    y_train = np.array(["A"] * 6 + ["B"] * 6)

    print(f"训练样本: A类={len(class_A)} 个, B类={len(class_B)} 个")
    print(f"特征: 触角长度, 翅膀长度")

    # 测试样本
    test_point = np.array([1.2, 1.8])

    # 用不同的 k 值预测
    ks = [1, 3, 5]
    for k in ks:
        knn = ManualKNNClassifier(k=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(test_point.reshape(1, -1))[0]

        # 获取距离信息
        result = knn.predict_with_distances(test_point.reshape(1, -1))[0]
        print(f"\n  k={k}: 预测类别 = {pred}")
        for i in range(k):
            idx = result["indices"][i]
            print(f"    第{i+1}近邻: 样本{idx} (类别={result['labels'][i]}), "
                  f"坐标={X_train[idx]}, 距离={result['distances'][i]:.4f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制训练样本
    ax.scatter(class_A[:, 0], class_A[:, 1], c="steelblue", marker="o", s=100,
               edgecolors="k", label="A类昆虫")
    ax.scatter(class_B[:, 0], class_B[:, 1], c="darkorange", marker="s", s=100,
               edgecolors="k", label="B类昆虫")

    # 绘制测试点
    ax.scatter(*test_point, c="red", marker="*", s=300, edgecolors="k",
               zorder=10, label="测试点 (1.2, 1.8)")

    # 绘制不同 k 值范围
    for k, color, style in zip([1, 3, 5], ["red", "green", "purple"],
                                ["-", "--", "-."]):
        knn = ManualKNNClassifier(k=k)
        knn.fit(X_train, y_train)
        result = knn.predict_with_distances(test_point.reshape(1, -1))[0]

        # 画圆 (以测试点为中心, 最远邻居距离为半径)
        radius = result["distances"][-1]
        circle = plt.Circle(test_point, radius, fill=False, color=color,
                            linestyle=style, linewidth=1.8,
                            label=f"k={k} 范围 (r={radius:.3f})")
        ax.add_patch(circle)

    ax.set_xlabel("触角长度")
    ax.set_ylabel("翅膀长度")
    ax.set_title("KNN 昆虫分类 — 不同 k 值决策范围")
    ax.legend(loc="lower left")
    ax.set_xlim(1.0, 1.6)
    ax.set_ylim(1.4, 2.4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task1_insect_knn.png"), dpi=150)
    plt.show()


# ============================================================
# 手动 KNN 回归器
# ============================================================
class ManualKNNRegressor:
    """手动实现的 KNN 回归器"""

    def __init__(self, k: int = 3, weighted: bool = False):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X.astype(float)
        self.y_train = y.astype(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X.astype(float))
        predictions = []
        for x in X:
            dists = np.array([euclidean_distance(x, xt) for xt in self.X_train])
            nearest_idx = np.argsort(dists)[:self.k]
            nearest_dists = dists[nearest_idx]
            nearest_vals = self.y_train[nearest_idx]

            if self.weighted:
                # 距离加权: 权重 = 1/distance, 避免除零
                weights = 1.0 / (nearest_dists + 1e-10)
                pred = np.sum(weights * nearest_vals) / np.sum(weights)
            else:
                pred = np.mean(nearest_vals)
            predictions.append(pred)
        return np.array(predictions)


# ============================================================
# Task 2: KNN 回归 — Bodyfat 数据集
# ============================================================
def task2_bodyfat_knn_regression():
    print("\n" + "=" * 60)
    print("Task 2: KNN 回归 — Bodyfat 数据集")
    print("=" * 60)

    # 生成模拟 bodyfat 数据 (近似真实数据分布)
    np.random.seed(123)
    n = 50
    # 特征: abdomen circumference (腹围, cm)
    abdomen = np.random.normal(92, 9, n)
    # 目标: bodyfat percentage (体脂率, %)
    # bodyfat ≈ 0.6 * abdomen - 30 + noise
    bodyfat = 0.6 * abdomen - 30 + np.random.normal(0, 3, n)
    # 限制范围
    bodyfat = np.clip(bodyfat, 5, 45)
    abdomen = np.clip(abdomen, 70, 120)

    X = abdomen.reshape(-1, 1)
    y = bodyfat

    print(f"样本数: {n}")
    print(f"特征 (腹围): 均值={abdomen.mean():.2f}, 标准差={abdomen.std():.2f}")
    print(f"目标 (体脂率): 均值={bodyfat.mean():.2f}, 标准差={bodyfat.std():.2f}")
    print(f"相关系数 (腹围 vs 体脂率): {np.corrcoef(abdomen, bodyfat)[0,1]:.4f}")

    # Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()

    results = {}
    ks = [1, 3, 5, 7, 10]

    for k in ks:
        # --- sklearn ---
        sk_model = KNeighborsRegressor(n_neighbors=k)
        y_preds_sk = []
        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            sk_model.fit(X_tr, y_tr)
            y_preds_sk.append(sk_model.predict(X_te)[0])
        y_preds_sk = np.array(y_preds_sk)
        mse_sk = mean_squared_error(y, y_preds_sk)
        r2_sk = r2_score(y, y_preds_sk)

        # --- 手动实现 ---
        manual_model = ManualKNNRegressor(k=k, weighted=False)
        y_preds_manual = []
        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            manual_model.fit(X_tr, y_tr)
            y_preds_manual.append(manual_model.predict(X_te)[0])
        y_preds_manual = np.array(y_preds_manual)
        mse_manual = mean_squared_error(y, y_preds_manual)
        r2_manual = r2_score(y, y_preds_manual)

        results[k] = {
            "sklearn": {"MSE": mse_sk, "R2": r2_sk},
            "manual": {"MSE": mse_manual, "R2": r2_manual},
        }
        print(f"\n  k={k:2d}  sklearn: MSE={mse_sk:.4f}, R²={r2_sk:.4f}  |  "
              f"手动: MSE={mse_manual:.4f}, R²={r2_manual:.4f}")

    # 找出最优 k
    best_k = min(results, key=lambda k: results[k]["sklearn"]["MSE"])
    print(f"\n  最优 k = {best_k} (sklearn MSE 最小: {results[best_k]['sklearn']['MSE']:.4f})")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 不同 k 的 MSE
    ax1 = axes[0]
    k_list = list(results.keys())
    mse_sk = [results[k]["sklearn"]["MSE"] for k in k_list]
    mse_manual = [results[k]["manual"]["MSE"] for k in k_list]
    ax1.plot(k_list, mse_sk, "o-", color="steelblue", linewidth=2, markersize=8,
             label="sklearn (LOOCV MSE)")
    ax1.plot(k_list, mse_manual, "s--", color="darkorange", linewidth=2, markersize=8,
             label="手动实现 (LOOCV MSE)")
    ax1.set_xlabel("k 值")
    ax1.set_ylabel("MSE")
    ax1.set_title("Bodyfat KNN回归: 不同 k 值的 LOOCV MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右: 最优 k 的预测 vs 真实值
    ax2 = axes[1]
    best_model = KNeighborsRegressor(n_neighbors=best_k)
    best_model.fit(X, y)
    y_pred_best = best_model.predict(X)
    ax2.scatter(y, y_pred_best, c="steelblue", edgecolors="k", s=50, alpha=0.7)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2,
             label="理想拟合线")
    ax2.set_xlabel("真实体脂率 (%)")
    ax2.set_ylabel("预测体脂率 (%)")
    ax2.set_title(f"Bodyfat: k={best_k} 预测 vs 真实值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task2_bodyfat_knn.png"), dpi=150)
    plt.show()


# ============================================================
# Task 3: KNN 回归 — 运输成本预测
# ============================================================
def task3_shipping_cost():
    print("\n" + "=" * 60)
    print("Task 3: KNN 回归 — 运输成本预测")
    print("=" * 60)

    # 10 个样本: (体积 m³, 重量 ton, 成本 千元)
    raw_data = np.array([
        [2.0, 1.5, 180],
        [3.0, 2.0, 220],
        [4.0, 2.8, 280],
        [5.0, 3.5, 350],
        [2.5, 1.8, 200],
        [3.5, 2.2, 250],
        [4.5, 3.0, 310],
        [5.5, 3.8, 380],
        [6.0, 4.0, 410],
        [1.5, 1.0, 150],
    ])
    X_raw = raw_data[:, :2]   # 体积, 重量
    y_raw = raw_data[:, 2]     # 成本

    print(f"训练样本: {len(X_raw)} 个")
    print(f"特征: 体积 (m³), 重量 (ton)")
    print(f"目标: 运输成本 (千元)")

    # 测试样本
    test_raw = np.array([3.8, 3.2])

    # --- 数据标准化 ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()
    test_scaled = scaler_X.transform(test_raw.reshape(1, -1))

    # 用不同 k 在原始标准化数据上进行算术平均和距离加权预测
    ks = [1, 3, 5]

    for k in ks:
        print(f"\n--- k = {k} ---")

        # 算术平均
        knn_avg = ManualKNNRegressor(k=k, weighted=False)
        knn_avg.fit(X_scaled, y_scaled)
        pred_scaled_avg = knn_avg.predict(test_scaled.reshape(1, -1))[0]
        pred_avg = scaler_y.inverse_transform([[pred_scaled_avg]])[0, 0]

        # 距离加权
        knn_w = ManualKNNRegressor(k=k, weighted=True)
        knn_w.fit(X_scaled, y_scaled)
        pred_scaled_w = knn_w.predict(test_scaled.reshape(1, -1))[0]
        pred_w = scaler_y.inverse_transform([[pred_scaled_w]])[0, 0]

        print(f"  算术平均预测成本: {pred_avg:.2f} 千元")
        print(f"  距离加权预测成本: {pred_w:.2f} 千元")

        # 显示最近邻详细信息
        dists = np.array([euclidean_distance(test_scaled.ravel(), xt)
                          for xt in X_scaled])
        nearest_idx = np.argsort(dists)[:k]
        print(f"  {k} 个最近邻 (标准化空间距离, 原始成本):")
        for i, idx in enumerate(nearest_idx):
            print(f"    第{i+1}近邻: 样本{idx} (体积={X_raw[idx,0]:.1f}, "
                  f"重量={X_raw[idx,1]:.1f}), 成本={y_raw[idx]:.0f}千元, "
                  f"距离={dists[idx]:.4f}")

    # LOOCV 评估
    print(f"\n--- LOOCV 评估 (标准化数据) ---")
    loo = LeaveOneOut()
    for k in [1, 3, 5]:
        knn_loocv = ManualKNNRegressor(k=k, weighted=False)
        y_preds = []
        for train_idx, test_idx in loo.split(X_scaled):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y_scaled[train_idx], y_scaled[test_idx]
            knn_loocv.fit(X_tr, y_tr)
            y_preds.append(knn_loocv.predict(X_te)[0])
        y_preds = np.array(y_preds)
        # 还原到原始尺度
        y_preds_raw = scaler_y.inverse_transform(y_preds.reshape(-1, 1)).ravel()
        y_raw_copy = y_raw
        mse_loocv = mean_squared_error(y_raw_copy, y_preds_raw)
        print(f"  k={k}: LOOCV MSE = {mse_loocv:.2f}, RMSE = {np.sqrt(mse_loocv):.2f}")

    # 可视化: 原始数据 + 测试点
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 3D-like 散点 (体积 vs 重量, 颜色=成本)
    ax1 = axes[0]
    scatter = ax1.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw, cmap="viridis",
                          s=150, edgecolors="k", zorder=5)
    ax1.scatter(*test_raw, c="red", marker="*", s=300, edgecolors="k",
                zorder=10, label="测试点 (3.8, 3.2)")
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("运输成本 (千元)")
    ax1.set_xlabel("体积 (m³)")
    ax1.set_ylabel("重量 (ton)")
    ax1.set_title("运输成本数据分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右: 标准化空间中的样本 (用颜色区分成本)
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_raw, cmap="viridis",
                           s=150, edgecolors="k", zorder=5)
    ax2.scatter(test_scaled[0, 0], test_scaled[0, 1], c="red", marker="*",
                s=300, edgecolors="k", zorder=10, label="测试点 (标准化)")
    # 标注每个点的索引
    for i in range(len(X_scaled)):
        ax2.annotate(str(i), (X_scaled[i, 0], X_scaled[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("运输成本 (千元)")
    ax2.set_xlabel("标准化体积")
    ax2.set_ylabel("标准化重量")
    ax2.set_title("标准化空间中的样本分布")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task3_shipping_knn.png"), dpi=150)
    plt.show()


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    task1_knn_insect_classification()
    task2_bodyfat_knn_regression()
    task3_shipping_cost()
