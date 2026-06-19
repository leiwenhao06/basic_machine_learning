# -*- coding: utf-8 -*-
"""
02_linear_regression — 线性回归

包含:
  1. 最小二乘法从零实现 (正规方程)
  2. 房屋面积 vs 价格 — 线性回归
  3. 身高 vs 体重 — 线性回归
  4. 与 sklearn LinearRegression 对比
"""

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 中文字体设置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 最小二乘线性回归（从零实现）
# ============================================================
class LeastSquaresLinearRegression:
    """使用正规方程 (Normal Equation) 的最小二乘线性回归"""

    def __init__(self):
        self.coef_ = None      # 斜率 (单个特征时) 或 系数向量
        self.intercept_ = None  # 截距

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        正规方程: theta = (X^T X)^{-1} X^T y
        X 包含一列全1作为截距项
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置列
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """返回 R^2 决定系数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# ============================================================
# 辅助绘图函数
# ============================================================
def plot_regression_result(X, y, model, xlabel, ylabel, title,
                           manual_model=None, save_name=None):
    """绘制散点图 + 回归线, 支持同时展示手动实现与 sklearn 结果"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X, y, color="steelblue", alpha=0.7, edgecolors="k", s=60, label="数据点")

    x_line = np.linspace(X.min() - (X.max() - X.min()) * 0.1,
                         X.max() + (X.max() - X.min()) * 0.1, 200).reshape(-1, 1)

    # sklearn 预测
    y_pred_sk = model.predict(x_line)
    r2_sk = model.score(X, y)
    ax.plot(x_line, y_pred_sk, "r-", linewidth=2.5,
            label=f"sklearn (R²={r2_sk:.4f})")

    # 手动实现预测
    if manual_model is not None:
        y_pred_manual = manual_model.predict(x_line)
        r2_manual = manual_model.score(X, y)
        ax.plot(x_line, y_pred_manual, "g--", linewidth=2.5,
                label=f"手动实现 (R²={r2_manual:.4f})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        save_path = os.path.join(os.path.dirname(__file__), save_name)
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# Task 1: 房屋面积 vs 价格
# ============================================================
def task_house_price():
    print("=" * 60)
    print("Task 1: 房屋面积 vs 价格 — 线性回归")
    print("=" * 60)

    # 数据
    area = np.array([100, 120, 150, 200, 220, 250, 300, 350, 400, 450, 500, 600],
                    dtype=float).reshape(-1, 1)
    price = np.array([570, 610, 645, 745, 798, 845, 942, 1145, 1370, 1545, 1730, 1880],
                     dtype=float)

    # --- 手动实现 ---
    manual_lr = LeastSquaresLinearRegression()
    manual_lr.fit(area, price)
    slope = manual_lr.coef_[0]
    intercept = manual_lr.intercept_
    y_pred_manual = manual_lr.predict(area)
    r2_manual = manual_lr.score(area, price)
    mse_manual = mean_squared_error(price, y_pred_manual)

    print(f"\n[手动实现] 回归方程: price = {intercept:.2f} + {slope:.4f} × area")
    print(f"  R² = {r2_manual:.4f}")
    print(f"  MSE = {mse_manual:.2f}")

    # 预测一个新房面积
    test_area = np.array([[380]])
    pred_price = manual_lr.predict(test_area)[0]
    print(f"  预测 380 sqm 房价: {pred_price:.2f}")

    # --- sklearn ---
    sk_lr = LinearRegression()
    sk_lr.fit(area, price)
    print(f"\n[sklearn] 回归方程: price = {sk_lr.intercept_:.2f} + {sk_lr.coef_[0]:.4f} × area")
    print(f"  R² = {sk_lr.score(area, price):.4f}")

    # 绘图
    plot_regression_result(area, price, sk_lr,
                           xlabel="房屋面积 (平方米)",
                           ylabel="房屋价格 (千元)",
                           title="房屋面积 vs 价格 线性回归",
                           manual_model=manual_lr,
                           save_name="house_price_regression.png")


# ============================================================
# Task 2: 身高 vs 体重
# ============================================================
def task_height_weight():
    print("\n" + "=" * 60)
    print("Task 2: 身高 vs 体重 — 线性回归")
    print("=" * 60)

    # 生成15个样本
    np.random.seed(123)
    height = np.linspace(155, 190, 15).reshape(-1, 1)  # cm
    # 体重 ≈ 身高 * 0.9 - 95 + 噪声
    weight = height.ravel() * 0.9 - 95 + np.random.normal(0, 3, 15)

    # --- 手动实现 ---
    manual_lr = LeastSquaresLinearRegression()
    manual_lr.fit(height, weight)
    slope = manual_lr.coef_[0]
    intercept = manual_lr.intercept_
    y_pred_manual = manual_lr.predict(height)
    r2_manual = manual_lr.score(height, weight)
    mse_manual = mean_squared_error(weight, y_pred_manual)

    print(f"\n[手动实现] 回归方程: weight = {intercept:.2f} + {slope:.4f} × height")
    print(f"  R² = {r2_manual:.4f}")
    print(f"  MSE = {mse_manual:.2f}")

    # --- sklearn ---
    sk_lr = LinearRegression()
    sk_lr.fit(height, weight)
    print(f"\n[sklearn] 回归方程: weight = {sk_lr.intercept_:.2f} + {sk_lr.coef_[0]:.4f} × height")
    print(f"  R² = {sk_lr.score(height, weight):.4f}")

    # 预测
    test_h = np.array([[175]])
    pred_w_manual = manual_lr.predict(test_h)[0]
    pred_w_sk = sk_lr.predict(test_h)[0]
    print(f"\n  预测 175cm 体重: 手动 {pred_w_manual:.2f} kg, sklearn {pred_w_sk:.2f} kg")

    # 绘图
    plot_regression_result(height, weight, sk_lr,
                           xlabel="身高 (cm)",
                           ylabel="体重 (kg)",
                           title="身高 vs 体重 线性回归",
                           manual_model=manual_lr,
                           save_name="height_weight_regression.png")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    task_house_price()
    task_height_weight()
