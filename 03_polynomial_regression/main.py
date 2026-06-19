# -*- coding: utf-8 -*-
"""
03_polynomial_regression — 多项式回归

包含:
  Task 1: 模拟二次数据拟合，最小二乘法对比系数
  Task 2: GDP 预测 (2004-2023 数据, 二次多项式回归, 预测 2024)
  Task 3: 时间-浓度回归 (对比线性与二次多项式, RMSE 与 R² 评估)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 中文字体设置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 多项式回归 (从零最小二乘实现)
# ============================================================
class PolynomialRegressionLS:
    """基于正规方程的多项式回归"""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coef_ = None        # 从低次到高次: [a0, a1, ..., ad]
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_poly = self.poly_features.fit_transform(X)   # (n, degree+1)
        # 正规方程
        theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        self.coef_ = theta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self.poly_features.transform(X)
        return X_poly @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# ============================================================
# Task 1: 模拟二次数据拟合
# ============================================================
def task1_simulated_quadratic():
    print("=" * 60)
    print("Task 1: 模拟二次数据拟合")
    print("=" * 60)

    # 真实模型: y = 2 + 0.1*x + 0.9*x^2
    np.random.seed(42)
    n = 80
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y_true_coef = np.array([2.0, 0.1, 0.9])  # [a0, a1, a2]
    y = y_true_coef[0] + y_true_coef[1] * X.ravel() + y_true_coef[2] * X.ravel() ** 2
    noise = np.random.normal(0, 0.4, n)
    y_noisy = y + noise

    # 拟合二次多项式
    model = PolynomialRegressionLS(degree=2)
    model.fit(X, y_noisy)

    print(f"  真实系数: a0={y_true_coef[0]:.4f}, a1={y_true_coef[1]:.4f}, a2={y_true_coef[2]:.4f}")
    print(f"  拟合系数: a0={model.coef_[0]:.4f}, a1={model.coef_[1]:.4f}, a2={model.coef_[2]:.4f}")

    r2 = model.score(X, y_noisy)
    rmse = np.sqrt(mean_squared_error(y_noisy, model.predict(X)))
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}")

    # 绘图
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X, y_noisy, alpha=0.5, color="steelblue", edgecolors="k",
               s=30, label="带噪声样本点")
    ax.plot(X, y, "g-", linewidth=2, label="真实曲线")
    ax.plot(X, y_pred, "r--", linewidth=2, label=f"拟合曲线 (R²={r2:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("二次多项式拟合 y = 2 + 0.1x + 0.9x² + noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task1_quadratic_fit.png"), dpi=150)
    plt.show()


# ============================================================
# Task 2: GDP 预测
# ============================================================
def task2_gdp_prediction():
    print("\n" + "=" * 60)
    print("Task 2: GDP 预测 — 二次多项式回归")
    print("=" * 60)

    # 中国 GDP 数据 (2004-2023, 单位: 万亿元人民币)
    # 来源参考: 国家统计局 (实际拟合用)
    years = np.arange(2004, 2024).reshape(-1, 1)
    gdp = np.array([
        16.18, 18.73, 21.94, 27.01, 31.92, 34.85, 41.21, 48.79, 53.86,
        59.30, 64.36, 68.89, 74.64, 83.20, 91.93, 98.65, 101.36, 114.37,
        121.02, 126.06
    ], dtype=float)

    # 使用年份编码 (从 0 开始，方便数值计算)
    year_code = years - 2004  # 0 ~ 19

    # 二次多项式回归
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(year_code)

    # 正规方程求解
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ gdp
    gdp_pred_train = X_poly @ theta
    r2 = r2_score(gdp, gdp_pred_train)
    rmse = np.sqrt(mean_squared_error(gdp, gdp_pred_train))

    print(f"  多项式系数: a0={theta[0]:.4f}, a1={theta[1]:.4f}, a2={theta[2]:.4f}")
    print(f"  回归方程: GDP = {theta[0]:.4f} + {theta[1]:.4f}×(year-2004) + {theta[2]:.4f}×(year-2004)²")
    print(f"  训练集 R² = {r2:.4f}, RMSE = {rmse:.4f}")

    # 预测 2024 年 GDP
    year_2024_code = np.array([[20]])  # 2024 - 2004 = 20
    X_2024_poly = poly.transform(year_2024_code)
    gdp_2024_pred = (X_2024_poly @ theta)[0]
    print(f"  预测 2024 年 GDP: {gdp_2024_pred:.2f} 万亿元")

    # 绘图
    year_smooth = np.linspace(0, 21, 100).reshape(-1, 1)
    X_smooth_poly = poly.transform(year_smooth)
    gdp_smooth = X_smooth_poly @ theta

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(years, gdp, color="steelblue", edgecolors="k", s=60,
               zorder=5, label="历史 GDP 数据")
    ax.plot(year_smooth.ravel() + 2004, gdp_smooth, "r-", linewidth=2,
            label="二次多项式拟合")
    ax.scatter([2024], [gdp_2024_pred], color="red", edgecolors="k",
               s=100, zorder=6, marker="*", label=f"2024 预测: {gdp_2024_pred:.1f}")
    ax.set_xlabel("年份")
    ax.set_ylabel("GDP (万亿元)")
    ax.set_title("中国 GDP 二次多项式回归预测 (2004-2023)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task2_gdp_prediction.png"), dpi=150)
    plt.show()


# ============================================================
# Task 3: 时间-浓度回归
# ============================================================
def generate_time_conc_file(filepath: str):
    """生成示例 time_conc.txt 文件"""
    np.random.seed(7)
    t = np.arange(0, 21, 1)
    # 浓度递减模型: C(t) = 100 * exp(-0.15*t) + noise
    conc = 100 * np.exp(-0.15 * t) + np.random.normal(0, 2, len(t))
    conc = np.maximum(conc, 0)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# 时间(min)\t浓度(mg/L)\n")
        for ti, ci in zip(t, conc):
            f.write(f"{ti}\t{ci:.2f}\n")
    print(f"[INFO] 已生成时间-浓度文件: {filepath}")
    return t.reshape(-1, 1), conc


def task3_time_concentration():
    print("\n" + "=" * 60)
    print("Task 3: 时间-浓度回归 — 线性 vs 二次多项式")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, "time_conc.txt")

    # 生成或读取数据
    if not os.path.exists(data_file):
        t, conc = generate_time_conc_file(data_file)
    else:
        data = np.loadtxt(data_file, delimiter="\t", skiprows=1)
        t = data[:, 0].reshape(-1, 1)
        conc = data[:, 1]

    X = t.astype(float)
    y = conc.astype(float)

    # --- 线性回归 ---
    lr = PolynomialRegressionLS(degree=1)
    lr.fit(X, y)
    y_pred_linear = lr.predict(X)
    r2_linear = lr.score(X, y)
    rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
    print(f"\n[线性回归]  R² = {r2_linear:.4f}, RMSE = {rmse_linear:.4f}")
    print(f"  方程: conc = {lr.coef_[0]:.4f} + {lr.coef_[1]:.4f} × time")

    # --- 二次多项式回归 ---
    pr = PolynomialRegressionLS(degree=2)
    pr.fit(X, y)
    y_pred_poly = pr.predict(X)
    r2_poly = pr.score(X, y)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
    print(f"\n[二次多项式] R² = {r2_poly:.4f}, RMSE = {rmse_poly:.4f}")
    print(f"  方程: conc = {pr.coef_[0]:.4f} + {pr.coef_[1]:.4f}×t + {pr.coef_[2]:.4f}×t²")

    # 比较
    print(f"\n[比较] 二次多项式 R² 优于线性: {r2_poly > r2_linear} "
          f"(ΔR² = {r2_poly - r2_linear:.4f})")

    # 绘图
    t_smooth = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color="steelblue", edgecolors="k", s=50,
               zorder=5, label="观测数据")
    ax.plot(t_smooth, lr.predict(t_smooth), "g--", linewidth=2,
            label=f"线性拟合 (R²={r2_linear:.3f}, RMSE={rmse_linear:.3f})")
    ax.plot(t_smooth, pr.predict(t_smooth), "r-", linewidth=2.5,
            label=f"二次多项式 (R²={r2_poly:.3f}, RMSE={rmse_poly:.3f})")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("浓度 (mg/L)")
    ax.set_title("时间-浓度: 线性 vs 二次多项式回归")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "task3_time_conc.png"), dpi=150)
    plt.show()


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    task1_simulated_quadratic()
    task2_gdp_prediction()
    task3_time_concentration()
