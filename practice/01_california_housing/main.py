# -*- coding: utf-8 -*-
"""
加州房价预测 — 回归分析及应用 (California Housing Price Prediction)
===========================================================================
包含:
  - HousingDataAnalyzer:  数据加载、探索、预处理、相关性分析
  - UnivariateLinearRegression:  一元线性回归
  - MultivariateLinearRegression: 多元线性回归
  - PolynomialRegression:  多项式回归
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # 非交互式后端, 直接保存 PNG
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 中文字体 ----------
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "california_housing.csv")


# ==================== 工具函数 ====================
def _generate_sample_data(n_samples=200, seed=42):
    """生成示例加州房价数据。"""
    rng = np.random.RandomState(seed)
    longitude = rng.uniform(-124.35, -114.31, n_samples)
    latitude = rng.uniform(32.54, 41.95, n_samples)
    housing_median_age = rng.randint(1, 53, n_samples)
    total_rooms = rng.randint(2, 10000, n_samples)
    total_bedrooms = (total_rooms * rng.uniform(0.25, 0.75, n_samples)).astype(float)
    # 人为制造缺失值
    nan_idx = rng.choice(n_samples, size=15, replace=False)
    total_bedrooms[nan_idx] = np.nan
    population = rng.randint(3, 6000, n_samples)
    households = rng.randint(1, 1800, n_samples)
    median_income = np.round(rng.uniform(0.5, 15.0, n_samples), 4)
    # median_house_value 与 median_income 呈正相关, 并加入噪声
    median_house_value = (
        50000 + 35000 * median_income + rng.normal(0, 25000, n_samples)
    )
    median_house_value = np.clip(median_house_value, 15000, 500001).astype(int)
    ocean_proximity = rng.choice(
        ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
        size=n_samples,
        p=[0.1, 0.35, 0.30, 0.20, 0.05],
    )

    df = pd.DataFrame({
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "median_house_value": median_house_value,
        "ocean_proximity": ocean_proximity,
    })
    return df


# ====================================================================
# 1. 数据加载与分析
# ====================================================================
class HousingDataAnalyzer:
    """加州房价数据加载、探索与预处理。"""

    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.corr_matrix = None

    def load_data(self):
        """加载 CSV 或生成示例数据。"""
        if os.path.exists(self.csv_path):
            print(f"[INFO] 从 {self.csv_path} 加载数据")
            self.df = pd.read_csv(self.csv_path)
        else:
            print("[INFO] 数据文件不存在, 生成示例数据 (200 条)")
            self.df = _generate_sample_data(200)
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.df.to_csv(self.csv_path, index=False)
            print(f"[INFO] 示例数据已保存至 {self.csv_path}")
        return self.df

    def explore(self):
        """数据探索: 基本信息、缺失值、描述性统计。"""
        print("\n" + "=" * 60)
        print("数据探索")
        print("=" * 60)
        print(f"数据形状: {self.df.shape}")
        print(f"\n前 5 行:\n{self.df.head()}")
        print(f"\n数据类型:\n{self.df.dtypes}")
        print(f"\n缺失值统计:\n{self.df.isnull().sum()}")
        print(f"\n描述性统计:\n{self.df.describe()}")

    def preprocess(self):
        """预处理: LabelEncode ocean_proximity, 均值填充 total_bedrooms。"""
        df = self.df.copy()

        # 缺失值 —— 均值填充
        if df["total_bedrooms"].isnull().sum() > 0:
            fill_val = df["total_bedrooms"].mean()
            df["total_bedrooms"].fillna(fill_val, inplace=True)
            print(f"[预处理] total_bedrooms 缺失值已用均值 ({fill_val:.2f}) 填充")

        # LabelEncode 分类特征
        df["ocean_proximity"] = self.label_encoder.fit_transform(
            df["ocean_proximity"].astype(str)
        )
        print(f"[预处理] ocean_proximity 编码映射: "
              f"{dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 分离特征与目标
        self.feature_names = [c for c in df.columns if c != "median_house_value"]
        self.X = df[self.feature_names]
        self.y = df["median_house_value"]
        print(f"[预处理] 特征数: {len(self.feature_names)}, 样本数: {len(df)}")
        return self.X, self.y

    def correlation_analysis(self):
        """相关性分析与特征重要性排序。"""
        if self.X is None:
            self.preprocess()
        corr = self.X.copy()
        corr["median_house_value"] = self.y
        self.corr_matrix = corr.corr()

        # 与目标变量的相关性 (绝对值排序)
        target_corr = (
            self.corr_matrix["median_house_value"]
            .drop("median_house_value")
            .abs()
            .sort_values(ascending=False)
        )
        print("\n" + "=" * 60)
        print("特征重要性 (按 |r| 排序)")
        print("=" * 60)
        for name, val in target_corr.items():
            bar = "#" * int(val * 40)
            print(f"  {name:25s}  |r| = {val:.4f}  {bar}")
        return target_corr

    def plot_correlation_heatmap(self):
        """保存相关性热力图。"""
        if self.corr_matrix is None:
            self.correlation_analysis()
        plt.figure(figsize=(10, 8))
        im = plt.imshow(self.corr_matrix, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, shrink=0.8)
        plt.xticks(range(len(self.corr_matrix.columns)),
                   self.corr_matrix.columns, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(len(self.corr_matrix.columns)),
                   self.corr_matrix.columns, fontsize=8)
        plt.title("特征相关性热力图", fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "01_correlation_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")


# ====================================================================
# 2. 一元线性回归
# ====================================================================
class UnivariateLinearRegression:
    """一元线性回归 —— 使用与目标相关性最高的特征。"""

    def __init__(self, analyzer: HousingDataAnalyzer):
        self.analyzer = analyzer
        self.best_feature = None
        self.corr_value = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.model = None
        self.y_train_pred = self.y_test_pred = None

    def prepare(self):
        """选择最佳特征并划分数据集。"""
        target_corr = self.analyzer.correlation_analysis()
        # 取绝对值最大的特征
        self.best_feature = target_corr.index[0]
        self.corr_value = target_corr.iloc[0]
        print(f"\n[一元回归] 最佳特征: '{self.best_feature}', "
              f"|r| = {self.corr_value:.4f}")

        X = self.analyzer.X[[self.best_feature]].values
        y = self.analyzer.y.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self

    def train(self):
        """训练一元线性回归。"""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)

        coef, intercept = self.model.coef_[0], self.model.intercept_
        print(f"\n  拟合结果: y = {intercept:.2f} + {coef:.2f} * {self.best_feature}")
        self._report_metrics()
        return self

    def _report_metrics(self):
        """输出训练/测试指标。"""
        for tag, y_true, y_pred in [
            ("训练集", self.y_train, self.y_train_pred),
            ("测试集", self.y_test, self.y_test_pred),
        ]:
            print(f"\n  --- {tag} ---")
            print(f"    R²   : {r2_score(y_true, y_pred):.4f}")
            print(f"    MSE  : {mean_squared_error(y_true, y_pred):.2f}")
            print(f"    RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
            print(f"    MAE  : {mean_absolute_error(y_true, y_pred):.2f}")

    def plot(self):
        """散点图 + 回归线。"""
        plt.figure(figsize=(8, 5))
        plt.scatter(self.X_train, self.y_train, alpha=0.5, s=20, label="训练集")
        plt.scatter(self.X_test, self.y_test, alpha=0.5, s=20, label="测试集")

        # 回归线 (在 X 的全部范围内)
        x_line = np.linspace(self.X_train.min(), self.X_train.max(), 200).reshape(-1, 1)
        y_line = self.model.predict(x_line)
        plt.plot(x_line, y_line, "r-", linewidth=2, label="回归线")

        plt.xlabel(self.best_feature.replace("_", " ").title())
        plt.ylabel("Median House Value")
        plt.title(f"一元线性回归: {self.best_feature} → 房价")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "02_univariate_regression.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")


# ====================================================================
# 3. 多元线性回归
# ====================================================================
class MultivariateLinearRegression:
    """多元线性回归 —— 选取 top-5 特征, 标准化后训练。"""

    def __init__(self, analyzer: HousingDataAnalyzer, top_k=5):
        self.analyzer = analyzer
        self.top_k = top_k
        self.selected_features = None
        self.scaler = StandardScaler()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.model = None
        self.y_train_pred = self.y_test_pred = None

    def prepare(self):
        """选取 top-k 特征并标准化。"""
        target_corr = self.analyzer.correlation_analysis()
        self.selected_features = list(target_corr.index[:self.top_k])
        print(f"\n[多元回归] 选取的 {self.top_k} 个特征: {self.selected_features}")

        X = self.analyzer.X[self.selected_features].values
        y = self.analyzer.y.values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self

    def train(self):
        """训练多元线性回归。"""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)

        print(f"\n  截距: {self.model.intercept_:.2f}")
        for name, coef in zip(self.selected_features, self.model.coef_):
            print(f"    {name:25s}  系数 = {coef:10.2f}")

        print("\n  --- 性能对比 ---")
        for tag, y_true, y_pred in [
            ("训练集", self.y_train, self.y_train_pred),
            ("测试集", self.y_test, self.y_test_pred),
        ]:
            print(f"  {tag}: R²={r2_score(y_true, y_pred):.4f}, "
                  f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        return self

    def plot_coefficients(self):
        """特征系数柱状图。"""
        coefs = self.model.coef_
        colors = ["#2ca02c" if c >= 0 else "#d62728" for c in coefs]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(range(len(coefs)), coefs, color=colors)
        plt.axhline(y=0, color="black", linewidth=0.8)
        plt.xticks(range(len(coefs)), self.selected_features, rotation=30, ha="right")
        plt.ylabel("标准化系数")
        plt.title("多元线性回归 — 特征系数")
        # 标注数值
        for bar, val in zip(bars, coefs):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (500 if val >= 0 else -2000),
                     f"{val:.0f}", ha="center", fontsize=8)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "03_multivariate_coefficients.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")

    def plot_residuals(self):
        """残差图。"""
        residuals = self.y_test - self.y_test_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 残差 vs 预测值
        axes[0].scatter(self.y_test_pred, residuals, alpha=0.5, s=15)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("预测值")
        axes[0].set_ylabel("残差")
        axes[0].set_title("残差 vs 预测值")

        # 残差直方图
        axes[1].hist(residuals, bins=25, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--")
        axes[1].set_xlabel("残差")
        axes[1].set_ylabel("频数")
        axes[1].set_title("残差分布")

        plt.tight_layout()
        path = os.path.join(FIG_DIR, "04_residual_plots.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")


# ====================================================================
# 4. 多项式回归
# ====================================================================
class PolynomialRegression:
    """多项式回归 —— 基于最佳特征, 搜索最优 degree (1-4)。"""

    def __init__(self, analyzer: HousingDataAnalyzer):
        self.analyzer = analyzer
        self.best_feature = None
        self.X_train_raw = self.X_test_raw = None
        self.y_train = self.y_test = None
        self.results = {}          # degree -> dict(metrics)
        self.best_degree = None
        self.best_model = None
        self.best_poly = None
        self.best_scaler = StandardScaler()

    def prepare(self):
        """使用最佳特征准备数据。"""
        target_corr = self.analyzer.correlation_analysis()
        self.best_feature = target_corr.index[0]
        X = self.analyzer.X[[self.best_feature]].values
        y = self.analyzer.y.values
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self

    def search_optimal_degree(self, max_degree=4):
        """搜索 degree=1..max_degree 中测试集 R² 最高的。"""
        print(f"\n[多项式回归] 在度 1-{max_degree} 中搜索最优...")
        for d in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=d, include_bias=False)
            X_train_poly = poly.fit_transform(self.X_train_raw)
            X_test_poly = poly.transform(self.X_test_raw)

            # 标准化 (多项式特征尺度差异大)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_poly)
            X_test_scaled = scaler.transform(X_test_poly)

            model = LinearRegression()
            model.fit(X_train_scaled, self.y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            self.results[d] = {"r2": r2, "rmse": rmse, "model": model,
                               "poly": poly, "scaler": scaler}
            print(f"  degree={d}:  R²(test)={r2:.4f},  RMSE={rmse:.2f}")

        self.best_degree = max(self.results, key=lambda k: self.results[k]["r2"])
        best = self.results[self.best_degree]
        self.best_model = best["model"]
        self.best_poly = best["poly"]
        self.best_scaler = best["scaler"]
        print(f"\n  最优度: {self.best_degree} (R²={best['r2']:.4f})")
        return self

    def compare_with_linear(self, univariate_r2_test):
        """与一元/多元线性回归对比。"""
        best = self.results[self.best_degree]
        print("\n" + "=" * 60)
        print("模型对比 (测试集)")
        print("=" * 60)
        print(f"  一元线性回归       R² = {univariate_r2_test:.4f}")
        print(f"  多项式回归 (d={self.best_degree})  R² = {best['r2']:.4f}")

    def plot_degree_curve(self):
        """绘制 degree vs R² 曲线。"""
        degrees = sorted(self.results.keys())
        r2_vals = [self.results[d]["r2"] for d in degrees]
        rmse_vals = [self.results[d]["rmse"] for d in degrees]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#1f77b4"
        ax1.plot(degrees, r2_vals, "o-", color=color1, linewidth=2, markersize=8)
        ax1.set_xlabel("多项式度 (Degree)")
        ax1.set_ylabel("R²", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        for d, r2 in zip(degrees, r2_vals):
            ax1.annotate(f"{r2:.4f}", (d, r2), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

        ax2 = ax1.twinx()
        color2 = "#d62728"
        ax2.plot(degrees, rmse_vals, "s--", color=color2, linewidth=2, markersize=8)
        ax2.set_ylabel("RMSE", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_xticks(degrees)
        plt.title(f"多项式度对模型性能的影响 (特征: {self.best_feature})")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "05_polynomial_degree_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")

    def plot_polynomial_fit(self):
        """绘制多项式拟合曲线 vs 数据散点。"""
        X_sorted_idx = np.argsort(self.X_train_raw.ravel())
        X_sorted = self.X_train_raw[X_sorted_idx]
        X_test_poly = self.best_poly.transform(X_sorted)
        X_test_scaled = self.best_scaler.transform(X_test_poly)
        y_curve = self.best_model.predict(X_test_scaled)

        plt.figure(figsize=(8, 5))
        plt.scatter(self.X_train_raw, self.y_train, alpha=0.4, s=15, label="训练集")
        plt.scatter(self.X_test_raw, self.y_test, alpha=0.4, s=15, label="测试集")
        plt.plot(X_sorted, y_curve, "r-", linewidth=2,
                 label=f"多项式拟合 (d={self.best_degree})")
        plt.xlabel(self.best_feature.replace("_", " ").title())
        plt.ylabel("Median House Value")
        plt.title(f"多项式回归 (degree={self.best_degree})")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "06_polynomial_fit.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[图表] 已保存: {path}")


# ====================================================================
# main
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("加州房价预测 — 回归分析及应用")
    print("=" * 60)

    # ---- 数据分析 ----
    analyzer = HousingDataAnalyzer()
    analyzer.load_data()
    analyzer.explore()
    analyzer.preprocess()
    analyzer.plot_correlation_heatmap()

    # ---- 一元线性回归 ----
    uni = UnivariateLinearRegression(analyzer)
    uni.prepare().train()
    uni.plot()
    uni_r2_test = r2_score(uni.y_test, uni.y_test_pred)

    # ---- 多元线性回归 ----
    multi = MultivariateLinearRegression(analyzer, top_k=5)
    multi.prepare().train()
    multi.plot_coefficients()
    multi.plot_residuals()

    # ---- 多项式回归 ----
    poly_reg = PolynomialRegression(analyzer)
    poly_reg.prepare().search_optimal_degree(max_degree=4)
    poly_reg.compare_with_linear(uni_r2_test)
    poly_reg.plot_degree_curve()
    poly_reg.plot_polynomial_fit()

    print("\n" + "=" * 60)
    print("所有任务完成! PNG 图表保存于:", FIG_DIR)
    print("=" * 60)
