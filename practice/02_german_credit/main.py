# -*- coding: utf-8 -*-
"""
德国信用风险分析 — 逻辑回归 (German Credit Risk Analysis)
===========================================================================
包含:
  - 数据加载与生成
  - 数据预处理 (编码、标准化、特征工程)
  - **手动实现逻辑回归** (sigmoid + 梯度下降, 不使用 sklearn)
  - sklearn 逻辑回归对比
  - 决策边界可视化
  - 分类报告 (precision / recall / F1)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "german_credit_data.csv")


# ==================== 工具函数 ====================
def _generate_sample_data(n_samples=1000, seed=42):
    """生成示例德国信用数据 (1000 条)。"""
    rng = np.random.RandomState(seed)
    ages = rng.randint(19, 75, n_samples)
    sex = rng.choice(["female", "male"], n_samples, p=[0.4, 0.6])
    job = rng.choice([0, 1, 2, 3], n_samples)          # 0=unskilled, 1=skilled, 2=highly skilled, 3=management
    housing = rng.choice(["own", "rent", "free"], n_samples, p=[0.5, 0.4, 0.1])
    saving = rng.choice(["little", "moderate", "quite rich", "rich", np.nan],
                        n_samples, p=[0.3, 0.3, 0.1, 0.05, 0.25])
    checking = rng.choice(["little", "moderate", "rich", np.nan],
                          n_samples, p=[0.35, 0.35, 0.1, 0.2])
    credit_amount = rng.randint(250, 20000, n_samples).astype(float)
    duration = rng.randint(4, 72, n_samples).astype(float)
    purpose = rng.choice(
        ["car", "furniture/equipment", "radio/TV", "domestic appliances",
         "repairs", "education", "business", "vacation/others"],
        n_samples
    )
    # Risk: 与 credit_amount, duration, saving, checking 相关
    logit = (
        -2.5
        + 0.00015 * credit_amount
        + 0.04 * duration
        - 0.6 * (ages / 50)
        + rng.normal(0, 0.8, n_samples)
    )
    # saving 和 checking 影响
    for i in range(n_samples):
        if isinstance(saving[i], str):
            if saving[i] in ("quite rich", "rich"):
                logit[i] -= 0.8
            elif saving[i] == "moderate":
                logit[i] -= 0.3
        if isinstance(checking[i], str):
            if checking[i] == "rich":
                logit[i] -= 0.5
            elif checking[i] == "moderate":
                logit[i] -= 0.2
    prob = 1 / (1 + np.exp(-logit))
    risk = np.where(prob > 0.5, "bad", "good")

    df = pd.DataFrame({
        "ID": np.arange(1, n_samples + 1),
        "Age": ages,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose,
        "Risk": risk,
    })
    return df


# ====================================================================
# 1. 数据预处理
# ====================================================================
class GermanCreditPreprocessor:
    """德国信用数据加载与预处理。"""

    SAVING_ORDER = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    CHECKING_ORDER = {"none": 0, "little": 1, "moderate": 2, "rich": 3}

    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.df_raw = None
        self.X = None
        self.y = None
        self.feature_names = None                # 所有特征名 (after one-hot)
        self.numerical_cols = ["Age", "Credit amount", "Duration"]
        self.scaler = StandardScaler()
        self.ohe_housing = OneHotEncoder(sparse_output=False, drop="first")
        self.ohe_purpose = OneHotEncoder(sparse_output=False, drop="first")

    def load(self):
        """加载数据。"""
        if os.path.exists(self.csv_path):
            print(f"[INFO] 从 {self.csv_path} 加载数据")
            self.df_raw = pd.read_csv(self.csv_path)
        else:
            print("[INFO] 数据文件不存在, 生成示例数据 (1000 条)")
            self.df_raw = _generate_sample_data(1000)
            self.df_raw.to_csv(self.csv_path, index=False)
            print(f"[INFO] 示例数据已保存至 {self.csv_path}")
        return self.df_raw

    def explore(self):
        """数据概览。"""
        print("\n" + "=" * 60)
        print("数据探索")
        print("=" * 60)
        print(f"形状: {self.df_raw.shape}")
        print(f"缺失值:\n{self.df_raw.isnull().sum()}")
        print(f"Risk 分布:\n{self.df_raw['Risk'].value_counts()}")
        print(f"\n数据类型:\n{self.df_raw.dtypes}")

    def preprocess(self):
        """完整预处理流程。"""
        df = self.df_raw.copy()

        # (1) 丢弃 ID 列
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)

        # (2) Risk 二值编码: good=0, bad=1
        df["Risk"] = df["Risk"].map({"good": 0, "bad": 1}).astype(int)

        # (3) 缺失值处理
        for col in ["Saving accounts", "Checking account"]:
            if col in df.columns:
                df[col] = df[col].fillna("none")

        # (4) Sex 编码: female=0, male=1
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map({"female": 0, "male": 1})

        # (5) 序数编码 Saving / Checking
        if "Saving accounts" in df.columns:
            df["Saving accounts"] = df["Saving accounts"].map(self.SAVING_ORDER)
        if "Checking account" in df.columns:
            df["Checking account"] = df["Checking account"].map(self.CHECKING_ORDER)

        # (6) One-hot 编码 Housing 和 Purpose
        housing_encoded = self.ohe_housing.fit_transform(df[["Housing"]])
        housing_names = [f"Housing_{c}" for c in
                         self.ohe_housing.categories_[0][1:]]  # drop='first'
        purpose_encoded = self.ohe_purpose.fit_transform(df[["Purpose"]])
        purpose_names = [f"Purpose_{c}" for c in
                         self.ohe_purpose.categories_[0][1:]]

        # (7) 特征工程: 组合特征
        df["Credit_Per_Duration"] = df["Credit amount"] / (df["Duration"] + 1e-8)
        df["Age_Per_Job"] = df["Age"] / (df["Job"] + 1)
        df["Credit_Per_Age"] = df["Credit amount"] / (df["Age"] + 1e-8)

        engineered_names = ["Credit_Per_Duration", "Age_Per_Job", "Credit_Per_Age"]

        # (8) 合并所有特征
        df_encoded = pd.DataFrame(housing_encoded, columns=housing_names)
        df_purpose = pd.DataFrame(purpose_encoded, columns=purpose_names)

        base_features = ["Age", "Sex", "Job", "Saving accounts", "Checking account",
                         "Credit amount", "Duration"]
        X_base = df[base_features + engineered_names].values
        X_others = np.hstack([df_encoded.values, df_purpose.values])

        X_combined = np.hstack([X_base, X_others])

        # 需要标准化的列索引 (前 7 个 base 中有 3 个数值列: Age(0), Credit amount(5), Duration(6))
        # base_features = [Age, Sex, Job, Saving accounts, Checking account, Credit amount, Duration]
        # idx of numerical in X_combined: 0(Age), 5(Credit amount), 6(Duration)
        X_combined[:, [0, 5, 6]] = self.scaler.fit_transform(X_combined[:, [0, 5, 6]])

        self.y = df["Risk"].values

        # 构建特征名列表
        self.feature_names = (
            base_features + engineered_names + housing_names + purpose_names
        )
        self.X = X_combined
        print(f"[预处理] 特征维度: {self.X.shape}, 正例 (bad) 比例: {self.y.mean():.3f}")
        return self.X, self.y, self.feature_names


# ====================================================================
# 2. 手动逻辑回归 (不使用 sklearn)
# ====================================================================
class ManualLogisticRegression:
    """手动实现逻辑回归 (sigmoid + 梯度下降)。"""

    def __init__(self, learning_rate=0.1, n_iterations=2000, tol=1e-6, verbose=False):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        """Sigmoid 激活函数: sigma(z) = 1 / (1 + e^(-z))"""
        # 数值稳定版本
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, X, y):
        """二分类交叉熵损失。"""
        m = len(y)
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        # 防止 log(0)
        eps = 1e-15
        loss = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
        return loss

    def fit(self, X, y):
        """梯度下降训练。"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)

            # 梯度计算
            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)

            # 参数更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 记录损失
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if self.verbose and i % 200 == 0:
                print(f"  Iter {i:5d}: loss = {loss:.6f}")

            # 早停
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"  早停于 iter {i}")
                break

        return self

    def predict_proba(self, X):
        """预测正类概率。"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别 (0/1)。"""
        return (self.predict_proba(X) >= threshold).astype(int)


# ====================================================================
# 3. 特征选择与决策边界
# ====================================================================
def select_top2_features(X, y, feature_names):
    """使用互信息 (mutual information) 选出与目标最相关的 2 个特征 (仅从原始数值特征中选，便于可视化)。"""
    # 优先在原始数值特征中选择
    candidates = {}
    for i, name in enumerate(feature_names):
        if np.std(X[:, i]) > 1e-6:
            # 使用简单的 F-score-like 度量: 组间差异
            good_mask = y == 0
            bad_mask = y == 1
            mean_good = np.mean(X[good_mask, i])
            mean_bad = np.mean(X[bad_mask, i])
            std_pooled = np.sqrt(
                (np.var(X[good_mask, i]) + np.var(X[bad_mask, i])) / 2 + 1e-8
            )
            score = abs(mean_good - mean_bad) / std_pooled
            candidates[name] = (i, score)

    # 按得分降序取前 2
    sorted_feats = sorted(candidates.items(), key=lambda kv: kv[1][1], reverse=True)
    top2 = sorted_feats[:2]
    indices = [v[0] for _, v in top2]
    names = [k for k, _ in top2]
    print(f"\n[特征选择] Top-2 最具区分力的特征: {names}")
    return indices, names


def plot_decision_boundary(X_2d, y, model, feat_names, tag="manual"):
    """为 2 特征数据绘制决策边界。"""
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid)[:, 1]
    else:
        Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, cmap="RdYlBu", alpha=0.6)
    plt.colorbar(label="P(bad)")

    # 散点
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="RdYlBu",
                          edgecolors="k", linewidths=0.5, s=30)
    plt.contour(xx, yy, Z, levels=[0.5], colors="green", linewidths=2,
                linestyles="--")

    plt.xlabel(feat_names[0])
    plt.ylabel(feat_names[1])
    plt.title(f"逻辑回归决策边界 ({tag})")
    plt.legend(handles=scatter.legend_elements()[0],
               labels=["good", "bad"], title="Risk")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"10_decision_boundary_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {path}")


# ====================================================================
# main
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("德国信用风险分析 — 逻辑回归")
    print("=" * 60)

    # ---- 数据加载与预处理 ----
    preprocessor = GermanCreditPreprocessor()
    preprocessor.load()
    preprocessor.explore()
    X, y, feature_names = preprocessor.preprocess()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    # ---- 手动逻辑回归 ----
    print("\n" + "=" * 60)
    print("手动逻辑回归 (梯度下降)")
    print("=" * 60)
    manual_lr = ManualLogisticRegression(
        learning_rate=0.5, n_iterations=5000, verbose=True
    )
    manual_lr.fit(X_train, y_train)

    # 损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(manual_lr.loss_history, linewidth=1.5)
    plt.xlabel("迭代次数")
    plt.ylabel("交叉熵损失")
    plt.title("手动逻辑回归 — 损失函数收敛曲线")
    plt.tight_layout()
    loss_path = os.path.join(FIG_DIR, "07_loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {loss_path}")

    # 评估手动模型
    y_pred_manual = manual_lr.predict(X_test)
    acc_manual = accuracy_score(y_test, y_pred_manual)
    err_manual = 1 - acc_manual
    print(f"\n手动 Logistic Regression:")
    print(f"  准确率 (Accuracy) : {acc_manual:.4f}")
    print(f"  错误率 (Error)    : {err_manual:.4f}")
    print(f"  Precision (bad=1): {precision_score(y_test, y_pred_manual):.4f}")
    print(f"  Recall    (bad=1): {recall_score(y_test, y_pred_manual):.4f}")
    print(f"  F1-score  (bad=1): {f1_score(y_test, y_pred_manual):.4f}")

    # ---- sklearn 逻辑回归对比 ----
    print("\n" + "=" * 60)
    print("sklearn 逻辑回归 (对比)")
    print("=" * 60)
    sk_lr = SklearnLogReg(max_iter=5000, random_state=42)
    sk_lr.fit(X_train, y_train)
    y_pred_sk = sk_lr.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"sklearn LogisticRegression:")
    print(f"  准确率 (Accuracy) : {acc_sk:.4f}")
    print(f"  错误率 (Error)    : {1 - acc_sk:.4f}")
    print(f"  Precision (bad=1): {precision_score(y_test, y_pred_sk):.4f}")
    print(f"  Recall    (bad=1): {recall_score(y_test, y_pred_sk):.4f}")
    print(f"  F1-score  (bad=1): {f1_score(y_test, y_pred_sk):.4f}")

    # ---- 分类报告 ----
    print("\n" + "=" * 60)
    print("分类报告 (sklearn)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_sk,
                                target_names=["good", "bad"]))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_sk)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=14, fontweight="bold")
    plt.xticks([0, 1], ["good", "bad"])
    plt.yticks([0, 1], ["good", "bad"])
    plt.xlabel("预测")
    plt.ylabel("实际")
    plt.title("混淆矩阵 (sklearn)")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "08_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {cm_path}")

    # ---- 决策边界 (Top-2 特征) ----
    top2_idx, top2_names = select_top2_features(X_train, y_train, feature_names)
    X_train_2d = X_train[:, top2_idx]
    X_test_2d = X_test[:, top2_idx]

    # 手动模型 — 重新在 2D 数据上训练用于绘图
    print("\n[决策边界] 训练手动逻辑回归 (2 特征)...")
    manual_2d = ManualLogisticRegression(learning_rate=0.5, n_iterations=3000)
    manual_2d.fit(X_train_2d, y_train)
    plot_decision_boundary(X_train_2d, y_train, manual_2d, top2_names, tag="manual")

    # sklearn 模型 — 也重新在 2D 数据上训练
    sk_2d = SklearnLogReg(max_iter=5000, random_state=42)
    sk_2d.fit(X_train_2d, y_train)
    plot_decision_boundary(X_train_2d, y_train, sk_2d, top2_names, tag="sklearn")

    # ---- 特征权重可视化 ----
    plt.figure(figsize=(10, 5))
    # 仅显示可解释的特征 (非 one-hot)
    n_display = min(20, len(feature_names))
    idx = np.argsort(np.abs(manual_lr.weights))[::-1][:n_display]
    names_display = [feature_names[i] for i in idx]
    weights_display = manual_lr.weights[idx]
    colors = ["#d62728" if w < 0 else "#2ca02c" for w in weights_display]
    plt.barh(range(len(names_display)), weights_display, color=colors)
    plt.yticks(range(len(names_display)), names_display)
    plt.xlabel("权重值")
    plt.title("手动逻辑回归 — 特征权重 (Top-20)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    weight_path = os.path.join(FIG_DIR, "09_feature_weights.png")
    plt.savefig(weight_path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {weight_path}")

    print("\n" + "=" * 60)
    print("所有任务完成! PNG 图表保存于:", FIG_DIR)
    print("=" * 60)
