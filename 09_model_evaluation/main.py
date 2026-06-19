"""
09_model_evaluation — 模型性能评估
============================================
Task 1: Breast cancer binary classification — confusion matrix & metrics
Task 2: Pima Diabetes — LogisticRegression vs DecisionTree 5-fold stratified CV
Task 3: Class imbalance analysis on Pima Diabetes
Task 4: PR curve & ROC curve comparison (GaussianNB vs SVM/RBF)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PIMA_CSV = os.path.join(DATA_DIR, "pima-diabetes.csv")


# ===================================================================
# Helper: generate Pima Diabetes CSV if missing
# ===================================================================
def generate_pima_csv():
    """Generate a realistic Pima Indians Diabetes dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)

    np.random.seed(42)
    n = 768
    n_diabetic = 268
    n_healthy = 500

    # Healthy population parameters
    preg_healthy = np.random.poisson(lam=2.5, size=n_healthy)
    glucose_healthy = np.random.normal(loc=105, scale=15, size=n_healthy)
    bp_healthy = np.random.normal(loc=66, scale=10, size=n_healthy)
    skin_healthy = np.random.normal(loc=27, scale=8, size=n_healthy)
    insulin_healthy = np.random.normal(loc=110, scale=45, size=n_healthy)
    bmi_healthy = np.random.normal(loc=30, scale=6, size=n_healthy)
    dpf_healthy = np.random.normal(loc=0.45, scale=0.25, size=n_healthy)
    age_healthy = np.random.normal(loc=29, scale=8, size=n_healthy)

    # Diabetic population parameters
    preg_diab = np.random.poisson(lam=4.5, size=n_diabetic)
    glucose_diab = np.random.normal(loc=145, scale=30, size=n_diabetic)
    bp_diab = np.random.normal(loc=74, scale=14, size=n_diabetic)
    skin_diab = np.random.normal(loc=32, scale=10, size=n_diabetic)
    insulin_diab = np.random.normal(loc=170, scale=70, size=n_diabetic)
    bmi_diab = np.random.normal(loc=35, scale=7, size=n_diabetic)
    dpf_diab = np.random.normal(loc=0.58, scale=0.35, size=n_diabetic)
    age_diab = np.random.normal(loc=37, scale=10, size=n_diabetic)

    # Concatenate
    preg = np.concatenate([preg_healthy, preg_diab])
    glucose = np.concatenate([glucose_healthy, glucose_diab])
    bp = np.concatenate([bp_healthy, bp_diab])
    skin = np.concatenate([skin_healthy, skin_diab])
    insulin = np.concatenate([insulin_healthy, insulin_diab])
    bmi = np.concatenate([bmi_healthy, bmi_diab])
    dpf = np.concatenate([dpf_healthy, dpf_diab])
    age = np.concatenate([age_healthy, age_diab])
    outcome = np.concatenate([np.zeros(n_healthy), np.ones(n_diabetic)])

    # Clip to plausible ranges
    preg = np.clip(np.round(preg), 0, 17).astype(int)
    glucose = np.clip(glucose, 0, 200)
    bp = np.clip(bp, 0, 122)
    skin = np.clip(skin, 0, 99)
    insulin = np.clip(insulin, 0, 846)
    bmi = np.clip(bmi, 0, 67)
    dpf = np.clip(dpf, 0.08, 2.42)
    age = np.clip(np.round(age), 21, 81).astype(int)

    df = pd.DataFrame({
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Outcome": outcome.astype(int),
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(PIMA_CSV, index=False)
    print(f"[生成] Pima Diabetes CSV 已保存至 {PIMA_CSV}")
    return df


def load_pima():
    """Load Pima data; generate if missing."""
    if not os.path.exists(PIMA_CSV):
        return generate_pima_csv()
    return pd.read_csv(PIMA_CSV)


# ===================================================================
# Task 1: Breast cancer binary classification evaluation
# ===================================================================
def task1_breast_cancer():
    print("\n" + "=" * 70)
    print("Task 1: 乳腺癌二分类 — 逻辑回归性能评估")
    print("=" * 70)

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n混淆矩阵:\n{cm}")
    print(f"准确率 (Accuracy):    {acc:.4f}")
    print(f"精确率 (Precision):   {prec:.4f}")
    print(f"召回率 (Recall):      {rec:.4f}")
    print(f"F1-score:              {f1:.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

    # -------- Plot 1: Confusion matrix heatmap --------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=data.target_names, yticklabels=data.target_names,
                ax=axes[0])
    axes[0].set_title("混淆矩阵 (Confusion Matrix)")
    axes[0].set_xlabel("预测标签")
    axes[0].set_ylabel("真实标签")

    # -------- Plot 2: Metrics bar chart --------
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [acc, prec, rec, f1]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = axes[1].bar(metrics_names, metrics_values, color=colors, edgecolor="white")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("模型评估指标 (Model Evaluation Metrics)")
    axes[1].set_ylabel("数值")
    for bar, val in zip(bars, metrics_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "09_model_evaluation", "task1_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[Task 1 完成] 图已保存。")


# ===================================================================
# Task 2: Pima Diabetes — 5-fold stratified CV comparison
# ===================================================================
def task2_pima_cv():
    print("\n" + "=" * 70)
    print("Task 2: Pima Diabetes — 逻辑回归 vs 决策树 (5折分层交叉验证)")
    print("=" * 70)

    df = load_pima()
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": []}
               for name in models}

    for name, model in models.items():
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results[name]["accuracy"].append(accuracy_score(y_te, y_pred))
            results[name]["precision"].append(precision_score(y_te, y_pred))
            results[name]["recall"].append(recall_score(y_te, y_pred))
            results[name]["f1"].append(f1_score(y_te, y_pred))

    # --- Comparison table ---
    print(f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 62)
    for name in models:
        print(f"{name:<22} {np.mean(results[name]['accuracy']):>10.4f} "
              f"{np.mean(results[name]['precision']):>10.4f} "
              f"{np.mean(results[name]['recall']):>10.4f} "
              f"{np.mean(results[name]['f1']):>10.4f}")

    # --- Box plot of CV F1 scores ---
    fig, ax = plt.subplots(figsize=(8, 5))
    f1_data = [results[name]["f1"] for name in models]
    bp = ax.boxplot(f1_data, labels=list(models.keys()), patch_artist=True,
                    widths=0.4)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("5折交叉验证 F1-score 箱线图")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "09_model_evaluation", "task2_f1_boxplot.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- Conclusion ---
    lr_f1 = np.mean(results["LogisticRegression"]["f1"])
    dt_f1 = np.mean(results["DecisionTree"]["f1"])
    winner = "LogisticRegression" if lr_f1 > dt_f1 else "DecisionTree"
    print(f"\n[结论] {winner} 在 5 折交叉验证下 F1 表现更优 "
          f"(LR: {lr_f1:.4f}, DT: {dt_f1:.4f})")
    print("[Task 2 完成]")


# ===================================================================
# Task 3: Class imbalance analysis
# ===================================================================
def task3_imbalance():
    print("\n" + "=" * 70)
    print("Task 3: Pima Diabetes — 类别不平衡分析")
    print("=" * 70)

    df = load_pima()
    counts = df["Outcome"].value_counts().sort_index()
    n_neg, n_pos = counts[0], counts[1]

    print(f"\n非糖尿病 (0): {n_neg}  ({n_neg / len(df) * 100:.1f}%)")
    print(f"糖尿病   (1): {n_pos}  ({n_pos / len(df) * 100:.1f}%)")
    print(f"不平衡比例: {n_neg / n_pos:.2f} : 1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    labels = ["非糖尿病 (0)", "糖尿病 (1)"]
    axes[0].pie([n_neg, n_pos], labels=labels, autopct="%1.1f%%",
                colors=["#55A868", "#C44E52"], startangle=90, explode=(0, 0.05))
    axes[0].set_title("类别分布 — 饼图")

    # Bar chart
    bars = axes[1].bar(labels, [n_neg, n_pos], color=["#55A868", "#C44E52"],
                       edgecolor="white")
    axes[1].set_title("类别分布 — 柱状图")
    axes[1].set_ylabel("样本数量")
    for bar, val in zip(bars, [n_neg, n_pos]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(val), ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "09_model_evaluation", "task3_imbalance.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # Discussion
    print("""
[讨论] 为什么准确率不足以评估不平衡数据？

1. 准确率陷阱: 若模型将所有样本预测为多数类(非糖尿病)，准确率可达 65.1%
   (500/768)，看似不错，但实际对少数类(糖尿病)的识别率为 0%，毫无临床价值。

2. 精确率 (Precision): 衡量预测为糖尿病的样本中真正患病的比例。若假阳性过多，
   会导致不必要的进一步检查，增加医疗成本。

3. 召回率 (Recall): 衡量真实糖尿病患者被正确识别的比例。在医疗场景中，漏诊
   (假阴性) 的代价极高，因此高召回率至关重要。

4. F1-score: Precision 和 Recall 的调和平均，在两者之间取得平衡。

5. AUC-ROC: 反映模型在不同阈值下区分正负类的能力，不受类别分布影响，
   是评估不平衡分类问题的首选指标之一。

结论: 对于不平衡数据集，应综合使用 Precision、Recall、F1、AUC-ROC 等指标，
     而非仅依赖 Accuracy。
""")
    print("[Task 3 完成]")


# ===================================================================
# Task 4: PR curve & ROC curve comparison
# ===================================================================
def task4_pr_roc():
    print("\n" + "=" * 70)
    print("Task 4: PR 曲线 与 ROC 曲线对比 (GaussianNB vs SVM-RBF)")
    print("=" * 70)

    df = load_pima()
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "GaussianNB": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline for PR curve
    baseline_pr = y_test.sum() / len(y_test)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --- PR curve ---
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[0].plot(rec_curve, prec_curve, lw=2, label=f"{name} (AP={ap:.3f})")

        # --- ROC curve ---
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    # PR chart annotations
    axes[0].axhline(y=baseline_pr, color="gray", linestyle="--",
                    label=f"Baseline (y={baseline_pr:.3f})")
    axes[0].set_xlabel("召回率 Recall")
    axes[0].set_ylabel("精确率 Precision")
    axes[0].set_title("PR 曲线 (Precision-Recall Curve)")
    axes[0].legend(loc="lower left")
    axes[0].grid(alpha=0.3)

    # ROC chart annotations
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    axes[1].set_xlabel("假阳性率 FPR")
    axes[1].set_ylabel("真阳性率 TPR (Recall)")
    axes[1].set_title("ROC 曲线 (Receiver Operating Characteristic)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "09_model_evaluation", "task4_pr_roc.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    print("""
[讨论] PR 曲线 vs ROC 曲线在不平衡分类中的差异:

1. ROC 曲线的局限性:
   - ROC 曲线以 FPR(假阳性率)为横轴，FPR = FP / (FP + TN)，受真阴性(TN)主导。
   - 在不平衡数据中，TN 数量巨大，即使 FP 增加，FPR 变化也不明显，
     使得 ROC 曲线倾向于"过于乐观"。

2. PR 曲线的优势:
   - PR 曲线关注 Precision = TP / (TP + FP)，不受 TN 影响。
   - 在不平衡数据中，PR 曲线更能暴露模型在识别少数类方面的实际困难。
   - Baseline 线为 正类比例(而非 0.5)，使得评估更有意义。

3. 选择建议:
   - 当类别不平衡严重时，优先参考 PR 曲线和 Average Precision (AP)。
   - ROC-AUC 适合类别相对平衡的场景，或需要与随机分类器比较时。
   - 两者互补使用可全面评估模型性能。
""")
    print("[Task 4 完成]")


# ===================================================================
# main
# ===================================================================
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    task1_breast_cancer()
    task2_pima_cv()
    task3_imbalance()
    task4_pr_roc()

    print("\n" + "=" * 70)
    print("09_model_evaluation 全部任务完成!")
    print("=" * 70)
