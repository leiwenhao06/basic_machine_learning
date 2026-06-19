"""
10_ensemble_learning — 集成学习
=================================
Task 1: Breast cancer — Bagging / RF / AdaBoost / GBDT 对比 (10-fold CV)
Task 2: Bodyfat regression — 多种集成方法对比 + 特征重要性
Task 3: Boston housing — 10 种方法大全对比
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    KFold,
    ShuffleSplit,
)
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===================================================================
# Utility
# ===================================================================
def print_table(header, rows, col_widths=None):
    """Pretty-print a table."""
    if col_widths is None:
        col_widths = [max(len(str(c)) for c in col) + 2 for col in zip(header, *rows)]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt.format(*[f"{x:.4f}" if isinstance(x, float) else str(x) for x in row]))


# ===================================================================
# Task 1: Breast cancer — ensemble classifiers
# ===================================================================
def task1_breast_cancer_ensemble():
    print("\n" + "=" * 70)
    print("Task 1: 乳腺癌集成分类器对比")
    print("=" * 70)

    data = load_breast_cancer()
    X, y = data.data, data.target
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "Bagging (n=50, md=5)": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50, random_state=42),
        "RandomForest (n=100)": RandomForestClassifier(
            n_estimators=100, random_state=42),
        "AdaBoost (n=100, lr=0.8)": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.8, random_state=42),
        "GBDT (n=100, lr=0.1)": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42),
    }

    scorers = ["accuracy", "precision", "recall", "f1"]
    results = {}
    for name, model in models.items():
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        results[name] = {s: np.mean(cv_results[f"test_{s}"]) for s in scorers}
        # Also store raw folds for box-plot later if needed
        results[name]["_raw_f1"] = cv_results["test_f1"]

    # --- Comprehensive table ---
    header = ["Model", "Accuracy", "Precision", "Recall", "F1-score"]
    rows = [[name] + [results[name][s] for s in scorers] for name in models]
    print_table(header, rows)

    # --- Effect of max_depth on RF ---
    print("\n--- RandomForest max_depth 影响 ---")
    depths = [2, 5, 10, 15, 20, None]
    print(f"{'max_depth':<12} {'Accuracy':<10} {'F1':<10}")
    print("-" * 32)
    for d in depths:
        rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
        acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        print(f"{str(d):<12} {acc:<10.4f} {f1:<10.4f}")

    # --- Effect of n_estimators on AdaBoost ---
    print("\n--- AdaBoost n_estimators 影响 ---")
    n_est_list = [10, 50, 100, 200, 500]
    print(f"{'n_estimators':<14} {'Accuracy':<10} {'F1':<10}")
    print("-" * 34)
    for n in n_est_list:
        ada = AdaBoostClassifier(n_estimators=n, learning_rate=0.8, random_state=42)
        acc = cross_val_score(ada, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        f1 = cross_val_score(ada, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        print(f"{n:<14} {acc:<10.4f} {f1:<10.4f}")

    print("\n[Task 1 完成]")


# ===================================================================
# Task 2: Bodyfat regression
# ===================================================================
def generate_bodyfat(n_samples=252):
    """Generate synthetic bodyfat dataset (similar to real bodyfat data)."""
    np.random.seed(42)
    n = n_samples

    age = np.random.uniform(22, 81, n)
    weight = np.random.normal(80, 15, n)
    height = np.random.normal(175, 10, n)
    bmi = weight / ((height / 100) ** 2)
    neck = np.random.normal(38, 3, n)
    chest = np.random.normal(100, 10, n)
    abdomen = np.random.normal(90, 12, n)
    hip = np.random.normal(100, 10, n)
    thigh = np.random.normal(59, 5, n)
    knee = np.random.normal(38, 3, n)
    ankle = np.random.normal(23, 2, n)
    biceps = np.random.normal(32, 3, n)
    forearm = np.random.normal(28, 2, n)
    wrist = np.random.normal(18, 1, n)

    # bodyfat = f(abdomen, weight, wrist, age, ...) + noise
    bodyfat = (
        -30
        + 0.55 * abdomen
        - 0.15 * weight
        - 1.5 * wrist
        + 0.08 * age
        + 0.03 * hip
        + np.random.normal(0, 3, n)
    )
    bodyfat = np.clip(bodyfat, 1, 45)

    df = pd.DataFrame({
        "Age": age, "Weight": weight, "Height": height, "BMI": bmi,
        "Neck": neck, "Chest": chest, "Abdomen": abdomen, "Hip": hip,
        "Thigh": thigh, "Knee": knee, "Ankle": ankle, "Biceps": biceps,
        "Forearm": forearm, "Wrist": wrist, "BodyFat": bodyfat,
    })
    return df


def task2_bodyfat_regression():
    print("\n" + "=" * 70)
    print("Task 2: Bodyfat 回归 — 集成方法对比")
    print("=" * 70)

    df = generate_bodyfat(252)
    X = df.drop("BodyFat", axis=1).values
    y = df["BodyFat"].values

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scorers = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

    models = {
        "DecisionTree (md=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=50, random_state=42),
        "RandomForest (n=100)": RandomForestRegressor(
            n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
        "GBDT (n=100, lr=0.1)": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        mse = -np.mean(cv_results["test_neg_mean_squared_error"])
        mae = -np.mean(cv_results["test_neg_mean_absolute_error"])
        r2 = np.mean(cv_results["test_r2"])
        results[name] = {"MSE": mse, "MAE": mae, "R2": r2}

    header = ["Model", "MSE", "MAE", "R2"]
    rows = [[name] + [results[name][k] for k in ["MSE", "MAE", "R2"]] for name in models]
    print_table(header, rows)

    # --- Feature Importance from Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_names = df.drop("BodyFat", axis=1).columns.tolist()
    indices = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(feat_names)))
    axes[0].barh(range(len(feat_names)), importances[indices],
                 color=colors, edgecolor="white")
    axes[0].set_yticks(range(len(feat_names)))
    axes[0].set_yticklabels([feat_names[i] for i in indices])
    axes[0].set_xlabel("Importance")
    axes[0].set_title("Random Forest 特征重要性 (Bodyfat)")
    axes[0].invert_yaxis()

    # R2 comparison bar chart
    model_names = list(models.keys())
    r2_vals = [results[n]["R2"] for n in model_names]
    bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860"]
    axes[1].bar(model_names, r2_vals, color=bar_colors, edgecolor="white")
    axes[1].set_title("各模型 R2 对比")
    axes[1].set_ylabel("R2 Score")
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(r2_vals):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "10_ensemble_learning", "task2_bodyfat.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[Task 2 完成]")


# ===================================================================
# Task 3: Boston housing — 10 种方法
# ===================================================================
def generate_boston_data(n_samples=506):
    """Generate synthetic housing data mimicking the Boston dataset."""
    np.random.seed(42)
    n = n_samples
    X = np.column_stack([
        np.random.normal(3.6, 9, n),     # CRIM
        np.random.normal(11, 23, n),     # ZN
        np.random.normal(11, 7, n),      # INDUS
        np.random.choice([0, 1], n),     # CHAS
        np.random.normal(0.55, 0.12, n), # NOX
        np.random.normal(6.3, 0.7, n),   # RM
        np.random.normal(69, 28, n),     # AGE
        np.random.normal(3.8, 2.1, n),   # DIS
        np.random.choice(range(1, 25), n), # RAD
        np.random.normal(408, 170, n),    # TAX
        np.random.normal(18.5, 2.2, n),   # PTRATIO
        np.random.normal(357, 90, n),     # B
        np.random.normal(12.7, 7, n),     # LSTAT
    ])
    y = (0.5 * X[:, 5] + 0.3 * X[:, 0] - 0.4 * X[:, 12]
         + 0.2 * X[:, 7] + np.random.normal(0, 4, n))
    y = np.abs(y) + 5  # ensure positive median value
    return X, y


def task3_boston_housing():
    print("\n" + "=" * 70)
    print("Task 3: Boston 房价预测 — 10种方法大全对比")
    print("=" * 70)

    X, y = generate_boston_data(506)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=0.01)": Lasso(alpha=0.01, max_iter=5000),
        "DecisionTree (md=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "SVR (RBF)": SVR(kernel="rbf"),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "Bagging (n=50)": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=50, random_state=42),
        "RandomForest (n=100)": RandomForestRegressor(
            n_estimators=100, random_state=42),
        "AdaBoost (n=50)": AdaBoostRegressor(n_estimators=50, random_state=42),
        "GBDT (n=100,lr=0.1)": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        cv_results = cross_validate(
            model, X_scaled, y, cv=cv,
            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
            n_jobs=-1)
        mse = -np.mean(cv_results["test_neg_mean_squared_error"])
        mae = -np.mean(cv_results["test_neg_mean_absolute_error"])
        r2 = np.mean(cv_results["test_r2"])
        results[name] = {"MSE": mse, "MAE": mae, "R2": r2}

    # --- Custom regression accuracy ---
    # Train each model on full data and compute % within error bounds
    # Use ShuffleSplit for a clean estimate
    rs = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(rs.split(X_scaled, y))
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        # Relative error
        rel_err = np.abs((y_te - y_pred) / (y_te + 1e-8))
        acc_20 = np.mean(rel_err <= 0.20)
        acc_10 = np.mean(rel_err <= 0.10)
        results[name]["Acc@20%"] = acc_20
        results[name]["Acc@10%"] = acc_10

    # --- Comprehensive table ---
    metric_order = ["MSE", "MAE", "R2", "Acc@20%", "Acc@10%"]
    header = ["Model"] + metric_order
    rows = [[name] + [results[name][k] for k in metric_order] for name in models]
    print_table(header, rows)

    # --- GBDT Feature Importance ---
    gbdt = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42)
    gbdt.fit(X_scaled, y)
    feat_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                   "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    importances = gbdt.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Feature importance
    axes[0].barh(range(len(feat_names)), importances[idx],
                 color=plt.cm.plasma(np.linspace(0, 1, len(feat_names))),
                 edgecolor="white")
    axes[0].set_yticks(range(len(feat_names)))
    axes[0].set_yticklabels([feat_names[i] for i in idx])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Importance")
    axes[0].set_title("GBDT 特征重要性 (Boston Housing)")

    # Top models R2 comparison
    sorted_models = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)
    top_names = [n for n, _ in sorted_models]
    top_r2 = [results[n]["R2"] for n in top_names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_names)))
    axes[1].barh(range(len(top_names)), top_r2[::-1],
                 color=colors[::-1], edgecolor="white")
    axes[1].set_yticks(range(len(top_names)))
    axes[1].set_yticklabels(top_names[::-1])
    axes[1].set_xlabel("R2 Score")
    axes[1].set_title("10 种方法 R2 排名 (降序)")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "..", "10_ensemble_learning", "task3_boston.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # Best model
    best = sorted_models[0]
    print(f"\n[最佳模型] {best[0]} — R2 = {best[1]['R2']:.4f}, "
          f"MSE = {best[1]['MSE']:.4f}, MAE = {best[1]['MAE']:.4f}")
    print("[Task 3 完成]")


# ===================================================================
# main
# ===================================================================
if __name__ == "__main__":
    task1_breast_cancer_ensemble()
    task2_bodyfat_regression()
    task3_boston_housing()

    print("\n" + "=" * 70)
    print("10_ensemble_learning 全部任务完成!")
    print("=" * 70)
