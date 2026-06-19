"""
信用卡欺诈检测 — Credit Card Fraud Detection
==============================================
生成模拟信用卡交易数据，通过逻辑回归和KNN进行欺诈检测建模，
重点关注召回率（Recall），因为漏掉欺诈交易的成本远高于误报。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# 1. Generate synthetic credit card transaction data
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples=10000, fraud_ratio=0.005, random_state=42):
    """
    Generate synthetic credit card transaction data with imbalanced classes.

    Parameters
    ----------
    n_samples : int
        Total number of transactions.
    fraud_ratio : float
        Proportion of fraud cases (~0.5%).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 29)
        Feature matrix: Time, V1-V28 (but Time is dropped later).
    y : ndarray of shape (n_samples,)
        Labels: 0 = normal, 1 = fraud.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    # --- Normal transactions ---
    normal = rng.normal(loc=0, scale=1, size=(n_normal, 28))          # V1–V28
    normal_amount = np.abs(rng.exponential(scale=80, size=n_normal)) + 1
    normal_time = rng.integers(0, 172800, size=n_normal)              # seconds in 2 days

    # --- Fraud transactions ---
    # Make fraud slightly different: shift some features & make amounts smaller
    fraud = rng.normal(loc=0, scale=1, size=(n_fraud, 28))
    fraud[:, [2, 4, 11, 14]] += rng.normal(loc=3, scale=1.5, size=(n_fraud, 4))
    fraud[:, [7, 9]] -= rng.normal(loc=2, scale=1, size=(n_fraud, 2))
    fraud_amount = np.abs(rng.exponential(scale=30, size=n_fraud)) + 0.5
    fraud_time = rng.integers(0, 172800, size=n_fraud)

    # Combine
    X_normal = np.column_stack((normal_time.reshape(-1, 1), normal, normal_amount.reshape(-1, 1)))
    X_fraud = np.column_stack((fraud_time.reshape(-1, 1), fraud, fraud_amount.reshape(-1, 1)))

    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def get_feature_names():
    """Return list of feature column names."""
    return ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


# ---------------------------------------------------------------------------
# 2. Data exploration
# ---------------------------------------------------------------------------
def explore_data(X, y, feature_names):
    """Print class distribution, feature ranges, and check for missing values."""
    print("=" * 60)
    print("数据探索 Data Exploration")
    print("=" * 60)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n类别分布 Class Distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} ({100 * c / len(y):.4f}%)")

    # Feature value ranges
    print(f"\n特征取值范围 Feature Value Ranges:")
    for i, name in enumerate(feature_names):
        print(f"  {name:>8s}: min={X[:, i].min():+.4f}, max={X[:, i].max():+.4f}, "
              f"mean={X[:, i].mean():+.4f}, std={X[:, i].std():.4f}")

    # Missing values
    missing = np.isnan(X).sum()
    print(f"\n缺失值 Missing Values: {missing}")

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['正常 Normal', '欺诈 Fraud'], [counts[0], counts[1]],
                  color=['#2ecc71', '#e74c3c'])
    ax.set_title('信用卡交易类别分布\nCredit Card Transaction Class Distribution')
    ax.set_ylabel('样本数量 Sample Count')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val}\n({100*val/len(y):.2f}%)', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('practice/04_credit_card_fraud/01_class_distribution.png', dpi=150)
    plt.close()
    print("\n[图表已保存] 01_class_distribution.png")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(X, y, feature_names):
    """
    Drop Time column, apply StandardScaler to Amount and V features.
    Return scaled feature matrix and corresponding feature names.
    """
    # Drop Time column (index 0)
    X_no_time = X[:, 1:]  # V1-V28 + Amount
    selected_features = feature_names[1:]  # ['V1', ..., 'V28', 'Amount']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_no_time)
    return X_scaled, selected_features, scaler


# ---------------------------------------------------------------------------
# 4. Feature selection via correlation
# ---------------------------------------------------------------------------
def select_features_by_correlation(X_scaled, y, feature_names, threshold=0.1):
    """Keep only features whose absolute Pearson correlation with Class >= threshold."""
    corr_with_class = []
    for i, name in enumerate(feature_names):
        corr_val = np.corrcoef(X_scaled[:, i], y)[0, 1]
        corr_with_class.append((name, corr_val))

    print(f"\n各特征与Class的相关系数:")
    kept = []
    kept_idx = []
    for i, (name, corr_val) in enumerate(corr_with_class):
        status = "KEEP" if abs(corr_val) >= threshold else "DROP"
        if status == "KEEP":
            kept.append(name)
            kept_idx.append(i)
        print(f"  {name:>8s}: corr={corr_val:+.4f}  [{status}]")

    X_selected = X_scaled[:, kept_idx]
    print(f"\n保留特征: {kept} (共{len(kept)}个)")
    return X_selected, kept


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(X_scaled, y, feature_names, selected_features):
    """Plot correlation heatmap of selected features + Class."""
    # Build a small dataframe-like matrix for selected features + Class
    all_names = selected_features + ['Class']
    selected_idx = [i for i, name in enumerate(feature_names) if name in selected_features]
    data = np.column_stack((X_scaled[:, selected_idx], y.reshape(-1, 1)))
    corr_matrix = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, xticklabels=all_names, yticklabels=all_names,
                linewidths=0.5, ax=ax)
    ax.set_title('特征相关性热力图 Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('practice/04_credit_card_fraud/02_correlation_heatmap.png', dpi=150)
    plt.close()
    print("[图表已保存] 02_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 6. Model training and evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train the model and return comprehensive evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"模型: {model_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  (准确率): {acc:.4f}")
    print(f"  Precision (精确率): {prec:.4f}")
    print(f"  Recall    (召回率): {rec:.4f}")
    print(f"  F1-score  (F1分数): {f1:.4f}")
    if not np.isnan(auc):
        print(f"  AUC-ROC   (AUC值):  {auc:.4f}")

    return {
        'model': model, 'name': model_name,
        'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'cm': cm,
    }


# ---------------------------------------------------------------------------
# 7. Visualization: confusion matrices and ROC curve comparison
# ---------------------------------------------------------------------------
def plot_confusion_matrices(results):
    """Plot side-by-side confusion matrix heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, results):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常 Normal', '欺诈 Fraud'],
                    yticklabels=['正常 Normal', '欺诈 Fraud'],
                    ax=ax)
        ax.set_xlabel('预测 Predicted')
        ax.set_ylabel('真实 Actual')
        ax.set_title(f"{res['name']}\n"
                     f"Acc={res['accuracy']:.3f} Prec={res['precision']:.3f} "
                     f"Rec={res['recall']:.3f} F1={res['f1']:.3f}")
    plt.tight_layout()
    plt.savefig('practice/04_credit_card_fraud/03_confusion_matrices.png', dpi=150)
    plt.close()
    print("[图表已保存] 03_confusion_matrices.png")


def plot_roc_curves(results, y_test):
    """Plot overlaid ROC curves for model comparison."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#3498db', '#e74c3c']
    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['name']} (AUC={res['auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
    ax.set_xlabel('假阳性率 False Positive Rate')
    ax.set_ylabel('真阳性率 True Positive Rate (召回率 Recall)')
    ax.set_title('ROC曲线对比 ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('practice/04_credit_card_fraud/04_roc_comparison.png', dpi=150)
    plt.close()
    print("[图表已保存] 04_roc_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("信用卡欺诈检测 Credit Card Fraud Detection")
    print("=" * 60)

    # --- Generate data ---
    X, y = generate_synthetic_data(n_samples=10000, fraud_ratio=0.005, random_state=42)
    feature_names = get_feature_names()
    print(f"\n数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # --- Exploration ---
    explore_data(X, y, feature_names)

    # --- Preprocessing ---
    X_scaled, selected_fnames, scaler = preprocess(X, y, feature_names)

    # --- Feature selection ---
    X_selected, kept_features = select_features_by_correlation(
        X_scaled, y, selected_fnames, threshold=0.1
    )

    # --- Correlation heatmap ---
    plot_correlation_heatmap(X_scaled, y, selected_fnames, kept_features)

    # --- Train/Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n训练集 Train: {X_train.shape[0]}, 测试集 Test: {X_test.shape[0]}")
    print(f"测试集欺诈数: {y_test.sum()} ({100 * y_test.mean():.4f}%)")

    # --- Handle imbalance with SMOTE ---
    print("\n应用SMOTE处理类别不平衡...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE后训练集: {X_train_res.shape[0]} (正常: {(y_train_res==0).sum()}, "
          f"欺诈: {(y_train_res==1).sum()})")

    # --- Train & evaluate models ---
    lr = LogisticRegression(max_iter=2000, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    results = []
    results.append(evaluate_model(lr, "Logistic Regression", X_train_res, X_test, y_train_res, y_test))
    results.append(evaluate_model(knn, "KNN (k=5)", X_train_res, X_test, y_train_res, y_test))

    # --- Visualizations ---
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)

    # --- Conclusion ---
    print("\n" + "=" * 60)
    print("结论 Conclusion")
    print("=" * 60)
    lr_rec = results[0]['recall']
    knn_rec = results[1]['recall']

    if lr_rec >= knn_rec:
        better = "Logistic Regression"
        better_rec = lr_rec
        other_rec = knn_rec
    else:
        better = "KNN (k=5)"
        better_rec = knn_rec
        other_rec = lr_rec

    print(f"\n  逻辑回归召回率: {lr_rec:.4f}")
    print(f"  KNN召回率:      {knn_rec:.4f}")
    print(f"\n  在信用卡欺诈检测中，**召回率 (Recall)** 是最关键的指标——")
    print(f"  漏掉一笔欺诈交易（假阴性）的代价远高于误报一笔正常交易（假阳性）。")
    print(f"  {better} 的召回率 ({better_rec:.4f}) 更高，能捕获更多真实的欺诈案例，")
    print(f"  因此更适合作为欺诈检测模型。")
    if better == "Logistic Regression":
        print(f"  此外，逻辑回归具有更好的可解释性，便于理解各个特征对欺诈概率的贡献。")
    print(f"\n  建议: 在生产环境中使用 {better}，并持续监控召回率的变化趋势。")
    print(f"  同时可结合规则引擎对高风险交易进行人工复核。")
