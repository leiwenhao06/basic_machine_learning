"""
支持向量机实践 (SVM Practice)
=============================
Task 1: Iris 二分类 (versicolor vs virginica) — LinearSVC + 不同C值, 决策边界, 5折CV
Task 2: 乳腺癌分类 — 4种kernel对比, GridSearchCV, PCA可视化, 全指标
Task 3: Pima糖尿病 — 多分类器比较, 零值中位数填充, ROC曲线, 柱状图对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# ======================== 中文 / 字体设置 ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局随机种子
RANDOM_STATE = 42


# ================================================================
#     Task 1: Iris 二分类 (versicolor vs virginica) — LinearSVC
# ================================================================
def task1_iris_svm():
    """LinearSVC + StandardScaler, 不同C值, 决策边界, 5折CV"""
    print("=" * 60)
    print("Task 1: Iris 二分类 (versicolor vs virginica) — LinearSVC")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # 只取 versicolor(1) 和 virginica(2)
    mask = y != 0  # 排除 setosa
    X_binary = X[mask]
    y_binary = y[mask]
    # 将标签改为 0/1
    y_binary = np.where(y_binary == 1, 0, 1)  # versicolor=0, virginica=1
    target_names = ['versicolor', 'virginica']

    print(f"样本数: {X_binary.shape[0]}")
    print(f"特征数: {X_binary.shape[1]}")
    print(f"类别分布: versicolor={np.sum(y_binary==0)}, virginica={np.sum(y_binary==1)}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_binary)

    # 使用前2个主成分用于可视化 (仅用于绘图, 不改变分类)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # PCA 上的 LinearSVC (仅用于可视化决策边界)
    # 同时也在原始4维空间上做评估
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.3, random_state=RANDOM_STATE, stratify=y_binary
    )

    C_values = [0.01, 0.1, 1, 10, 100]
    print(f"\n不同 C 值的 5 折交叉验证结果 (原始4维空间):")

    cv_results = {}
    for C in C_values:
        svc = LinearSVC(C=C, max_iter=10000, dual=False, random_state=RANDOM_STATE)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(svc, X_scaled, y_binary, cv=skf, scoring='accuracy')
        cv_results[C] = scores
        print(f"  C={C:<8}  CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ---- 可视化: 每个 C 值的决策边界 (基于 PCA 2D) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(
        X_pca, y_binary, test_size=0.3, random_state=RANDOM_STATE, stratify=y_binary
    )

    for idx, C in enumerate(C_values):
        ax = axes[idx]

        # 在2维PCA空间训练
        clf = LinearSVC(C=C, max_iter=10000, dual=False, random_state=RANDOM_STATE)
        clf.fit(X_pca_train, y_pca_train)

        # 网格绘制决策边界
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

        # 支持向量 (LinearSVC 没有 support_vectors_ 属性, 这里在PCA空间标注全部点)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_binary,
                             cmap='coolwarm', edgecolors='k', s=60,
                             alpha=0.8)

        acc = accuracy_score(y_pca_test, clf.predict(X_pca_test))
        ax.set_title(f'C = {C}  (测试准确率={acc:.3f})', fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    # 隐藏多余的子图
    axes[5].axis('off')

    plt.suptitle('Iris 二分类 — LinearSVC 不同 C 值决策边界 (PCA 2D)', fontsize=14)
    plt.tight_layout()
    plt.savefig('08_svm_practice/task1_iris_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 08_svm_practice/task1_iris_svm.png\n")


# ================================================================
#        Task 2: 乳腺癌分类 — 4种kernel对比 + GridSearchCV
# ================================================================
def task2_breast_cancer_svm():
    """比较 linear, poly, rbf, sigmoid; GridSearchCV; PCA可视化"""
    print("=" * 60)
    print("Task 2: 乳腺癌分类 — 4种 Kernel SVM 对比")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # ---- 四种 Kernel 对比 ----
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = {}

    print("\n四种 Kernel SVM 测试结果:")
    print("-" * 70)
    print(f"{'Kernel':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)

    for kernel in kernels:
        svc = SVC(kernel=kernel, random_state=RANDOM_STATE, probability=True)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        y_proba = svc.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[kernel] = {
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc
        }
        print(f"{kernel:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")

    # ---- GridSearchCV for RBF ----
    print("\nRBF GridSearchCV (gamma + C):")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
    }
    grid = GridSearchCV(
        SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"  最佳参数: {grid.best_params_}")
    print(f"  最佳交叉验证准确率: {grid.best_score_:.4f}")

    best_svc = grid.best_estimator_
    y_pred_best = best_svc.predict(X_test)
    y_proba_best = best_svc.predict_proba(X_test)[:, 1]
    print(f"  测试集准确率: {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"  测试集AUC:    {roc_auc_score(y_test, y_proba_best):.4f}")

    # ---- PCA 2D 可视化 ----
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 四种 kernel 的决策边界 (在 PCA 空间)
    ax = axes[0]
    for kernel in kernels:
        clf = SVC(kernel=kernel, random_state=RANDOM_STATE)
        clf.fit(X_pca, y)

        # 绘制决策边界概要 (只画一条 decision boundary line)
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2,
                   linestyles=['-', '--', '-.', ':'][kernels.index(kernel)],
                   colors=['blue', 'orange', 'green', 'red'], zorder=3)

    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm',
               edgecolors='k', s=40, alpha=0.6)
    # 手动图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='-', label='linear'),
        Line2D([0], [0], color='orange', linestyle='--', label='poly'),
        Line2D([0], [0], color='green', linestyle='-.', label='rbf'),
        Line2D([0], [0], color='red', linestyle=':', label='sigmoid'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_title(f'四种 Kernel 决策边界 (PCA 2D, 解释方差={pca.explained_variance_ratio_.sum():.3f})', fontsize=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 子图2: 指标柱状图
    ax2 = axes[1]
    metrics_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
    x_pos = np.arange(len(metrics_names))
    width = 0.2

    for i, kernel in enumerate(kernels):
        values = [results[kernel][m] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']]
        ax2.bar(x_pos + i * width, values, width, label=kernel, alpha=0.8)

    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylabel('分数')
    ax2.set_title('四种 Kernel SVM 指标对比', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.8, 1.02)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('乳腺癌分类 — SVM Kernel 对比 + PCA 可视化', fontsize=14)
    plt.tight_layout()
    plt.savefig('08_svm_practice/task2_breast_cancer_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 08_svm_practice/task2_breast_cancer_svm.png\n")


# ================================================================
#          Task 3: Pima Diabetes 多分类器比较
# ================================================================
def load_or_create_pima():
    """加载 pima-diabetes.csv, 如果不存在则生成样本数据"""
    import os
    path = 'data/pima-diabetes.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"从 {path} 加载数据, 形状: {df.shape}")
        return df

    print("pima-diabetes.csv 未找到, 生成 768 条模拟数据...")
    np.random.seed(RANDOM_STATE)
    n = 768

    # 模拟 Pima 数据的统计特征
    pregnancies = np.random.poisson(3, n)
    glucose = np.random.normal(120, 30, n)
    blood_pressure = np.random.normal(70, 12, n)
    skin_thickness = np.random.normal(20, 10, n)
    insulin = np.random.exponential(80, n)
    bmi = np.random.normal(32, 7, n)
    dpf = np.random.gamma(0.5, 1, n)  # diabetes pedigree function
    age = np.random.randint(21, 80, n)

    # 基于特征生成 Outcome (模拟糖尿病概率)
    logit = (
        -8.0
        + 0.05 * glucose
        + 0.01 * bmi
        + 0.02 * age
        + 0.1 * dpf
        + 0.005 * insulin
        + 0.005 * skin_thickness
        + 0.005 * blood_pressure
        + 0.02 * pregnancies
        + np.random.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    outcome = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Outcome': outcome,
    })
    # 四舍五入
    df = df.round(1)
    df['Pregnancies'] = df['Pregnancies'].astype(int)
    df['Outcome'] = df['Outcome'].astype(int)

    os.makedirs('data', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"模拟数据已保存至 {path}, 形状: {df.shape}")
    return df


def task3_pima_diabetes():
    """Pima 糖尿病 — 零值中位数填充, 多分类器比较, ROC, 柱状图"""
    print("=" * 60)
    print("Task 3: Pima Diabetes 多分类器比较")
    print("=" * 60)

    df = load_or_create_pima()

    # ---- 零值中位数填充 ----
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print(f"\n零值填充前统计 (各列零值数量):")
    for col in zero_cols:
        n_zero = (df[col] <= 0).sum()
        print(f"  {col}: {n_zero} 个零值")

    for col in zero_cols:
        median_val = df.loc[df[col] > 0, col].median()
        df.loc[df[col] <= 0, col] = median_val

    print(f"\n零值填充后统计 (各列零值数量):")
    for col in zero_cols:
        n_zero = (df[col] <= 0).sum()
        print(f"  {col}: {n_zero} 个零值")

    # ---- 准备数据 ----
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # ---- 定义分类器 ----
    classifiers = {
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        'DecisionTree (max_depth=5)': DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
    }

    # ---- 评估 ----
    all_results = {}
    print("\n" + "-" * 80)
    print(f"{'分类器':<25} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'AUC-ROC':<8}")
    print("-" * 80)

    fig, (ax_roc, ax_bar) = plt.subplots(1, 2, figsize=(16, 7))

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        all_results[name] = {
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc
        }
        print(f"{name:<25} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {auc:<8.4f}")

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.4f})')

    # ROC 图
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax_roc.set_xlabel('假阳率 (FPR)', fontsize=12)
    ax_roc.set_ylabel('真阳率 (TPR)', fontsize=12)
    ax_roc.set_title('ROC 曲线对比 — Pima Diabetes', fontsize=13)
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    # ---- 柱状图对比 ----
    metrics_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
    clf_names = list(classifiers.keys())
    x_pos = np.arange(len(metrics_names))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, results) in enumerate(all_results.items()):
        values = [results[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']]
        ax_bar.bar(x_pos + i * width, values, width,
                   label=name, color=colors[i], alpha=0.85)

    ax_bar.set_xticks(x_pos + width * 2)
    ax_bar.set_xticklabels(metrics_names)
    ax_bar.set_ylabel('分数')
    ax_bar.set_title('多分类器指标对比 — Pima Diabetes', fontsize=13)
    ax_bar.legend(fontsize=7, loc='lower left')
    ax_bar.set_ylim(0.5, 1.02)
    ax_bar.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Pima Diabetes — 多分类器比较', fontsize=14)
    plt.tight_layout()
    plt.savefig('08_svm_practice/task3_pima_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 08_svm_practice/task3_pima_comparison.png")

    # ---- 结论 ----
    print("\n" + "=" * 60)
    print("结论:")
    # 按 AUC 排名
    ranked = sorted(all_results.items(), key=lambda x: x[1]['auc'], reverse=True)
    print(f"  最佳方法 (AUC): {ranked[0][0]} (AUC={ranked[0][1]['auc']:.4f})")
    for i, (name, r) in enumerate(ranked):
        print(f"    {i+1}. {name}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}, "
              f"Acc={r['accuracy']:.4f}")
    print()


# ================================================================
if __name__ == "__main__":
    import os
    os.makedirs('08_svm_practice', exist_ok=True)

    task1_iris_svm()
    task2_breast_cancer_svm()
    task3_pima_diabetes()
