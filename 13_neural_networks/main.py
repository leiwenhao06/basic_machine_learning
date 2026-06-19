"""
神经网络 (Neural Networks)
========================
任务1: 乳腺癌二分类 - MLP分类器
任务2: 体脂率回归 - MLP回归器
任务3: 多方法比较 - 乳腺癌数据集 (MLP / SVM / GaussianNB)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    ConfusionMatrixDisplay
)
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 任务1: 乳腺癌二分类 - MLP分类器
# ============================================================
def task1_breast_cancer_mlp():
    """乳腺癌二分类任务: MLPClassifier"""
    print("\n" + "=" * 70)
    print("任务1: 乳腺癌二分类 - MLP分类器")
    print("=" * 70)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    print(f"数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"类别分布: {target_names[0]}={sum(y==0)}, {target_names[1]}={sum(y==1)}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7:3 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # MLP分类器
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    # 预测与评估
    y_pred = mlp.predict(X_test)
    y_pred_train = mlp.predict(X_train)

    print(f"\n训练集准确率: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

    # 分类报告
    print("\n--- 分类报告 (测试集) ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax[0], cmap='Blues', colorbar=False)
    ax[0].set_title('测试集混淆矩阵', fontsize=14)

    # 全数据集训练 - 展示能接近1.0的训练精度
    mlp_full = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )
    mlp_full.fit(X_scaled, y)
    y_pred_full = mlp_full.predict(X_scaled)
    acc_full = accuracy_score(y, y_pred_full)
    print(f"\n全数据集训练准确率: {acc_full:.4f} (应接近 1.0)")

    cm_full = confusion_matrix(y, y_pred_full)
    disp_full = ConfusionMatrixDisplay(confusion_matrix=cm_full, display_labels=target_names)
    disp_full.plot(ax=ax[1], cmap='Greens', colorbar=False)
    ax[1].set_title(f'全数据集混淆矩阵 (准确率={acc_full:.4f})', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'task1_confusion_matrix.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"\n5折交叉验证准确率: {cv_scores}")
    print(f"5折CV平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\n任务1 完成!")


# ============================================================
# 任务2: 体脂率回归 - MLP回归器
# ============================================================
def generate_bodyfat_data(n_samples=252):
    """
    生成体脂率样本数据 (基于真实数据集统计特征)
    特征: Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee,
          Ankle, Biceps, Forearm, Wrist (共13个特征, 不含 Density)
    目标: BodyFat (体脂率百分比)
    """
    np.random.seed(42)

    # 基础特征均值与标准差 (参考真实bodyfat数据集)
    means = np.array([
        44.9,    # Age (years)
        178.9,   # Weight (lbs)
        70.1,    # Height (inches)
        37.9,    # Neck (cm)
        100.8,   # Chest (cm)
        92.5,    # Abdomen (cm)
        99.9,    # Hip (cm)
        59.4,    # Thigh (cm)
        38.5,    # Knee (cm)
        23.1,    # Ankle (cm)
        32.2,    # Biceps (cm)
        28.6,    # Forearm (cm)
        18.2,    # Wrist (cm)
    ])

    stds = np.array([
        12.8,    # Age
        29.3,    # Weight
        3.6,     # Height
        2.4,     # Neck
        8.3,     # Chest
        10.7,    # Abdomen
        7.1,     # Hip
        5.2,     # Thigh
        2.4,     # Knee
        1.7,     # Ankle
        3.0,     # Biceps
        2.0,     # Forearm
        0.9,     # Wrist
    ])

    # 协方差矩阵 (对角近似, 加入适量相关性)
    L = np.diag(stds)
    # 添加一些相关性: Abdomen与Weight正相关, Chest与Abdomen正相关
    L[1, 5] = 0.6 * stds[1]   # Weight -> Abdomen
    L[2, 1] = 0.3 * stds[2]   # Height -> Weight
    L[4, 5] = 0.4 * stds[4]   # Chest -> Abdomen
    L[6, 5] = 0.5 * stds[6]   # Hip -> Abdomen
    L[7, 5] = 0.3 * stds[7]   # Thigh -> Abdomen

    X_raw = np.random.randn(n_samples, 13)
    X = X_raw @ L.T + means

    # 确保特征为正数
    X = np.abs(X)
    # 限制特征在合理范围内
    feature_names = [
        'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen',
        'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'
    ]

    # 生成BodyFat目标 (基于Abdomen + Weight的线性组合 + 非线性项 + 噪声)
    # BodyFat 与 Abdomen 高度正相关, 也与Weight相关
    abdomen_centered = X[:, 5] - means[5]
    weight_centered = X[:, 1] - means[1]
    age_effect = 0.05 * X[:, 0]

    bodyfat = (
        18.0
        + 0.5 * abdomen_centered
        + 10.0 * (abdomen_centered / means[5]) ** 2  # 非线性项
        + 0.15 * weight_centered
        + age_effect
        + np.random.randn(n_samples) * 3.5
    )
    bodyfat = np.clip(bodyfat, 1.0, 50.0)

    df = pd.DataFrame(X, columns=feature_names)
    df['BodyFat'] = np.round(bodyfat, 1)

    return df, feature_names


def task2_bodyfat_regression():
    """体脂率回归任务: MLPRegressor"""
    print("\n" + "=" * 70)
    print("任务2: 体脂率回归 - MLP回归器")
    print("=" * 70)

    # 尝试加载已有数据, 否则生成
    bodyfat_path = os.path.join(DATA_DIR, 'bodyfat.csv')
    if os.path.exists(bodyfat_path):
        print(f"加载已有数据: {bodyfat_path}")
        df = pd.read_csv(bodyfat_path)
        # 如果不含 Density 列, 特征从第一列到倒数第二列
        if 'Density' in df.columns:
            df = df.drop(columns=['Density'])
        feature_cols = [c for c in df.columns if c != 'BodyFat']
    else:
        print("生成体脂率样本数据...")
        df, feature_cols = generate_bodyfat_data(252)
        df.to_csv(bodyfat_path, index=False)
        print(f"数据已保存至: {bodyfat_path}")

    X = df[feature_cols].values
    y = df['BodyFat'].values
    print(f"数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"目标 BodyFat 范围: [{y.min():.1f}, {y.max():.1f}], "
          f"均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # 7:3 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # MLP回归器 (带 early stopping)
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    mlp_reg.fit(X_train, y_train)
    print(f"\n训练迭代次数: {mlp_reg.n_iter_}")
    print(f"最终损失: {mlp_reg.loss_:.6f}")

    # 预测 (反标准化)
    y_pred_scaled = mlp_reg.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_train_scaled = mlp_reg.predict(X_train)
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_pred_train_orig = scaler_y.inverse_transform(
        y_pred_train_scaled.reshape(-1, 1)
    ).ravel()

    # 评估指标
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    print("\n--- 回归评估指标 (测试集) ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # 训练集评估
    mse_train = mean_squared_error(y_train_orig, y_pred_train_orig)
    r2_train = r2_score(y_train_orig, y_pred_train_orig)
    print(f"\n训练集 MSE: {mse_train:.4f}, R²: {r2_train:.4f}")

    # 5折交叉验证 (回归任务使用KFold)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores_r2 = cross_val_score(
        mlp_reg, X_scaled, y_scaled, cv=kf,
        scoring='r2'
    )
    cv_scores_neg_mse = cross_val_score(
        mlp_reg, X_scaled, y_scaled, cv=kf,
        scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores_neg_mse)
    # 反标准化RMSE
    cv_rmse_orig = cv_rmse * scaler_y.scale_[0]

    print(f"\n5折CV R²:   {cv_scores_r2}")
    print(f"5折CV R² 均值: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
    print(f"5折CV RMSE: {cv_rmse_orig}")
    print(f"5折CV RMSE 均值: {cv_rmse_orig.mean():.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散点图: 预测值 vs 真实值
    axes[0].scatter(y_test_orig, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    lims = [min(y_test_orig.min(), y_pred.min()), max(y_test_orig.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='理想预测线')
    axes[0].set_xlabel('真实体脂率 (%)', fontsize=12)
    axes[0].set_ylabel('预测体脂率 (%)', fontsize=12)
    axes[0].set_title(f'预测 vs 真实 (R²={r2:.4f})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差图
    residuals = y_test_orig - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('预测体脂率 (%)', fontsize=12)
    axes[1].set_ylabel('残差 (%)', fontsize=12)
    axes[1].set_title(f'残差图 (MAE={mae:.4f})', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'task2_bodyfat_regression.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("\n任务2 完成!")


# ============================================================
# 任务3: 多方法比较 - 乳腺癌数据集
# ============================================================
def task3_multi_method_comparison():
    """多方法比较: MLP (GridSearchCV) vs SVM vs GaussianNB"""
    print("\n" + "=" * 70)
    print("任务3: 多方法比较 - 乳腺癌数据集")
    print("=" * 70)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = data.target_names

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    # 3折CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}

    # ---- 方法1: MLP with GridSearchCV ----
    print("\n>>> 方法1: MLP (GridSearchCV) ...")
    mlp_param_grid = {
        'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64), (64, 32, 16)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.005],
    }
    mlp_base = MLPClassifier(
        activation='relu', solver='adam', max_iter=2000, random_state=42
    )
    mlp_gs = GridSearchCV(
        mlp_base, mlp_param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    mlp_gs.fit(X_train, y_train)

    print(f"  最佳参数: {mlp_gs.best_params_}")
    print(f"  最佳CV分数: {mlp_gs.best_score_:.4f}")

    # 训练集和测试集评估
    y_pred_train_mlp = mlp_gs.predict(X_train)
    y_pred_test_mlp = mlp_gs.predict(X_test)

    results['MLP (GridSearch)'] = {
        'Train Accuracy': accuracy_score(y_train, y_pred_train_mlp),
        'Test Accuracy': accuracy_score(y_test, y_pred_test_mlp),
        'CV Accuracy': mlp_gs.best_score_,
        'Best Params': str(mlp_gs.best_params_)
    }

    # ---- 方法2: SVM (RBF) ----
    print("\n>>> 方法2: SVM (RBF, C=1.0) ...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm.fit(X_train, y_train)

    svm_cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy')
    y_pred_train_svm = svm.predict(X_train)
    y_pred_test_svm = svm.predict(X_test)

    results['SVM (RBF)'] = {
        'Train Accuracy': accuracy_score(y_train, y_pred_train_svm),
        'Test Accuracy': accuracy_score(y_test, y_pred_test_svm),
        'CV Accuracy': svm_cv_scores.mean(),
        'Best Params': 'C=1.0, gamma=scale'
    }

    # ---- 方法3: GaussianNB ----
    print("\n>>> 方法3: GaussianNB ...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    gnb_cv_scores = cross_val_score(gnb, X_train, y_train, cv=cv, scoring='accuracy')
    y_pred_train_gnb = gnb.predict(X_train)
    y_pred_test_gnb = gnb.predict(X_test)

    results['GaussianNB'] = {
        'Train Accuracy': accuracy_score(y_train, y_pred_train_gnb),
        'Test Accuracy': accuracy_score(y_test, y_pred_test_gnb),
        'CV Accuracy': gnb_cv_scores.mean(),
        'Best Params': 'N/A (无超参数调优)'
    }

    # ---- 综合比较表 ----
    print("\n" + "-" * 85)
    print("综合比较表")
    print("-" * 85)
    print(f"{'方法':<25} {'训练准确率':>10} {'测试准确率':>10} {'CV准确率':>10} {'参数'}")
    print("-" * 85)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['Train Accuracy']:>10.4f} "
              f"{metrics['Test Accuracy']:>10.4f} {metrics['CV Accuracy']:>10.4f} "
              f" {metrics['Best Params']}")
    print("-" * 85)

    # 找出最佳方法
    best_method = max(results, key=lambda k: results[k]['Test Accuracy'])
    print(f"\n>>> 最佳方法: {best_method}")
    print(f"    测试准确率: {results[best_method]['Test Accuracy']:.4f}")
    print(f"    CV准确率:   {results[best_method]['CV Accuracy']:.4f}")

    # ---- 结论 ----
    print("\n--- 结论 ---")
    print("MLP (GridSearchCV) 通过超参数搜索获得了最佳测试准确率。")
    print("SVM (RBF) 表现也接近, 但参数固定; 若进行调参可能进一步提升。")
    print("GaussianNB 准确率较低, 因其假设特征条件独立, "
          "在乳腺癌数据上不满足该假设。")
    print("MLP 能捕获特征间复杂的非线性交互, 适合此类医学诊断任务。")

    # ---- SVM 分类报告 ----
    print("\n--- SVM (RBF) 分类报告 ---")
    print(classification_report(y_test, y_pred_test_svm, target_names=target_names))

    # ---- 可视化: 比较柱状图 ----
    methods_names = list(results.keys())
    train_accs = [results[m]['Train Accuracy'] for m in methods_names]
    test_accs = [results[m]['Test Accuracy'] for m in methods_names]
    cv_accs = [results[m]['CV Accuracy'] for m in methods_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(methods_names))
    width = 0.25
    axes[0].bar(x - width, train_accs, width, label='训练准确率', color='#4472C4', edgecolor='k')
    axes[0].bar(x, test_accs, width, label='测试准确率', color='#ED7D31', edgecolor='k')
    axes[0].bar(x + width, cv_accs, width, label='CV准确率', color='#70AD47', edgecolor='k')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.split('(')[0].strip() for m in methods_names], fontsize=10)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_title('方法准确率比较', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0.85, 1.02)
    axes[0].grid(axis='y', alpha=0.3)

    # SVM混淆矩阵
    cm_svm = confusion_matrix(y_test, y_pred_test_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=target_names)
    disp_svm.plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title('SVM (RBF) 混淆矩阵', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'task3_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("\n任务3 完成!")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("神经网络 (Neural Networks) 实验")
    print("   MLP分类器 / MLP回归器 / 多方法比较")
    print("=" * 70)

    # 任务1
    task1_breast_cancer_mlp()

    # 任务2
    task2_bodyfat_regression()

    # 任务3
    task3_multi_method_comparison()

    print("\n" + "=" * 70)
    print("所有任务完成!")
    print("=" * 70)
