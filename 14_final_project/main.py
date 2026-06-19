"""
期末综合项目 (Final Project)
===========================
第一部分 (40%): 连续值预测 — 地深肥沃指数 (线性回归 vs SVR)
第二部分 (40%): 分类 — 乳腺癌诊断 (GaussianNB vs RandomForest)
第三部分 (20%): 不平衡数据与降维 (多分类器对比 + PCA + t-SNE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
MODULE_DIR = BASE_DIR


# ============================================================
# 数据生成工具函数
# ============================================================
def generate_fertility_depth_data(n_samples=50):
    """
    生成地深肥沃指数样本数据
    关系: 肥沃指数随深度增加呈非线性下降趋势
    fertility = 95*exp(-depth/40) + 8*sin(depth/15) + noise
    """
    np.random.seed(42)
    depth = np.linspace(0.5, 200, n_samples)
    # 添加少量随机偏移使数据点不完全等距
    depth += np.random.uniform(-1.5, 1.5, size=n_samples)
    depth = np.clip(depth, 0.1, 210)

    # 非线性关系: 指数衰减 + 正弦波动 + 噪声
    fertility = (
        95 * np.exp(-depth / 40)
        + 8 * np.sin(depth / 15)
        + np.random.normal(0, 3, size=n_samples)
    )
    fertility = np.clip(fertility, 0, 100)

    df = pd.DataFrame({
        '深度(m)': np.round(depth, 1),
        '肥沃指数': np.round(fertility, 2)
    })
    df = df.sort_values('深度(m)').reset_index(drop=True)
    return df


def save_fertility_depth_data(df):
    """保存地深肥沃指数数据到 .txt 和 .csv"""
    txt_path = os.path.join(DATA_DIR, '地深肥沃指数.txt')
    csv_path = os.path.join(DATA_DIR, '地深肥沃指数.csv')

    # 保存为 tab 分隔的 txt
    df.to_csv(txt_path, sep='\t', index=False, encoding='utf-8-sig')

    # 保存为 csv
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"样本数据已生成:")
    print(f"  TXT: {txt_path}")
    print(f"  CSV: {csv_path}")
    return txt_path, csv_path


def load_fertility_depth_data(txt_path, csv_path):
    """加载并验证两种格式数据的一致性"""
    df_txt = pd.read_csv(txt_path, sep='\t', encoding='utf-8-sig')
    df_csv = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 兼容不同的列名: x→深度(m), y→肥沃指数
    rename_map = {}
    if 'x' in df_txt.columns and '深度(m)' not in df_txt.columns:
        rename_map['x'] = '深度(m)'
    if 'y' in df_txt.columns and '肥沃指数' not in df_txt.columns:
        rename_map['y'] = '肥沃指数'
    if rename_map:
        df_txt = df_txt.rename(columns=rename_map)
        df_csv = df_csv.rename(columns=rename_map)

    # 验证一致性
    assert df_txt.shape == df_csv.shape, "行数/列数不一致!"
    assert np.allclose(df_txt.values, df_csv.values, rtol=1e-5), "数值不一致!"
    print(f"数据一致性验证通过! ({df_txt.shape[0]} 行 x {df_txt.shape[1]} 列)")

    return df_txt


# ============================================================
# 第一部分: 地深肥沃指数预测 (40%)
# ============================================================
def part1_fertility_depth():
    """第一部分: 连续值预测 — 地深肥沃指数"""
    print("\n" + "=" * 70)
    print("第一部分: 连续值预测 — 地深肥沃指数 (线性回归 vs SVR)")
    print("=" * 70)

    # ---- 1.1 加载或生成数据 ----
    print("\n>>> 1.1 数据准备")
    txt_path = os.path.join(DATA_DIR, '地深肥沃指数.txt')
    csv_path = os.path.join(DATA_DIR, '地深肥沃指数.csv')
    if os.path.exists(txt_path) and os.path.exists(csv_path):
        print(f"从 {DATA_DIR} 加载已有数据")
        df_loaded = load_fertility_depth_data(txt_path, csv_path)
    else:
        df = generate_fertility_depth_data(n_samples=50)
        txt_path, csv_path = save_fertility_depth_data(df)
        df_loaded = load_fertility_depth_data(txt_path, csv_path)
    print(f"\n数据前5行:\n{df_loaded.head()}")
    print(f"\n数据统计:\n{df_loaded.describe()}")

    X = df_loaded[['深度(m)']].values
    y = df_loaded['肥沃指数'].values

    # 7:3 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # ---- 1.2 方法1: 线性回归 ----
    print("\n" + "-" * 50)
    print(">>> 1.2 方法1: 线性回归 (Linear Regression)")
    print("-" * 50)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 系数
    a = lr.coef_[0]
    b = lr.intercept_
    print(f"拟合公式: y = {a:.4f} * x + {b:.4f}")
    print(f"斜率 a = {a:.4f}, 截距 b = {b:.4f}")

    # 预测
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_test)

    # 评估指标
    mse_lr = mean_squared_error(y_test, y_pred_test_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_test_lr)
    r2_lr = r2_score(y_test, y_pred_test_lr)
    mae_train_lr = mean_absolute_error(y_train, y_pred_train_lr)
    r2_train_lr = r2_score(y_train, y_pred_train_lr)

    print(f"\n--- 线性回归评估 ---")
    print(f"训练集: MAE={mae_train_lr:.4f}, R²={r2_train_lr:.4f}")
    print(f"测试集: MSE={mse_lr:.4f}, MAE={mae_lr:.4f}, R²={r2_lr:.4f}")

    # ---- 1.3 方法2: SVR with GridSearchCV ----
    print("\n" + "-" * 50)
    print(">>> 1.3 方法2: SVR (GridSearchCV)")
    print("-" * 50)

    # StandardScaler for SVR (both X and y)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    svr_param_grid = {
        'kernel': ['rbf', 'poly', 'linear'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.05, 0.1],
        'gamma': ['scale', 'auto', 0.1, 0.5],
    }

    svr = SVR()
    svr_gs = GridSearchCV(
        svr, svr_param_grid, cv=5, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=0
    )
    svr_gs.fit(X_train_scaled, y_train_scaled)

    print(f"最佳参数: {svr_gs.best_params_}")
    print(f"最佳CV分数 (neg_MAE): {svr_gs.best_score_:.4f}")

    # 预测并反标准化
    y_pred_train_scaled = svr_gs.predict(X_train_scaled)
    y_pred_test_scaled = svr_gs.predict(X_test_scaled)
    y_pred_train_svr = scaler_y.inverse_transform(
        y_pred_train_scaled.reshape(-1, 1)
    ).ravel()
    y_pred_test_svr = scaler_y.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).ravel()

    mae_svr = mean_absolute_error(y_test, y_pred_test_svr)
    r2_svr = r2_score(y_test, y_pred_test_svr)
    mae_train_svr = mean_absolute_error(y_train, y_pred_train_svr)
    r2_train_svr = r2_score(y_train, y_pred_train_svr)

    print(f"\n--- SVR评估 ---")
    print(f"训练集: MAE={mae_train_svr:.4f}, R²={r2_train_svr:.4f}")
    print(f"测试集: MAE={mae_svr:.4f}, R²={r2_svr:.4f}")

    # ---- 1.4 性能比较 ----
    print("\n" + "-" * 50)
    print(">>> 1.4 性能比较")
    print("-" * 50)

    comparison = pd.DataFrame({
        '方法': ['线性回归 (Linear Regression)', 'SVR (GridSearchCV)'],
        '训练MAE': [mae_train_lr, mae_train_svr],
        '测试MAE': [mae_lr, mae_svr],
        '训练R²': [r2_train_lr, r2_train_svr],
        '测试R²': [r2_lr, r2_svr],
    })
    print(comparison.to_string(index=False))

    if mae_svr < mae_lr:
        winner_mae = 'SVR'
    else:
        winner_mae = '线性回归'
    if r2_svr > r2_lr:
        winner_r2 = 'SVR'
    else:
        winner_r2 = '线性回归'

    print(f"\n>>> 结论:")
    print(f"  MAE 更低的方法: {winner_mae} ({min(mae_lr, mae_svr):.4f})")
    print(f"  R² 更高的方法: {winner_r2} ({max(r2_lr, r2_svr):.4f})")
    print(f"\n  原因分析: 肥沃指数与深度之间为非线性关系")
    print(f"  (指数衰减 + 正弦波动), 线性回归只能拟合直线,")
    print(f"  而 SVR 通过 RBF核函数能够捕捉非线性模式,")
    print(f"  因此 SVR 在该问题上表现更优。")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始数据散点与拟合曲线
    X_plot = np.linspace(0, 210, 500).reshape(-1, 1)
    X_plot_scaled = scaler_X.transform(X_plot)

    y_lr_plot = lr.predict(X_plot)
    y_svr_plot_scaled = svr_gs.predict(X_plot_scaled)
    y_svr_plot = scaler_y.inverse_transform(y_svr_plot_scaled.reshape(-1, 1)).ravel()

    axes[0].scatter(X, y, s=60, c='steelblue', edgecolors='k', linewidth=0.5,
                    alpha=0.7, label='原始数据', zorder=5)
    axes[0].plot(X_plot, y_lr_plot, 'r-', linewidth=2, label='线性回归', zorder=4)
    axes[0].plot(X_plot, y_svr_plot, 'g-', linewidth=2, label='SVR (RBF)', zorder=4)
    axes[0].set_xlabel('深度 (m)', fontsize=12)
    axes[0].set_ylabel('肥沃指数', fontsize=12)
    axes[0].set_title('地深肥沃指数: 线性回归 vs SVR', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SVR 预测值 vs 真实值
    axes[1].scatter(y_test, y_pred_test_svr, s=80, c='forestgreen',
                    edgecolors='k', linewidth=0.5, alpha=0.7, label='SVR')
    axes[1].scatter(y_test, y_pred_test_lr, s=80, c='tomato',
                    edgecolors='k', linewidth=0.5, alpha=0.7, label='线性回归')
    lims = [min(y_test.min(), y_pred_test_lr.min(), y_pred_test_svr.min()),
            max(y_test.max(), y_pred_test_lr.max(), y_pred_test_svr.max())]
    axes[1].plot(lims, lims, 'k--', linewidth=1.5, label='理想线')
    axes[1].set_xlabel('真实肥沃指数', fontsize=12)
    axes[1].set_ylabel('预测肥沃指数', fontsize=12)
    axes[1].set_title('预测值 vs 真实值 (测试集)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODULE_DIR, 'part1_fertility_depth.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("\n第一部分 完成!")
    return comparison


# ============================================================
# 第二部分: 乳腺癌分类 (40%)
# ============================================================
def part2_breast_cancer():
    """第二部分: 分类 — 乳腺癌诊断 (GaussianNB vs RandomForest)"""
    print("\n" + "=" * 70)
    print("第二部分: 分类 — 乳腺癌诊断 (GaussianNB vs RandomForest)")
    print("=" * 70)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    print(f"\n数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"类别: {target_names[0]} ({sum(y==0)}), "
          f"{target_names[1]} ({sum(y==1)})")

    # 标准化 (GaussianNB不需要, 但为了一致性且RF不受影响, 可以统一使用)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80/20 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    def evaluate_classifier(model, model_name, X_train, X_test, y_train, y_test,
                            feature_names=None, show_feature_importance=False,
                            top_n=10):
        """通用分类器评估函数"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # 获取概率 (如果支持)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        # 指标
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
        else:
            roc_auc = None
            fpr, tpr = None, None

        results = {
            'model_name': model_name,
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'fpr': fpr,
            'tpr': tpr,
            'cm': confusion_matrix(y_test, y_pred),
        }

        # 特征重要性
        if show_feature_importance and feature_names is not None:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                results['feature_importance'] = {
                    'names': [feature_names[i] for i in indices],
                    'values': importances[indices],
                }

        return results

    # ---- 方法1: GaussianNB ----
    print("\n" + "-" * 50)
    print(">>> 方法1: Gaussian Naive Bayes")
    print("-" * 50)

    gnb = GaussianNB()
    result_gnb = evaluate_classifier(
        gnb, 'GaussianNB', X_train, X_test, y_train, y_test
    )

    print(f"\n训练准确率:   {result_gnb['accuracy_train']:.4f}")
    print(f"测试准确率:   {result_gnb['accuracy_test']:.4f}")
    print(f"精确率 (Precision): {result_gnb['precision']:.4f}")
    print(f"召回率 (Recall):    {result_gnb['recall']:.4f}")
    print(f"F1分数:             {result_gnb['f1']:.4f}")
    print(f"ROC-AUC:            {result_gnb['roc_auc']:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, result_gnb['y_pred'],
                                target_names=target_names))

    # ---- 方法2: RandomForest ----
    print("\n" + "-" * 50)
    print(">>> 方法2: Random Forest (n=100, max_depth=10, max_features='sqrt')")
    print("-" * 50)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    result_rf = evaluate_classifier(
        rf, 'RandomForest', X_train, X_test, y_train, y_test,
        feature_names=feature_names, show_feature_importance=True, top_n=10
    )

    print(f"\n训练准确率:   {result_rf['accuracy_train']:.4f}")
    print(f"测试准确率:   {result_rf['accuracy_test']:.4f}")
    print(f"精确率 (Precision): {result_rf['precision']:.4f}")
    print(f"召回率 (Recall):    {result_rf['recall']:.4f}")
    print(f"F1分数:             {result_rf['f1']:.4f}")
    print(f"ROC-AUC:            {result_rf['roc_auc']:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, result_rf['y_pred'],
                                target_names=target_names))

    # 特征重要性 Top 10
    print("\n特征重要性 Top 10:")
    fi = result_rf['feature_importance']
    for i, (name, val) in enumerate(zip(fi['names'], fi['values'])):
        print(f"  {i+1:2d}. {name:<30s} {val:.4f}")

    # ---- 5折CV (RandomForest) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"\nRandomForest 5折CV准确率: {cv_scores_rf}")
    print(f"随机森林 5折CV平均准确率: {cv_scores_rf.mean():.4f} "
          f"(+/- {cv_scores_rf.std() * 2:.4f})")

    # ---- 比较表 ----
    print("\n" + "-" * 60)
    print(">>> 方法比较表")
    print("-" * 60)
    comparison = pd.DataFrame({
        '方法': ['GaussianNB', 'RandomForest'],
        '训练准确率': [
            result_gnb['accuracy_train'], result_rf['accuracy_train']
        ],
        '测试准确率': [
            result_gnb['accuracy_test'], result_rf['accuracy_test']
        ],
        '精确率': [result_gnb['precision'], result_rf['precision']],
        '召回率': [result_gnb['recall'], result_rf['recall']],
        'F1分数': [result_gnb['f1'], result_rf['f1']],
        'ROC-AUC': [result_gnb['roc_auc'], result_rf['roc_auc']],
    })
    print(comparison.to_string(index=False))

    # 结论
    if result_rf['f1'] > result_gnb['f1']:
        best = 'RandomForest'
    else:
        best = 'GaussianNB'
    print(f"\n>>> 结论: {best} 在乳腺癌诊断中表现更好。")
    print("  RandomForest 通过集成多棵决策树, 能捕获特征间的交互效应,")
    print("  且对数据分布无强假设, 通常优于朴素贝叶斯。")
    print("  但 GaussianNB 训练速度快、可解释性好, 适合基线模型。")
    print("  对于医疗诊断, 综合考虑精确率和召回率的F1分数, RandomForest更优。")

    # ---- 可视化 ----
    fig = plt.figure(figsize=(16, 10))

    # (1) GNB 混淆矩阵
    ax1 = fig.add_subplot(2, 3, 1)
    ConfusionMatrixDisplay(confusion_matrix=result_gnb['cm'],
                           display_labels=target_names).plot(
        ax=ax1, cmap='Blues', colorbar=False
    )
    ax1.set_title('GaussianNB 混淆矩阵', fontsize=12)

    # (2) RF 混淆矩阵
    ax2 = fig.add_subplot(2, 3, 2)
    ConfusionMatrixDisplay(confusion_matrix=result_rf['cm'],
                           display_labels=target_names).plot(
        ax=ax2, cmap='Greens', colorbar=False
    )
    ax2.set_title('RandomForest 混淆矩阵', fontsize=12)

    # (3) ROC曲线
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(result_gnb['fpr'], result_gnb['tpr'],
             label=f"GaussianNB (AUC={result_gnb['roc_auc']:.3f})",
             linewidth=2, color='steelblue')
    ax3.plot(result_rf['fpr'], result_rf['tpr'],
             label=f"RandomForest (AUC={result_rf['roc_auc']:.3f})",
             linewidth=2, color='darkorange')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器')
    ax3.set_xlabel('假阳性率 (FPR)', fontsize=10)
    ax3.set_ylabel('真阳性率 (TPR)', fontsize=10)
    ax3.set_title('ROC曲线比较', fontsize=12)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # (4) RF 特征重要性
    ax4 = fig.add_subplot(2, 3, 4)
    fi_names = fi['names'][::-1]
    fi_vals = fi['values'][::-1]
    ax4.barh(range(len(fi_names)), fi_vals, color='forestgreen', edgecolor='k')
    ax4.set_yticks(range(len(fi_names)))
    ax4.set_yticklabels(fi_names, fontsize=8)
    ax4.set_xlabel('重要性', fontsize=10)
    ax4.set_title('RandomForest 特征重要性 Top 10', fontsize=12)

    # (5) 指标比较柱状图
    ax5 = fig.add_subplot(2, 3, 5)
    metrics = ['精确率', '召回率', 'F1分数', 'ROC-AUC']
    gnb_vals = [result_gnb['precision'], result_gnb['recall'],
                result_gnb['f1'], result_gnb['roc_auc']]
    rf_vals = [result_rf['precision'], result_rf['recall'],
               result_rf['f1'], result_rf['roc_auc']]
    x = np.arange(len(metrics))
    w = 0.3
    ax5.bar(x - w/2, gnb_vals, w, label='GaussianNB', color='steelblue',
            edgecolor='k')
    ax5.bar(x + w/2, rf_vals, w, label='RandomForest', color='darkorange',
            edgecolor='k')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=10)
    ax5.set_ylabel('分数', fontsize=10)
    ax5.set_title('两方法指标比较', fontsize=12)
    ax5.legend()
    ax5.set_ylim(0.85, 1.02)
    ax5.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODULE_DIR, 'part2_breast_cancer.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("\n第二部分 完成!")
    return result_gnb, result_rf


# ============================================================
# 第三部分: 不平衡数据与降维 (20%)
# ============================================================
def part3_imbalanced_and_dim_reduction():
    """第三部分: 不平衡数据处理与降维可视化"""
    print("\n" + "=" * 70)
    print("第三部分: 不平衡数据与降维 (多分类器 + PCA + t-SNE)")
    print("=" * 70)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- 3.1 类别不平衡分析 ----
    print("\n>>> 3.1 类别不平衡分析")
    unique, counts = np.unique(y, return_counts=True)
    print(f"良性 (Benign, y=0): {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
    print(f"恶性 (Malignant, y=1): {counts[1]} ({counts[1]/len(y)*100:.1f}%)")
    imbalance_ratio = counts[0] / counts[1]
    print(f"不平衡比例 (良性/恶性): {imbalance_ratio:.2f}:1")
    print(f"该数据集存在轻微类别不平衡")

    # ---- 3.2 五分类器对比 (StratifiedKFold 5-fold) ----
    print("\n>>> 3.2 五分类器对比 (StratifiedKFold 5-fold)")

    classifiers = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale',
                         random_state=42, probability=True),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    cv_results = {}
    for name, clf in classifiers.items():
        print(f"  评估: {name} ...")
        scores = cross_validate(
            clf, X_scaled, y, cv=cv, scoring=scoring_metrics, n_jobs=-1
        )
        cv_results[name] = {
            'accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
            'precision': (scores['test_precision'].mean(), scores['test_precision'].std()),
            'recall': (scores['test_recall'].mean(), scores['test_recall'].std()),
            'f1': (scores['test_f1'].mean(), scores['test_f1'].std()),
            'roc_auc': (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()),
        }

    # 打印详细结果
    print(f"\n{'模型':<25} {'准确率':>14} {'精确率':>14} {'召回率':>14} "
          f"{'F1分数':>14} {'ROC-AUC':>14}")
    print("-" * 100)
    for name, metrics in cv_results.items():
        print(f"{name:<25} {metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.3f}  "
              f"{metrics['precision'][0]:.4f}±{metrics['precision'][1]:.3f}  "
              f"{metrics['recall'][0]:.4f}±{metrics['recall'][1]:.3f}  "
              f"{metrics['f1'][0]:.4f}±{metrics['f1'][1]:.3f}  "
              f"{metrics['roc_auc'][0]:.4f}±{metrics['roc_auc'][1]:.3f}")

    # 按F1分数排名
    ranking = sorted(cv_results.items(), key=lambda x: x[1]['f1'][0], reverse=True)
    print(f"\n>>> 模型排名 (按F1分数):")
    for rank, (name, metrics) in enumerate(ranking, 1):
        print(f"  {rank}. {name:<25s} F1={metrics['f1'][0]:.4f}±{metrics['f1'][1]:.3f}")

    # ---- 3.3 PCA ----
    print("\n>>> 3.3 PCA 降维分析")

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 解释方差
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_var_ratio)

    print(f"前2个主成分解释方差: PC1={explained_var_ratio[0]*100:.2f}%, "
          f"PC2={explained_var_ratio[1]*100:.2f}%")
    print(f"前2个主成分累计解释方差: {(explained_var_ratio[0]+explained_var_ratio[1])*100:.2f}%")
    print(f"前5个主成分累计解释方差: {cumsum[4]*100:.2f}%")
    print(f"前10个主成分累计解释方差: {cumsum[9]*100:.2f}%")

    # 找到解释95%方差所需的主成分数
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"解释95%方差所需主成分数: {n_components_95}")

    # ---- 3.4 t-SNE ----
    print("\n>>> 3.4 t-SNE 降维可视化")
    print("  计算t-SNE (可能需要几十秒)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    print("  t-SNE完成!")

    # ---- 3.5 综合可视化 ----
    print("\n>>> 3.5 生成综合可视化图表")

    # 获取所有分类器的ROC和PR曲线数据 (用于后续绘图)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    roc_data = {}
    pr_data = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            # SVM without probability would need decision_function
            y_prob = clf.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, auc_val)

        pre, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (pre, rec)

    # === 创建综合可视化大图 ===
    fig = plt.figure(figsize=(20, 18))

    # (1) 类别分布饼图
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.pie([counts[0], counts[1]], labels=[f'{target_names[0]}\n({counts[0]})',
                                            f'{target_names[1]}\n({counts[1]})'],
            colors=['#5B9BD5', '#ED7D31'], autopct='%1.1f%%',
            explode=(0, 0.05), startangle=90)
    ax1.set_title('类别分布', fontsize=13, fontweight='bold')

    # (2) 性能热力图
    ax2 = fig.add_subplot(3, 4, 2)
    model_names = list(cv_results.keys())
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC']
    heatmap_data = np.zeros((len(model_names), len(metric_names)))
    for i, name in enumerate(model_names):
        for j, m in enumerate(['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
            heatmap_data[i, j] = cv_results[name][m][0]
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=metric_names, yticklabels=model_names,
                ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('性能热力图 (5折CV均值)', fontsize=13, fontweight='bold')

    # (3) P/R/F1 柱状图
    ax3 = fig.add_subplot(3, 4, 3)
    x = np.arange(len(model_names))
    w = 0.25
    prec_vals = [cv_results[n]['precision'][0] for n in model_names]
    rec_vals = [cv_results[n]['recall'][0] for n in model_names]
    f1_vals = [cv_results[n]['f1'][0] for n in model_names]
    ax3.bar(x - w, prec_vals, w, label='精确率', color='#4472C4', edgecolor='k')
    ax3.bar(x, rec_vals, w, label='召回率', color='#ED7D31', edgecolor='k')
    ax3.bar(x + w, f1_vals, w, label='F1分数', color='#70AD47', edgecolor='k')
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.split('(')[0].strip() for n in model_names],
                        fontsize=7, rotation=20)
    ax3.set_ylabel('分数', fontsize=9)
    ax3.set_title('P / R / F1 比较', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.set_ylim(0.80, 1.05)
    ax3.grid(axis='y', alpha=0.3)

    # (4) PCA 累积方差图
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
    ax4.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='95% 阈值')
    ax4.axvline(x=n_components_95, color='gray', linestyle='--', linewidth=1,
                label=f'n={n_components_95}')
    ax4.set_xlabel('主成分数', fontsize=10)
    ax4.set_ylabel('累积解释方差', fontsize=10)
    ax4.set_title('PCA 累积方差', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # (5) PCA 2D 散点图
    ax5 = fig.add_subplot(3, 4, 5)
    for cls, label, color in zip([0, 1], target_names, ['#5B9BD5', '#ED7D31']):
        mask = y == cls
        ax5.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label,
                    alpha=0.6, edgecolors='k', linewidth=0.3, s=40)
    ax5.set_xlabel(f'PC1 ({explained_var_ratio[0]*100:.1f}%)', fontsize=10)
    ax5.set_ylabel(f'PC2 ({explained_var_ratio[1]*100:.1f}%)', fontsize=10)
    ax5.set_title('PCA 二维可视化', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # (6) t-SNE 2D 散点图
    ax6 = fig.add_subplot(3, 4, 6)
    for cls, label, color in zip([0, 1], target_names, ['#5B9BD5', '#ED7D31']):
        mask = y == cls
        ax6.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=label,
                    alpha=0.6, edgecolors='k', linewidth=0.3, s=40)
    ax6.set_xlabel('t-SNE 维度1', fontsize=10)
    ax6.set_ylabel('t-SNE 维度2', fontsize=10)
    ax6.set_title('t-SNE 二维可视化', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # (7) ROC 曲线 (所有模型)
    ax7 = fig.add_subplot(3, 4, 7)
    colors_roc = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    for (name, (fpr, tpr, auc_val)), c in zip(roc_data.items(), colors_roc):
        ax7.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_val:.3f})', color=c)
    ax7.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机')
    ax7.set_xlabel('假阳性率 (FPR)', fontsize=10)
    ax7.set_ylabel('真阳性率 (TPR)', fontsize=10)
    ax7.set_title('ROC 曲线比较', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=7, loc='lower right')
    ax7.grid(True, alpha=0.3)

    # (8) PR 曲线 (所有模型)
    ax8 = fig.add_subplot(3, 4, 8)
    for (name, (pre, rec)), c in zip(pr_data.items(), colors_roc):
        ax8.plot(rec, pre, linewidth=2, label=name, color=c)
    ax8.set_xlabel('召回率 (Recall)', fontsize=10)
    ax8.set_ylabel('精确率 (Precision)', fontsize=10)
    ax8.set_title('P-R 曲线比较', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=7, loc='lower left')
    ax8.grid(True, alpha=0.3)

    # (9) 各主成分解释方差 (碎石图)
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.bar(range(1, 16), explained_var_ratio[:15], color='steelblue', edgecolor='k')
    ax9.plot(range(1, 16), explained_var_ratio[:15], 'ro-', markersize=4)
    ax9.set_xlabel('主成分', fontsize=10)
    ax9.set_ylabel('解释方差比例', fontsize=10)
    ax9.set_title('PCA 碎石图 (前15个)', fontsize=13, fontweight='bold')
    ax9.grid(axis='y', alpha=0.3)

    # (10) 准确率对比柱状图 (带误差棒)
    ax10 = fig.add_subplot(3, 4, 10)
    acc_means = [cv_results[n]['accuracy'][0] for n in model_names]
    acc_stds = [cv_results[n]['accuracy'][1] for n in model_names]
    short_names = [n.split('(')[0].strip() for n in model_names]
    bars = ax10.bar(short_names, acc_means, color=plt.cm.Set2(np.linspace(0, 1, 5)),
                    edgecolor='k')
    ax10.errorbar(short_names, acc_means, yerr=acc_stds, fmt='none',
                  ecolor='black', capsize=5)
    ax10.set_ylabel('准确率', fontsize=10)
    ax10.set_title('5折CV准确率 (±1std)', fontsize=13, fontweight='bold')
    ax10.set_ylim(0.88, 1.0)
    ax10.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, acc_means):
        ax10.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                  f'{v:.3f}', ha='center', fontsize=8)

    # (11) PCA 二维分布密度估计
    ax11 = fig.add_subplot(3, 4, 11)
    for cls, label, cmap in zip([0, 1], target_names, ['Blues', 'Oranges']):
        mask = y == cls
        sns.kdeplot(x=X_pca[mask, 0], y=X_pca[mask, 1], ax=ax11,
                    cmap=cmap, fill=True, alpha=0.4, label=label, levels=5)
    ax11.set_xlabel(f'PC1 ({explained_var_ratio[0]*100:.1f}%)', fontsize=10)
    ax11.set_ylabel(f'PC2 ({explained_var_ratio[1]*100:.1f}%)', fontsize=10)
    ax11.set_title('PCA 类别密度分布', fontsize=13, fontweight='bold')
    ax11.legend(fontsize=9)

    # (12) 模型排名 (水平柱状图)
    ax12 = fig.add_subplot(3, 4, 12)
    rank_names = [r[0] for r in ranking]
    rank_f1s = [r[1]['f1'][0] for r in ranking]
    rank_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(rank_names)))[::-1]
    ax12.barh(rank_names, rank_f1s, color=rank_colors, edgecolor='k')
    ax12.set_xlabel('F1分数', fontsize=10)
    ax12.set_title('模型F1排名', fontsize=13, fontweight='bold')
    for i, v in enumerate(rank_f1s):
        ax12.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

    plt.suptitle('乳腺癌数据集综合分析', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(MODULE_DIR, 'part3_comprehensive.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("\n第三部分 完成!")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("期末综合项目 (Final Project)")
    print("  第一部分: 连续值预测 — 地深肥沃指数")
    print("  第二部分: 分类 — 乳腺癌诊断")
    print("  第三部分: 不平衡数据与降维")
    print("=" * 70)

    # 第一部分
    part1_fertility_depth()

    # 第二部分
    part2_breast_cancer()

    # 第三部分
    part3_imbalanced_and_dim_reduction()

    print("\n" + "=" * 70)
    print("期末综合项目 — 所有部分完成!")
    print("=" * 70)
