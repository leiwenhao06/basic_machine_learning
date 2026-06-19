"""
决策树 (Decision Tree)
=====================
Task 1: 信用卡交易分类 — 15条样本, entropy准则, 5折交叉验证, 可视化决策树与混淆矩阵
Task 2: 红酒分类 — sklearn load_wine, train/test 7:3, max_depth=4, 准确率/分类报告/特征重要性
Task 3: 天气打球决策 — 14条训练+4条预测, LabelEncode, export_text文本规则, 可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine

# ======================== 中文 / 字体设置 ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ================================================================
#                          Task 1: 信用卡交易分类
# ================================================================
def task1_credit_card():
    """15 条样本 [Refund, MaritalStatus, Income, Cheat] 决策树分类"""
    print("=" * 60)
    print("Task 1: 信用卡交易分类")
    print("=" * 60)

    # ---- 构造数据 ----
    data = {
        'Refund':       ['Yes','No','No','Yes','No','No','Yes','Yes','No','No',
                          'No','Yes','No','Yes','No'],
        'MaritalStatus':['Single','Married','Single','Married','Divorced','Married',
                         'Divorced','Single','Married','Single',
                         'Single','Married','Single','Divorced','Married'],
        'Income':       [125, 100, 70, 120, 95, 60, 220, 85, 75, 90,
                          55, 110, 80, 65, 105],
        'Cheat':        ['No','No','No','No','Yes','No','No','No','No','Yes',
                          'Yes','Yes','No','Yes','No']
    }
    df = pd.DataFrame(data)
    print("原始数据 (前5行):")
    print(df.head(10).to_string(index=False))

    # ---- 编码 ----
    le_refund = LabelEncoder()
    le_marital = LabelEncoder()
    le_cheat = LabelEncoder()

    df['Refund_enc']        = le_refund.fit_transform(df['Refund'])
    df['MaritalStatus_enc'] = le_marital.fit_transform(df['MaritalStatus'])
    df['Cheat_enc']         = le_cheat.fit_transform(df['Cheat'])   # No=0, Yes=1

    X = df[['Refund_enc', 'MaritalStatus_enc', 'Income']].values
    y = df['Cheat_enc'].values

    # ---- 决策树 (entropy) ----
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X, y)

    # ---- 5 折交叉验证 ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    print(f"\n5折交叉验证准确率: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---- 训练集预测 + 混淆矩阵 ----
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n训练集准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(y, y_pred, target_names=['No', 'Yes']))

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 决策树
    feature_names = ['Refund', 'MaritalStatus', 'Income']
    class_names = ['No', 'Yes']
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=axes[0])
    axes[0].set_title('信用卡交易 — 决策树 (entropy)', fontsize=13)

    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(ax=axes[1], cmap='Blues')
    axes[1].set_title('训练集混淆矩阵', fontsize=13)

    plt.tight_layout()
    plt.savefig('05_decision_tree/task1_credit_card.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 05_decision_tree/task1_credit_card.png\n")


# ================================================================
#                          Task 2: 红酒分类
# ================================================================
def task2_wine():
    """sklearn load_wine, train/test 7:3, max_depth=4"""
    print("=" * 60)
    print("Task 2: 红酒分类 (load_wine)")
    print("=" * 60)

    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"类别: {target_names}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 训练
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率: {acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ---- 特征重要性 ----
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n特征重要性 (降序):")
    for i in indices:
        print(f"  {feature_names[i]:>25s}: {importances[i]:.4f}")

    # 可视化特征重要性
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('重要性')
    ax.set_title('红酒分类 — 特征重要性 (max_depth=4)', fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('05_decision_tree/task2_wine_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 05_decision_tree/task2_wine_importance.png\n")


# ================================================================
#                       Task 3: 天气打球决策
# ================================================================
def task3_weather():
    """14 条训练 + 4 条测试, LabelEncode, export_text"""
    print("=" * 60)
    print("Task 3: 天气打球决策")
    print("=" * 60)

    # ---- 训练数据 (经典 Play Tennis) ----
    train_data = [
        ['Sunny',    'Hot',      'High',     'Weak',     'No'],
        ['Sunny',    'Hot',      'High',     'Strong',   'No'],
        ['Overcast', 'Hot',      'High',     'Weak',     'Yes'],
        ['Rain',     'Mild',     'High',     'Weak',     'Yes'],
        ['Rain',     'Cool',     'Normal',   'Weak',     'Yes'],
        ['Rain',     'Cool',     'Normal',   'Strong',   'No'],
        ['Overcast', 'Cool',     'Normal',   'Strong',   'Yes'],
        ['Sunny',    'Mild',     'High',     'Weak',     'No'],
        ['Sunny',    'Cool',     'Normal',   'Weak',     'Yes'],
        ['Rain',     'Mild',     'Normal',   'Weak',     'Yes'],
        ['Sunny',    'Mild',     'Normal',   'Strong',   'Yes'],
        ['Overcast', 'Mild',     'High',     'Strong',   'Yes'],
        ['Overcast', 'Hot',      'Normal',   'Weak',     'Yes'],
        ['Rain',     'Mild',     'High',     'Strong',   'No'],
    ]
    columns = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']
    df_train = pd.DataFrame(train_data, columns=columns)

    # ---- 测试数据 ----
    test_data = [
        ['Sunny',    'Mild',     'Normal',   'Weak'],
        ['Rain',     'Cool',     'High',     'Strong'],
        ['Overcast', 'Hot',      'Normal',   'Strong'],
        ['Sunny',    'Hot',      'High',     'Weak'],
    ]
    df_test = pd.DataFrame(test_data, columns=columns[:-1])

    # ---- 逐列 LabelEncode ----
    encoders = {}
    df_enc = pd.DataFrame()
    for col in columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_train[col])
        encoders[col] = le

    # 测试集编码 (使用相同 encoder)
    df_test_enc = pd.DataFrame()
    for col in columns[:-1]:
        df_test_enc[col] = encoders[col].transform(df_test[col])

    X_train = df_enc[columns[:-1]].values
    y_train = df_enc['Play'].values
    X_test  = df_test_enc.values

    # ---- 决策树 ----
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)

    # ---- 预测 ----
    y_pred = clf.predict(X_test)
    pred_labels = encoders['Play'].inverse_transform(y_pred)
    acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"训练集准确率: {acc:.4f}")

    print("\n测试样本预测结果:")
    for i, row in enumerate(test_data):
        print(f"  样本{i+1}: {row}  ->  预测: {pred_labels[i]}")

    # ---- 导出文本规则 ----
    print("\n决策树文本规则 (export_text):")
    print("-" * 40)
    text_rules = export_text(clf, feature_names=columns[:-1])
    print(text_rules)

    # ---- 可视化决策树 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=columns[:-1],
              class_names=encoders['Play'].classes_.tolist(),
              filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title('天气打球 — 决策树 (entropy)', fontsize=14)
    plt.tight_layout()
    plt.savefig('05_decision_tree/task3_weather_tree.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 05_decision_tree/task3_weather_tree.png\n")


# ================================================================
if __name__ == "__main__":
    import os
    os.makedirs('05_decision_tree', exist_ok=True)

    task1_credit_card()
    task2_wine()
    task3_weather()
