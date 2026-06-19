"""
朴素贝叶斯分类器 (Naive Bayes)
=============================
Task 1: 乳腺癌分类 — GaussianNB, 训练/测试准确率, 概率分布
Task 2: 文本情感分类 — CountVectorizer + MultinomialNB, 12句话预测
Task 3: 学生成绩分类 — 手工实现高斯朴素贝叶斯 (不用 sklearn NB)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# ======================== 中文 / 字体设置 ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ================================================================
#                    Task 1: 乳腺癌分类 (GaussianNB)
# ================================================================
def task1_breast_cancer():
    """GaussianNB 在 load_breast_cancer 上的应用"""
    print("=" * 60)
    print("Task 1: 乳腺癌分类 (GaussianNB)")
    print("=" * 60)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"类别: {data.target_names}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练高斯朴素贝叶斯
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # 评估
    y_train_pred = gnb.predict(X_train)
    y_test_pred  = gnb.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))

    # ---- 概率分布 ----
    proba = gnb.predict_proba(X_test)
    print("\n测试集前 10 个样本的预测概率分布 (第1列=恶性, 第2列=良性):")
    for i in range(10):
        true_label = data.target_names[y_test[i]]
        pred_label = data.target_names[y_test_pred[i]]
        print(f"  样本{i+1}: 真实={true_label}, 预测={pred_label}, "
              f"P(恶性)={proba[i,0]:.6f}, P(良性)={proba[i,1]:.6f}")

    # ---- 可视化概率分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 恶性样本的概率
    malignant_idx = np.where(y_test == 0)[0]
    if len(malignant_idx) > 0:
        axes[0].hist(proba[malignant_idx, 0], bins=15, alpha=0.7, color='red', edgecolor='black')
        axes[0].set_title('恶性样本 — 被预测为恶性的概率', fontsize=12)
        axes[0].set_xlabel('P(恶性)')
        axes[0].set_ylabel('频数')

    # 良性样本的概率
    benign_idx = np.where(y_test == 1)[0]
    axes[1].hist(proba[benign_idx, 1], bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('良性样本 — 被预测为良性的概率', fontsize=12)
    axes[1].set_xlabel('P(良性)')
    axes[1].set_ylabel('频数')

    plt.suptitle('测试集概率分布 (GaussianNB)', fontsize=14)
    plt.tight_layout()
    plt.savefig('06_naive_bayes/task1_probability.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[图已保存] 06_naive_bayes/task1_probability.png\n")


# ================================================================
#                 Task 2: 文本情感分类 (MultinomialNB)
# ================================================================
def task2_text_sentiment():
    """CountVectorizer + MultinomialNB 情感分类"""
    print("=" * 60)
    print("Task 2: 文本情感分类 (MultinomialNB)")
    print("=" * 60)

    # 12 句话: 前6条积极, 后6条消极
    sentences = [
        # 积极 (label=1)
        "I love this movie",
        "This is a great product",
        "Wonderful experience and amazing service",
        "I am very happy with the result",
        "Excellent quality and highly recommended",
        "Fantastic performance and beautiful design",
        # 消极 (label=0)
        "I hate this terrible movie",
        "This product is bad and disappointing",
        "Awful experience and poor service",
        "I am very sad and frustrated",
        "Poor quality and not worth the money",
        "Terrible performance and ugly design",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    # 向量化
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    y = np.array(labels)

    print(f"词汇表大小: {len(vectorizer.vocabulary_)}")
    print("词汇表:", list(vectorizer.vocabulary_.keys()))

    # 训练
    mnb = MultinomialNB()
    mnb.fit(X, y)

    # 训练集准确率
    y_pred = mnb.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n训练集准确率: {acc:.4f}")
    print("分类报告:")
    print(classification_report(y, y_pred, target_names=['消极', '积极']))

    # 预测新句子
    test_sentence = ["She is happy to go shopping"]
    X_test = vectorizer.transform(test_sentence)
    pred = mnb.predict(X_test)[0]
    proba = mnb.predict_proba(X_test)[0]
    result = "积极" if pred == 1 else "消极"
    print(f"\n预测句子: \"{test_sentence[0]}\"")
    print(f"预测结果: {result}")
    print(f"概率分布: P(消极)={proba[0]:.4f}, P(积极)={proba[1]:.4f}")
    print()


# ================================================================
#          Task 3: 学生成绩分类 — 手工实现高斯朴素贝叶斯
# ================================================================
def gaussian_pdf(x, mean, std):
    """高斯 (正态) 分布概率密度函数"""
    eps = 1e-9  # 防止除零
    coefficient = 1.0 / (np.sqrt(2.0 * np.pi) * (std + eps))
    exponent = np.exp(-0.5 * ((x - mean) / (std + eps)) ** 2)
    return coefficient * exponent


def task3_manual_gaussian_nb():
    """手工实现高斯朴素贝叶斯, 预测学生成绩=92的等级"""
    print("=" * 60)
    print("Task 3: 学生成绩分类 — 手工高斯朴素贝叶斯")
    print("=" * 60)

    # ---- 10 名学生的成绩与等级 ----
    # 每名学生: [平时成绩, 期末成绩, 等级]
    students = np.array([
        [85, 88, 1],   # 1 = 优秀
        [92, 95, 1],
        [78, 72, 0],   # 0 = 合格
        [90, 93, 1],
        [65, 68, 0],
        [88, 90, 1],
        [70, 74, 0],
        [95, 97, 1],
        [60, 62, 0],
        [82, 80, 0],
    ])

    X = students[:, :2]   # [平时成绩, 期末成绩]
    y = students[:, 2].astype(int)

    class_names = ['合格', '优秀']
    print("学生数据:")
    for i, (x1, x2, label) in enumerate(students):
        print(f"  学生{i+1}: 平时={x1:.0f}, 期末={x2:.0f}, 等级={class_names[int(label)]}")

    # ---- Step 1: 先验概率 P(C_k) ----
    n_total = len(y)
    classes = np.unique(y)
    priors = {}
    for c in classes:
        priors[c] = np.sum(y == c) / n_total
        print(f"\n先验概率 P({class_names[int(c)]}) = {priors[c]:.4f}")

    # ---- Step 2: 每类的均值 & 标准差 ----
    means = {}
    stds = {}
    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        stds[c]  = np.std(X_c, axis=0, ddof=1)  # 样本标准差
        print(f"\n[{class_names[int(c)]}] 均值: 平时={means[c][0]:.2f}, 期末={means[c][1]:.2f}")
        print(f"[{class_names[int(c)]}] 标准差: 平时={stds[c][0]:.2f}, 期末={stds[c][1]:.2f}")

    # ---- Step 3: 预测 score = 92 (平时=92, 期末=? ) ----
    # 题目说 "Predict score=92", 我们假设指平时成绩=92, 期末成绩也取一个合理的值
    # 更合理的解释: score=92 指的是一个综合/单一分数, 这里我们用平时=92, 期末=90 作为测试
    # 为了更贴合"score=92"的表述, 我们同时预测:平时成绩=92 (如果只有一维)
    # 根据题意, 我们用 平时=92 来预测 (单一特征更贴近"score=92")
    # 但数据是二维的, 我们按数据格式统一处理 —— 把 92 同时作为平时和期末的近似
    test_samples = {
        "平时92": np.array([92.0, 90.0]),
        "平时75": np.array([75.0, 72.0]),
    }

    print("\n" + "-" * 40)
    print("预测结果 (手工高斯朴素贝叶斯):")
    print("-" * 40)

    for name, x_test in test_samples.items():
        print(f"\n测试样本 ({name}): 平时={x_test[0]:.0f}, 期末={x_test[1]:.0f}")

        posteriors = {}
        for c in classes:
            # P(C_k) * prod(P(x_i | C_k))
            posterior = priors[c]
            for i in range(len(x_test)):
                likelihood = gaussian_pdf(x_test[i], means[c][i], stds[c][i])
                posterior *= likelihood
            posteriors[c] = posterior
            print(f"  P({class_names[int(c)]}) * 似然 = {posterior:.12f}")

        # 比较后验概率
        best_class = max(posteriors, key=posteriors.get)
        print(f"  预测结果: {class_names[int(best_class)]} "
              f"(概率比: {posteriors[1]/posteriors[0]:.6f} 倍)" if posteriors[0] > 0 else
              f"(后验最大: {class_names[int(best_class)]})")

    # ---- 与 sklearn 对比验证 ----
    print("\n" + "-" * 40)
    print("验证: sklearn GaussianNB 结果对比")
    print("-" * 40)
    gnb = GaussianNB()
    gnb.fit(X, y)
    for name, x_test in test_samples.items():
        pred = gnb.predict([x_test])[0]
        proba = gnb.predict_proba([x_test])[0]
        print(f"  {name}: 预测={class_names[int(pred)]}, "
              f"P(合格)={proba[0]:.4f}, P(优秀)={proba[1]:.4f}")
    print()


# ================================================================
if __name__ == "__main__":
    import os
    os.makedirs('06_naive_bayes', exist_ok=True)

    task1_breast_cancer()
    task2_text_sentiment()
    task3_manual_gaussian_nb()
