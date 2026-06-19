"""
基于朴素贝叶斯的文本分类 — Text Classification with Naive Bayes
================================================================
包含三个子任务：
  Task 1: 侮辱性评论过滤器（手动实现朴素贝叶斯）
  Task 2: 垃圾邮件分类（sklearn MultinomialNB + CountVectorizer）
  Task 3: 中文新闻分类（jieba分词 + TfidfVectorizer + MultinomialNB）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import seaborn as sns

# ---------------------------------------------------------------------------
# Chinese font support
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Try to import jieba for Chinese word segmentation
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告: jieba 未安装，中文分词将使用基于字符的特征。")
    print("安装: pip install jieba")


# ===================================================================
# Task 1: Insulting Comment Filter (Manual Naive Bayes)
# ===================================================================
def task1_insulting_comment_filter():
    """
    使用手动实现的朴素贝叶斯分类器过滤侮辱性评论。
    不依赖 sklearn 的 NB 模块。
    """
    print("=" * 60)
    print("Task 1: 侮辱性评论过滤器 Manual Naive Bayes")
    print("=" * 60)

    # --- Data ---
    dataSet = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]
    labels = np.array([0, 1, 0, 1, 0, 1])  # 0=正常, 1=侮辱

    print("\n训练数据 Training Data:")
    for sent, lbl in zip(dataSet, labels):
        tag = "侮辱 Insulting" if lbl == 1 else "正常 Normal"
        print(f"  [{tag}] {' '.join(sent)}")

    # --- Step 1: Create vocabulary ---
    def createVocabList(dataSet):
        """Create a list of all unique words from the dataset."""
        vocabSet = set()
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return sorted(list(vocabSet))

    # --- Step 2: Convert text to binary word vector ---
    def setOfWords2Vec(vocabList, inputSet):
        """
        Convert a document to a binary vector:
        1 if the word appears in the document, 0 otherwise.
        """
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
        return returnVec

    # --- Step 3: Train Naive Bayes with Laplace smoothing ---
    def trainNB(trainMatrix, trainCategory):
        """
        Compute p0Vect, p1Vect, pAbusive using Laplace smoothing.

        Laplace smoothing: add 1 to all word counts,
                           add 2 to denominator (for binary features).

        Returns log-probabilities to avoid underflow.
        """
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = np.sum(trainCategory) / float(numTrainDocs)

        # Laplace smoothing: initialize counts to 1, denominator to 2
        p0Num = np.ones(numWords)       # add 1 (Laplace)
        p1Num = np.ones(numWords)
        p0Denom = 2.0                   # add 2 (for binary, each word can be 0 or 1)
        p1Denom = 2.0

        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += np.sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += np.sum(trainMatrix[i])

        # Log probabilities
        p1Vect = np.log(p1Num / p1Denom)
        p0Vect = np.log(p0Num / p0Denom)

        return p0Vect, p1Vect, pAbusive

    # --- Step 4: Classify ---
    def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
        """
        Classify using log probabilities:
        p1 = log(pClass1) + sum(vec2Classify * p1Vec)
        p0 = log(1 - pClass1) + sum(vec2Classify * p0Vec)
        """
        p1 = np.log(pClass1) + np.sum(vec2Classify * p1Vec)
        p0 = np.log(1.0 - pClass1) + np.sum(vec2Classify * p0Vec)
        return 1 if p1 > p0 else 0

    # --- Build vocabulary ---
    vocabList = createVocabList(dataSet)
    print(f"\n词汇表 Vocabulary ({len(vocabList)} 词):")
    print(f"  {vocabList}")

    # --- Build training matrix ---
    trainMat = []
    for doc in dataSet:
        trainMat.append(setOfWords2Vec(vocabList, doc))

    # --- Train ---
    p0V, p1V, pAb = trainNB(trainMat, labels)
    print(f"\n训练结果 Training Results:")
    print(f"  pAbusive (侮辱类先验概率): {pAb:.4f}")
    print(f"  p0Vect shape: {p0V.shape}")
    print(f"  p1Vect shape: {p1V.shape}")

    # Print log-probabilities for key words
    print(f"\n关键词语的对数概率 Word Log-Probabilities:")
    key_words = ['stupid', 'garbage', 'love', 'dog']
    for word in key_words:
        if word in vocabList:
            idx = vocabList.index(word)
            print(f"  '{word}': P(word|正常)={np.exp(p0V[idx]):.4f}, "
                  f"P(word|侮辱)={np.exp(p1V[idx]):.4f}")

    # --- Test on training data ---
    print(f"\n在训练数据上的测试结果 Test on Training Data:")
    all_correct = True
    for i, doc in enumerate(dataSet):
        vec = setOfWords2Vec(vocabList, doc)
        pred = classifyNB(vec, p0V, p1V, pAb)
        result = "CORRECT" if pred == labels[i] else "WRONG"
        if pred != labels[i]:
            all_correct = False
        print(f"  [{result}] '{' '.join(doc)}' -> 预测={pred}, 真实={labels[i]}")

    # --- Test on new sentences ---
    testSentences = [
        ['love', 'my', 'dog'],
        ['stupid', 'garbage'],
        ['you', 'are', 'so', 'cute'],
        ['worthless', 'stupid', 'dog'],
    ]
    print(f"\n对新句子的预测 Predictions on New Sentences:")
    for sent in testSentences:
        vec = setOfWords2Vec(vocabList, sent)
        pred = classifyNB(vec, p0V, p1V, pAb)
        tag = "侮辱" if pred == 1 else "正常"
        print(f"  [{tag}] '{' '.join(sent)}'")


# ===================================================================
# Task 2: Spam Email Classification
# ===================================================================
def generate_spam_ham_data():
    """Generate sample spam/ham email data (~20 emails)."""
    spam_emails = [
        "URGENT You have won 1000 dollars Claim your prize now Click this link",
        "FREE entry in our weekly competition Win a brand new car Hurry",
        "Congratulations You have been selected for a free cruise Vacation offer",
        "BUY NOW Limited time offer 50 percent discount on all products",
        "Make money fast from home Earn thousands per week No experience needed",
        "Exclusive deal Get rich quick with this amazing investment opportunity",
        "Double your income today This method really works Act now",
        "Lose weight fast without diet or exercise Try this miracle pill",
        "Free credit report Get your score instantly No obligation Click below",
        "Your PayPal account needs verification Update your details now",
    ]

    ham_emails = [
        "Hi John Are we still meeting for lunch tomorrow at noon",
        "Can you please review the attached report and send your feedback",
        "Reminder Team meeting is scheduled for Friday at 3pm in room 204",
        "Thanks for your help with the project I really appreciate it",
        "Mom asking what time you will arrive for dinner this Sunday",
        "Your order has been shipped and will arrive by Wednesday Track here",
        "The presentation went well Thanks everyone for your contributions",
        "Please update the documentation with the latest changes from sprint",
        "Hi Professor I have a question about the homework assignment due Monday",
        "Happy birthday Hope you have a wonderful day with family and friends",
    ]

    texts = spam_emails + ham_emails
    labels = np.array([1] * len(spam_emails) + [0] * len(ham_emails))
    return texts, labels


def task2_spam_classification():
    """Spam classification using sklearn MultinomialNB + CountVectorizer."""
    print("\n" + "=" * 60)
    print("Task 2: 垃圾邮件分类 Spam Classification")
    print("=" * 60)

    texts, labels = generate_spam_ham_data()
    print(f"\n数据集: {len(texts)} 封邮件 (垃圾邮件: {labels.sum()}, "
          f"正常邮件: {len(labels) - labels.sum()})")

    # Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # CountVectorizer + MultinomialNB
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred = nb.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率 Test Accuracy: {acc:.4f}")
    print("\n分类报告 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['正常 Ham', '垃圾 Spam']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常 Ham', '垃圾 Spam'],
                yticklabels=['正常 Ham', '垃圾 Spam'], ax=ax)
    ax.set_xlabel('预测 Predicted')
    ax.set_ylabel('真实 Actual')
    ax.set_title('垃圾邮件分类混淆矩阵\nSpam Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('practice/06_text_classification/01_spam_confusion_matrix.png', dpi=150)
    plt.close()
    print("[图表已保存] 01_spam_confusion_matrix.png")

    # Example predictions on new emails
    new_emails = [
        "Hi can we reschedule our meeting to next Tuesday",
        "WIN BIG PRIZES click here for your free gift today",
        "Please find attached the quarterly report for your review",
        "Make money online earn 500 dollars per day guaranteed",
    ]
    new_vec = vectorizer.transform(new_emails)
    new_pred = nb.predict(new_vec)
    print("\n新邮件预测 Example Predictions:")
    for email, pred in zip(new_emails, new_pred):
        tag = "垃圾邮件 SPAM" if pred == 1 else "正常 HAM"
        print(f"  [{tag}] \"{email}\"")


# ===================================================================
# Task 3: Chinese News Classification
# ===================================================================
def generate_chinese_news_data():
    """Generate sample Chinese news data in 3 categories: 体育/科技/娱乐."""
    sports = [
        "中国队在世界杯预选赛中取得关键胜利球员表现出色",
        "NBA总决赛第七场湖人队逆转夺冠詹姆斯获得最有价值球员",
        "东京奥运会中国代表团获得38枚金牌位居奖牌榜第二",
        "中超联赛新赛季即将开幕各支球队积极备战引进外援",
        "网球大满贯赛事中国选手历史性闯入四强创最好成绩",
        "马拉松比赛今日举行来自全国的选手参加赛道风景优美",
        "中国女排在世界联赛中连胜三场展现出强大的团队实力",
        "游泳世锦赛中国选手打破亚洲纪录获得男子自由泳金牌",
        "足球青训体系建设取得进展为中国足球的未来发展奠定基础",
        "体操世锦赛中国队获得团体金牌展现出色的技术水平和稳定性",
        "羽毛球混合团体赛中国队击败日本队成功卫冕冠军",
        "校园足球联赛在全国各地展开培养学生对运动的兴趣和热爱",
        "乒乓球世界杯中国队包揽男女单打冠军展现绝对统治力",
        "冰雪运动在南方城市逐渐普及更多年轻人参与冬季项目",
        "武术比赛弘扬中华传统文化年轻选手展现精湛技艺",
    ]

    tech = [
        "人工智能技术快速发展深度学习在图像识别领域取得新突破",
        "华为发布最新款智能手机搭载自主研发芯片和鸿蒙操作系统",
        "量子计算机实现量子优越性在特定计算任务上超越经典计算机",
        "特斯拉推出全自动驾驶软件更新大幅提升车辆的环境感知能力",
        "区块链技术应用于供应链管理提高透明度和溯源效率",
        "中国空间站完成在轨建造航天员开展多项科学实验和技术验证",
        "5G网络覆盖范围持续扩大为物联网和智慧城市建设提供基础",
        "新能源汽车销量创历史新高电池技术和充电基础设施建设加速",
        "科学家研发出新型半导体材料有望突破芯片制程工艺瓶颈",
        "云计算市场规模持续增长企业数字化转型进程加快效率提升",
        "开源软件社区贡献者人数突破纪录推动技术创新和知识共享",
        "生物技术公司利用基因编辑技术开发出新型癌症治疗方案",
        "无人机物流配送在多个城市试点运营为快递行业带来革命",
        "网络安全技术升级应对日益复杂的网络攻击和数据泄露风险",
        "虚拟现实设备销量增长在教育培训和娱乐领域找到新应用场景",
    ]

    entertainment = [
        "热门电影票房突破50亿元创下国产电影票房新纪录",
        "知名歌手举办巡回演唱会门票开售即售罄歌迷热情高涨",
        "综艺节目收视率创新高真人秀节目内容引发观众热议",
        "著名导演新片即将开机主演阵容强大备受期待",
        "音乐节在多个城市同时举办吸引大量年轻观众参与",
        "电视剧改编自热门小说忠实还原原著精髓获观众好评",
        "网红直播带货销售额破亿新电商模式改变消费习惯",
        "电影节开幕红毯仪式星光熠熠众多明星出席活动",
        "经典老歌重新编曲翻唱引发怀旧热潮登上音乐排行榜",
        "动漫展会吸引数万粉丝参与cosplay表演精彩纷呈",
        "短剧剧集在视频平台热播紧凑叙事风格受到年轻观众喜爱",
        "颁奖典礼举行表彰年度优秀作品和演员现场气氛热烈",
        "传统戏曲与现代音乐融合创新表演形式吸引年轻观众",
        "美食纪录片展现各地饮食文化画面精美引发观看热潮",
        "儿童动画电影寓教于乐在暑期档获得家长和孩子的喜爱",
    ]

    texts = sports + tech + entertainment
    labels = np.array(
        [0] * len(sports) + [1] * len(tech) + [2] * len(entertainment)
    )
    categories = ['体育 Sports', '科技 Tech', '娱乐 Entertainment']
    return texts, labels, categories


def chinese_tokenizer(text):
    """Tokenize Chinese text. Uses jieba if available, else character-based."""
    if JIEBA_AVAILABLE:
        # Join all jieba-segmented words with space
        return ' '.join(jieba.cut(text))
    else:
        # Character-based: treat each character as a token
        return ' '.join(list(text))


def task3_chinese_news_classification():
    """Chinese news article classification with jieba + TfidfVectorizer + MultinomialNB."""
    print("\n" + "=" * 60)
    print("Task 3: 中文新闻分类 Chinese News Classification")
    print("=" * 60)

    texts, labels, categories = generate_chinese_news_data()
    n_per_class = [np.sum(labels == i) for i in range(3)]
    print(f"\n数据集: {len(texts)} 篇新闻")
    for i, cat in enumerate(categories):
        print(f"  {cat}: {n_per_class[i]} 篇")

    # Tokenize all texts
    tokenized_texts = [chinese_tokenizer(t) for t in texts]
    print(f"\n分词示例 Tokenization Example:")
    print(f"  原文: {texts[0][:50]}...")
    print(f"  分词: {tokenized_texts[0][:80]}...")
    if JIEBA_AVAILABLE:
        print("  (使用 jieba 分词)")
    else:
        print("  (使用基于字符的分词)")

    # Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        tokenized_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # TfidfVectorizer + MultinomialNB
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred = nb.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集准确率 Test Accuracy: {acc:.4f}")
    print("\n分类报告 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories, ax=ax)
    ax.set_xlabel('预测 Predicted')
    ax.set_ylabel('真实 Actual')
    ax.set_title('中文新闻分类混淆矩阵\nChinese News Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig('practice/06_text_classification/02_chinese_news_confusion_matrix.png', dpi=150)
    plt.close()
    print("[图表已保存] 02_chinese_news_confusion_matrix.png")

    # Predict new articles
    new_articles = [
        "中国篮球队在国际比赛中表现出色年轻球员展现潜力",
        "人工智能芯片研发取得重大突破性能提升数倍",
        "新上映的爱情电影观众反响热烈票房持续走高",
    ]
    expected_cats = [categories[0], categories[1], categories[2]]  # 体育, 科技, 娱乐

    print("\n新文章预测 Predict New Articles:")
    for article, expected in zip(new_articles, expected_cats):
        tokenized = chinese_tokenizer(article)
        vec = vectorizer.transform([tokenized])
        pred = nb.predict(vec)[0]
        print(f"  文章: \"{article}\"")
        print(f"    预测类别: {categories[pred]}")
        print(f"    期望类别: {expected}")

    # Per-category top features
    print(f"\n各类别最重要的TF-IDF特征词 Top Features per Category:")
    feature_names = vectorizer.get_feature_names_out()
    for i, cat in enumerate(categories):
        top_indices = np.argsort(nb.feature_log_prob_[i])[-5:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        print(f"  {cat}: {', '.join(top_words)}")

    # Bar chart of per-class accuracy
    class_acc = []
    for i in range(3):
        mask = (y_test == i)
        if mask.sum() > 0:
            class_acc.append(accuracy_score(y_test[mask], y_pred[mask]))
        else:
            class_acc.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(categories, class_acc, color=colors)
    ax.set_ylabel('准确率 Accuracy')
    ax.set_title('中文新闻分类 - 各类别准确率\nPer-Category Accuracy')
    for bar, val in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=11)
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig('practice/06_text_classification/03_per_category_accuracy.png', dpi=150)
    plt.close()
    print("[图表已保存] 03_per_category_accuracy.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("基于朴素贝叶斯的文本分类 Text Classification with Naive Bayes")
    print("=" * 60)

    task1_insulting_comment_filter()
    task2_spam_classification()
    task3_chinese_news_classification()

    print("\n" + "=" * 60)
    print("所有任务完成 All Tasks Completed")
    print("=" * 60)
