# 基础机器学习实例仓库

基于 Python 实现的基础机器学习实例集合，涵盖监督学习、无监督学习、集成学习、降维、聚类、神经网络等核心算法及其业务场景应用。所有模块均可独立运行，大部分算法同时提供**手动实现**和 **sklearn 调包**两个版本以便对照学习。

## 快速开始

```bash
pip install -r requirements.txt
python 04_knn/main.py                # 运行单个实验
python practice/01_california_housing/main.py
```

## 项目结构

```
├── 01_python_basics/              # Python编程基础与numpy练习
├── 02_linear_regression/          # 线性回归 (最小二乘法从零实现)
├── 03_polynomial_regression/      # 多项式回归 (GDP预测、时间-浓度回归)
├── 04_knn/                        # K近邻算法 (手动实现分类器与回归器)
├── 05_decision_tree/              # 决策树 (ID3/C4.5, 分类)
├── 06_naive_bayes/                # 朴素贝叶斯 (手动高斯NB + sklearn多项式NB)
├── 07_svm_principles/             # 支持向量机原理 (几何中心法 + 角度扫描最大间隔)
├── 08_svm_practice/               # 支持向量机实践 (多核对比、多分类器对比)
├── 09_model_evaluation/           # 模型性能评估 (混淆矩阵、PR/ROC曲线、不平衡分析)
├── 10_ensemble_learning/          # 集成学习 (Bagging/RF/AdaBoost/GBDT, 回归+分类)
├── 11_dimensionality_reduction/   # 降维算法 (PCA从零实现 + sklearn, 3D可视化)
├── 12_clustering/                 # 聚类 (K-means/层次/DBSCAN, 最优k值确定)
├── 13_neural_networks/            # 神经网络 (MLP分类+回归, GridSearchCV调参)
├── 14_final_project/              # 期末综合项目 (回归+分类+降维+不平衡处理)
├── practice/
│   ├── 01_california_housing/     # 加州房价预测 (一元/多元/多项式回归对比)
│   ├── 02_german_credit/          # 德国信用风险分析 (手动梯度下降逻辑回归)
│   ├── 03_movie_recommendation/   # 电影推荐系统 (协同过滤, Precision@N/Recall@N)
│   ├── 04_credit_card_fraud/      # 信用卡欺诈检测 (SMOTE过采样, 召回率优化)
│   ├── 05_handwritten_digits/     # 手写数字识别 (KNN/Softmax/决策树, 对抗攻击)
│   └── 06_text_classification/    # 文本分类 (手动朴素贝叶斯, 垃圾邮件, 新闻分类)
└── data/                          # 数据集目录 (脚本会自动生成缺失的数据文件)
```

## 核心实验详情

### 01 — Python 编程基础
- NumPy 数组操作与文件读写
- 学生成绩筛选、课程统计 (最高分/最低分/及格率/均值/标准差)
- 成绩相关性分析 (Pearson/Spearman 相关系数、散点图)
- 成绩分布分析 (直方图、Q-Q图、Shapiro-Wilk 正态性检验)

### 02 — 线性回归
- 从零实现最小二乘法 (正规方程 `θ = (XᵀX)⁻¹Xᵀy`)
- 房屋面积 vs 价格 (12 个样本)
- 身高 vs 体重 (15 个样本)
- 与 sklearn LinearRegression 对比

### 03 — 多项式回归
- 模拟二次数据 `y = 0.9x² + 0.1x + 2 + ε`，验证系数还原
- GDP 预测 (2004–2023 中国人均 GDP，预测 2024 年)
- 时间-浓度回归: 线性 vs 二次多项式对比 (RMSE + R²)

### 04 — K 近邻算法
- **手动 KNN 分类器**: 昆虫分类 (触长/翅长, k=1,3,5)，欧氏距离 + 多数投票
- **手动 KNN 回归器**: 运输成本预测 (体积/重量, 算术平均 vs 距离加权)
- sklearn KNN 回归 + LOOCV 评估 (体脂率数据集, k=1,3,5,7,10)

### 05 — 决策树
- 信用卡交易分类 (15 条, ID3/信息增益, 5 折交叉验证)
- 葡萄酒分类 (sklearn load_wine, max_depth=4, 特征重要性)
- 天气-游玩决策 (14 条训练, 4 条测试, 文本决策规则输出)

### 06 — 朴素贝叶斯分类器
- 乳腺癌诊断 (sklearn GaussianNB, 准确率 + 概率输出)
- 文本情感分析 (12 句, CountVectorizer + MultinomialNB)
- 学生成绩分类 — **手动高斯 NB 实现** (先验概率、正态似然、后验概率)

### 07 — 支持向量机原理
- 简化 SVM (几何中心法): 求两类中心 → 垂直平分线作为分类边界
- 近似最大间隔 SVM (角度扫描法): 0~π 扫描投影方向, 寻找最大 margin

### 08 — 支持向量机实践
- 鸢尾花二分类 (LinearSVC, 不同 C 值对比, 决策边界可视化)
- 乳腺癌分类 (4 种核函数对比: linear/poly/rbf/sigmoid, GridSearchCV 调参)
- 糖尿病多分类器对比 (SVM vs KNN vs 决策树, ROC 曲线, 综合指标排名)

### 09 — 模型性能评估
- 混淆矩阵、准确率、精确率、召回率、F1 分数详解
- 逻辑回归 vs 决策树 (5 折交叉验证)
- 类别不平衡分析 (糖尿病数据集 500:268): 为何准确率不可靠
- P-R 曲线 vs ROC 曲线 (朴素贝叶斯 vs SVM, 适用场景讨论)

### 10 — 集成学习
- 分类: Bagging / 随机森林 / AdaBoost / GBDT 四方法对比 (10 折 CV)
- 回归: 体脂率预测, 单棵树 → 集成方法的性能跃升
- 波士顿房价: **10 种方法**全面对比 (线性回归 → GBDT), 自定义正确率指标

### 11 — 降维算法
- **手动 PCA 实现** (协方差矩阵 + 特征值分解) vs sklearn PCA
- 主成分选取准则: 累计贡献率 (85%/95%) + Kaiser 准则 (特征值>1) + 碎石图
- 小麦种子数据集: 3D PCA 可视化 (3 个视角 + 3 个投影面)

### 12 — 聚类
- K-means (k=3) vs 层次聚类 (Ward): 轮廓系数/CH指数/DB指数
- DBSCAN: k-distance 图确定 eps, 噪声点识别与标注
- 最优 k 值确定: 肘部法则 + 轮廓系数 + CH 指数综合判断

### 13 — 神经网络
- MLP 分类 (乳腺癌, hidden_layer_sizes=(64,32), ReLU, Adam)
- MLP 回归 (体脂率, early_stopping, 残差分析)
- 多方法对比: MLP (GridSearchCV 24 种组合) vs SVM vs 朴素贝叶斯

### 14 — 期末综合项目
- **Part 1 (40%)**: 地深肥沃指数预测 (线性回归 vs SVR, GridSearch 48 种参数组合)
- **Part 2 (40%)**: 乳腺癌分类 (GaussianNB vs 随机森林, 完整评估)
- **Part 3 (20%)**: 不平衡数据处理 (5 分类器 StratifiedKFold, PCA + t-SNE 可视化)

## 综合实践详情

### 实践 1 — 加州房价预测
模块化设计: `HousingDataAnalyzer` → `UnivariateLinearRegression` → `MultivariateLinearRegression` → `PolynomialRegression`。自动寻找最优多项式次数 (1-4)，对比三种回归方法的性能提升。

### 实践 2 — 德国信用风险分析
**手动实现逻辑回归**: sigmoid 函数 + 梯度下降 + 早停机制。混合型数据预处理 (类别编码/独热编码/标准化/特征工程)，与 sklearn 对比。

### 实践 3 — 电影推荐系统
基于用户的协同过滤: 余弦相似度 + Jaccard 相似度, k=10 近邻搜索, 评分预测 (均值中心化加权)。离线评估: Precision@N / Recall@N (N=5/10/20)。

### 实践 4 — 信用卡欺诈检测
极度不平衡数据 (欺诈 ~0.5%)。SMOTE 过采样, 逻辑回归 vs KNN, 重点关注**召回率** (漏检欺诈的代价远大于误报)。

### 实践 5 — 手写数字识别
四个子任务: 数据探索 (8×8 灰度图可视化) → KNN (网格搜索最优 k) → Softmax 逻辑回归 (权重热力图) → 决策树 (对抗攻击: 翻转 top-3 像素, 观察预测变化)。

### 实践 6 — 文本分类
- 子任务 1: **手动实现朴素贝叶斯** (词表构建 → 词向量化 → 训练 → 分类, 拉普拉斯平滑)
- 子任务 2: 垃圾邮件分类 (sklearn CountVectorizer + MultinomialNB)
- 子任务 3: 中文新闻分类 (jieba 分词 + TfidfVectorizer, 3 类别)

## 算法总览

| 类别 | 算法 | 应用场景 |
|------|------|----------|
| 回归 | 线性回归、多项式回归、SVR、MLP | 房价、GDP、体脂率、肥沃指数 |
| 分类 | KNN、决策树、朴素贝叶斯、SVM、逻辑回归 | 信用评估、欺诈检测、数字识别、疾病诊断 |
| 集成学习 | Bagging、随机森林、AdaBoost、GBDT | 乳腺癌、房价、体脂率 |
| 聚类 | K-means、层次聚类、DBSCAN | 鸢尾花、无标签数据探索 |
| 降维 | PCA、t-SNE | 高维数据可视化、特征压缩 |
| 神经网络 | MLP | 分类、回归 |
| 推荐系统 | 协同过滤 (User-based CF) | 电影推荐 |
| 文本处理 | 词袋模型、TF-IDF、jieba 分词 | 情感分析、垃圾邮件、新闻分类 |

## 环境要求

- Python 3.8+
- 主要依赖: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, imbalanced-learn
- 中文显示: 需要系统安装 SimHei 字体 (Windows 自带)

## 注意事项

- 每个模块的 `main.py` 完全独立, 无跨模块依赖, 可直接运行
- 大多数模块在数据文件缺失时会自动生成模拟数据 (固定随机种子 42 保证可复现)
- 所有可视化图表保存为 PNG 文件, 位于各自模块目录下
- 实验报告 (PDF) 位于 `实验报告/` 目录, 可供参考算法原理与结果分析
