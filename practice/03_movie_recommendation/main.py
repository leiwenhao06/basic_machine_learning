# -*- coding: utf-8 -*-
"""
影评数据分析与电影推荐 — 协同过滤 (Movie Recommendation)
===========================================================================
包含:
  - 数据生成 (movies.csv / ratings.csv)
  - 数据探索 (评分分布、用户/电影活跃度)
  - User-based 协同过滤
    * 余弦相似度 & Jaccard 相似度
    * k 近邻搜索 (k=10)
    * 基于近邻的评分预测
    * Top-N 推荐
  - 离线评估: Precision@N / Recall@N
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

from collections import defaultdict

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)

MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")


# ==================== 数据生成 ====================
MOVIE_TITLES = [
    "Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)",
    "Waiting to Exhale (1995)", "Father of the Bride II (1995)",
    "Heat (1995)", "Sabrina (1995)", "Tom and Huck (1995)",
    "Sudden Death (1995)", "GoldenEye (1995)",
    "The American President (1995)", "Dracula: Dead and Loving It (1995)",
    "Balto (1995)", "Nixon (1995)", "Cutthroat Island (1995)",
    "Casino (1995)", "Sense and Sensibility (1995)", "Four Rooms (1995)",
    "Ace Ventura: When Nature Calls (1995)", "Money Train (1995)",
    "Get Shorty (1995)", "Copycat (1995)", "Assassins (1995)",
    "Powder (1995)", "Leaving Las Vegas (1995)", "Othello (1995)",
    "Now and Then (1995)", "Persuasion (1995)", "The City of Lost Children (1995)",
    "Shanghai Triad (1995)", "Dangerous Minds (1995)", "Twelve Monkeys (1995)",
    "Babe (1995)", "Dead Man Walking (1995)", "The Usual Suspects (1995)",
    "Mighty Aphrodite (1995)", "The Postman (1994)", "Braveheart (1995)",
    "Taxi Driver (1976)", "The Godfather (1972)", "Pulp Fiction (1994)",
    "The Shawshank Redemption (1994)", "Forrest Gump (1994)",
    "The Lion King (1994)", "Jurassic Park (1993)", "Schindler's List (1993)",
    "Fight Club (1999)", "The Matrix (1999)", "Inception (2010)",
    "Interstellar (2014)",
]


def _generate_movies(path):
    """生成 movies.csv."""
    df = pd.DataFrame({
        "movieId": np.arange(1, len(MOVIE_TITLES) + 1),
        "title": MOVIE_TITLES,
        "genres": np.random.choice(
            ["Action", "Comedy", "Drama", "Horror", "Sci-Fi",
             "Romance", "Thriller", "Adventure", "Animation"],
            size=len(MOVIE_TITLES)
        ),
    })
    df.to_csv(path, index=False)
    print(f"[INFO] 生成 movies.csv: {len(df)} 部电影")
    return df


def _generate_ratings(path, movies_df, seed=42):
    """生成 ratings.csv (~5000 条评分)。"""
    rng = np.random.RandomState(seed)
    n_users = 100
    movie_ids = movies_df["movieId"].values
    n_movies = len(movie_ids)

    ratings = []
    rating_id = 1

    # 多数用户会看 20-48 部电影，活跃度遵循幂律
    for uid in range(1, n_users + 1):
        # 每位用户看一定数量的电影 (偏态分布)
        n_rated = int(rng.exponential(scale=25) + 5)
        n_rated = min(n_rated, n_movies)
        watched = rng.choice(movie_ids, size=n_rated, replace=False)

        # 用户偏好向量 (某些用户偏爱特定类型 → 给高分)
        user_bias = rng.uniform(2.0, 4.5)
        for mid in watched:
            base_rating = user_bias + rng.normal(0, 0.8)
            # 部分电影天然高分
            if mid % 7 == 0:
                base_rating += 0.8
            elif mid % 13 == 0:
                base_rating -= 0.5
            rating = np.clip(np.round(base_rating), 1, 5).astype(int)
            ratings.append({
                "userId": uid,
                "movieId": mid,
                "rating": rating,
                "timestamp": 1000000000 + rating_id * 1000,
            })
            rating_id += 1

    df = pd.DataFrame(ratings)
    # 随机打乱
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[INFO] 生成 ratings.csv: {len(df)} 条评分, {n_users} 位用户")
    return df


def load_or_generate():
    """加载或生成 movie/ratings 数据。"""
    if os.path.exists(MOVIES_CSV) and os.path.exists(RATINGS_CSV):
        print(f"[INFO] 从文件加载数据")
        movies = pd.read_csv(MOVIES_CSV)
        ratings = pd.read_csv(RATINGS_CSV)
    else:
        print("[INFO] 数据文件不存在, 生成示例数据")
        movies = _generate_movies(MOVIES_CSV)
        ratings = _generate_ratings(RATINGS_CSV, movies)
    return movies, ratings


# ====================================================================
# 1. 数据探索
# ====================================================================
def explore_data(ratings, movies):
    """数据探索与可视化。"""
    print("\n" + "=" * 60)
    print("数据探索")
    print("=" * 60)
    print(f"评分记录数: {len(ratings)}")
    print(f"用户数:      {ratings['userId'].nunique()}")
    print(f"电影数:      {ratings['movieId'].nunique()}")
    print(f"评分均值:    {ratings['rating'].mean():.2f}")
    print(f"稀疏度:      {1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.4f}")

    # 评分分布
    user_counts = ratings.groupby("userId").size()
    movie_counts = ratings.groupby("movieId").size()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 评分分布
    axes[0].bar(*np.unique(ratings["rating"], return_counts=True),
                color=["#d62728", "#ff7f0e", "#ffbb00", "#2ca02c", "#1f77b4"])
    axes[0].set_xlabel("评分")
    axes[0].set_ylabel("频数")
    axes[0].set_title("评分分布")

    # 用户评分数量分布
    axes[1].hist(user_counts, bins=30, edgecolor="black")
    axes[1].set_xlabel("评分数")
    axes[1].set_ylabel("用户数")
    axes[1].set_title("用户活跃度分布")

    # 电影被评数量分布
    axes[2].hist(movie_counts, bins=30, edgecolor="black")
    axes[2].set_xlabel("被评次数")
    axes[2].set_ylabel("电影数")
    axes[2].set_title("电影热度分布")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "11_data_exploration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {path}")

    return user_counts, movie_counts


# ====================================================================
# 2. User-based 协同过滤
# ====================================================================
class UserBasedCF:
    """User-based 协同过滤推荐系统。"""

    def __init__(self, ratings_df, k=10, min_common=3):
        self.k = k
        self.min_common = min_common
        self.ratings = ratings_df
        self.user_ids = sorted(ratings_df["userId"].unique())
        self.movie_ids = sorted(ratings_df["movieId"].unique())

        # 映射
        self.user2idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie2idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2movie = {i: m for m, i in self.movie2idx.items()}

        # 构建评分矩阵 (NaN 表示未评分)
        self.R = np.full((len(self.user_ids), len(self.movie_ids)), np.nan)
        for _, row in ratings_df.iterrows():
            ui = self.user2idx[row["userId"]]
            mi = self.movie2idx[row["movieId"]]
            self.R[ui, mi] = row["rating"]

        # 用户均值 (用于中心化)
        self.user_means = np.nanmean(self.R, axis=1)

        # 中心化评分矩阵
        self.R_centered = self.R - self.user_means[:, np.newaxis]
        # NaN → 0 (用于相似度计算)
        self.R_centered_filled = np.nan_to_num(self.R_centered, nan=0.0)

        # 相似度缓存
        self.cosine_sim = None
        self.jaccard_sim = None

    def _jaccard_coef(self, u, v):
        """两个用户评分的 Jaccard 系数。"""
        mask_u = ~np.isnan(self.R[u])
        mask_v = ~np.isnan(self.R[v])
        inter = np.sum(mask_u & mask_v)
        union = np.sum(mask_u | mask_v)
        return inter / union if union > 0 else 0.0

    def compute_similarities(self):
        """计算用户间的余弦相似度与 Jaccard 相似度。"""
        n_users = len(self.user_ids)
        self.cosine_sim = np.zeros((n_users, n_users))
        self.jaccard_sim = np.zeros((n_users, n_users))

        # 余弦相似度 (基于中心化评分, 使用向量化)
        norms = np.linalg.norm(self.R_centered_filled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        R_normed = self.R_centered_filled / norms
        self.cosine_sim = R_normed @ R_normed.T
        np.fill_diagonal(self.cosine_sim, 0.0)

        # Jaccard 相似度
        mask = ~np.isnan(self.R)
        for i in range(n_users):
            for j in range(i + 1, n_users):
                inter = np.sum(mask[i] & mask[j])
                union = np.sum(mask[i] | mask[j])
                jac = inter / union if union > 0 else 0.0
                self.jaccard_sim[i, j] = jac
                self.jaccard_sim[j, i] = jac

        print(f"[相似度] 余弦相似度范围: [{self.cosine_sim.min():.3f}, {self.cosine_sim.max():.3f}]")
        print(f"[相似度] Jaccard 范围:     [{self.jaccard_sim.min():.3f}, {self.jaccard_sim.max():.3f}]")

    def _get_neighbors(self, user_idx, sim_matrix):
        """获取 k 个最近邻居 (排除自身)。"""
        sims = sim_matrix[user_idx].copy()
        sims[user_idx] = -np.inf
        # 只保留有共同评分 >= min_common 的邻居
        for v in range(len(self.user_ids)):
            if v == user_idx:
                continue
            common = np.sum(~np.isnan(self.R[user_idx]) & ~np.isnan(self.R[v]))
            if common < self.min_common:
                sims[v] = -np.inf
        neighbors = np.argsort(sims)[::-1][:self.k]
        return [(v, sim_matrix[user_idx, v]) for v in neighbors if sim_matrix[user_idx, v] > -1e9]

    def predict_rating(self, user_idx, movie_idx, sim_matrix):
        """预测指定用户对某电影的评分。"""
        if not np.isnan(self.R[user_idx, movie_idx]):
            return self.R[user_idx, movie_idx]

        num = 0.0
        den = 0.0
        for v_idx, sim in self._get_neighbors(user_idx, sim_matrix):
            if not np.isnan(self.R[v_idx, movie_idx]):
                num += sim * self.R_centered[v_idx, movie_idx]
                den += abs(sim)

        if den < 1e-8:
            return self.user_means[user_idx]
        return self.user_means[user_idx] + num / den

    def recommend(self, target_user_id, sim_matrix, N=10,
                  rating_threshold=3.5, exclude_watched=True):
        """为目标用户推荐 Top-N 电影。

        - 获取近邻喜欢的电影 (rating >= threshold)
        - 去除已看过的电影
        - 按预测评分排序
        """
        user_idx = self.user2idx[target_user_id]

        # 已看过的电影索引
        watched = set()
        if exclude_watched:
            watched = set(np.where(~np.isnan(self.R[user_idx]))[0])

        # 近邻喜欢 (> threshold) 的电影
        neighbor_movies = set()
        for v_idx, _ in self._get_neighbors(user_idx, sim_matrix):
            liked = np.where(
                (~np.isnan(self.R[v_idx])) & (self.R[v_idx] >= rating_threshold)
            )[0]
            neighbor_movies.update(liked)

        candidates = neighbor_movies - watched
        if not candidates:
            # 回退: 近邻评分过的所有未看过的电影
            for v_idx, _ in self._get_neighbors(user_idx, sim_matrix):
                rated = set(np.where(~np.isnan(self.R[v_idx]))[0])
                candidates.update(rated - watched)

        # 预测评分并排序
        scored = []
        for mi in candidates:
            pred = self.predict_rating(user_idx, mi, sim_matrix)
            scored.append((mi, pred))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = scored[:N]
        return [(self.idx2movie[mi], pred) for mi, pred in top_n]


# ====================================================================
# 3. 离线评估
# ====================================================================
def evaluate(cf: UserBasedCF, sim_matrix, N=10, test_ratio=0.2, seed=42):
    """评估 Precision@N 和 Recall@N。

    对每个用户随机留出 20% 评分作为测试集，用剩余 80% 训练，
    评估推荐的 Top-N 电影中有多少在测试集中且评分 >= 4.0。
    """
    rng = np.random.RandomState(seed)
    precision_scores = []
    recall_scores = []

    for user_id in cf.user_ids:
        ui = cf.user2idx[user_id]
        rated_indices = np.where(~np.isnan(cf.R[ui]))[0]
        if len(rated_indices) < 8:
            continue

        n_test = max(1, int(len(rated_indices) * test_ratio))
        test_idx = rng.choice(rated_indices, size=n_test, replace=False)
        train_idx = np.setdiff1d(rated_indices, test_idx)

        # 临时修改 R 矩阵 (仅保留训练评分)
        original_row = cf.R[ui].copy()
        cf.R[ui] = np.full(cf.R.shape[1], np.nan)
        cf.R[ui, train_idx] = original_row[train_idx]
        cf.user_means[ui] = np.nanmean(cf.R[ui])

        # 重新计算中心化矩阵
        cf.R_centered[ui] = cf.R[ui] - cf.user_means[ui]
        cf.R_centered_filled[ui] = np.nan_to_num(cf.R_centered[ui], nan=0.0)

        # 推荐
        recs = cf.recommend(user_id, sim_matrix, N=N, rating_threshold=4.0,
                            exclude_watched=True)
        rec_movies = {mid for mid, _ in recs}

        # 测试集中真正喜欢的 (rating >= 4.0)
        test_relevant = set()
        for mi in test_idx:
            if original_row[mi] >= 4.0:
                test_relevant.add(cf.idx2movie[mi])

        hit = rec_movies & test_relevant
        prec = len(hit) / N if N > 0 else 0.0
        rec = len(hit) / len(test_relevant) if test_relevant else 0.0
        precision_scores.append(prec)
        recall_scores.append(rec)

        # 恢复
        cf.R[ui] = original_row
        cf.user_means[ui] = np.nanmean(cf.R[ui])
        cf.R_centered[ui] = cf.R[ui] - cf.user_means[ui]
        cf.R_centered_filled[ui] = np.nan_to_num(cf.R_centered[ui], nan=0.0)

    avg_prec = np.mean(precision_scores) if precision_scores else 0.0
    avg_rec = np.mean(recall_scores) if recall_scores else 0.0
    return avg_prec, avg_rec, precision_scores, recall_scores


# ====================================================================
# main
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("影评数据分析与电影推荐 — 协同过滤")
    print("=" * 60)

    # ---- 加载/生成数据 ----
    movies, ratings = load_or_generate()

    # ---- 数据探索 ----
    user_counts, movie_counts = explore_data(ratings, movies)

    # ---- 构建 CF 系统 ----
    print("\n" + "=" * 60)
    print("User-based 协同过滤")
    print("=" * 60)
    cf = UserBasedCF(ratings, k=10, min_common=3)
    cf.compute_similarities()

    # ---- 为目标用户推荐 ----
    target_user = cf.user_ids[0]
    print(f"\n目标用户: userId={target_user}")

    # 该用户已评分的电影
    ui = cf.user2idx[target_user]
    rated_mask = ~np.isnan(cf.R[ui])
    watched_movies = [(cf.idx2movie[mi], cf.R[ui, mi])
                      for mi in np.where(rated_mask)[0]]
    watched_movies.sort(key=lambda x: x[1], reverse=True)
    print(f"  用户已评分 {len(watched_movies)} 部电影")
    print(f"  平均评分: {cf.user_means[ui]:.2f}")
    print(f"  高分电影 (>=4):")
    for mid, r in watched_movies[:5]:
        title = movies.loc[movies["movieId"] == mid, "title"].values
        title = title[0] if len(title) > 0 else str(mid)
        print(f"    [{r}] {title}")

    # 余弦相似度推荐
    print(f"\n  --- 余弦相似度 Top-10 推荐 ---")
    recs_cos = cf.recommend(target_user, cf.cosine_sim, N=10, rating_threshold=3.5)
    for rank, (mid, pred) in enumerate(recs_cos, 1):
        title = movies.loc[movies["movieId"] == mid, "title"].values
        title = title[0] if len(title) > 0 else str(mid)
        print(f"    #{rank}  [预测评分={pred:.2f}] {title}")

    # Jaccard 相似度推荐
    print(f"\n  --- Jaccard 相似度 Top-10 推荐 ---")
    recs_jac = cf.recommend(target_user, cf.jaccard_sim, N=10, rating_threshold=3.5)
    for rank, (mid, pred) in enumerate(recs_jac, 1):
        title = movies.loc[movies["movieId"] == mid, "title"].values
        title = title[0] if len(title) > 0 else str(mid)
        print(f"    #{rank}  [预测评分={pred:.2f}] {title}")

    # ---- 相似度矩阵可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(cf.cosine_sim, cmap="RdYlBu", aspect="auto")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title("用户余弦相似度矩阵")
    axes[0].set_xlabel("用户索引")
    axes[0].set_ylabel("用户索引")

    im1 = axes[1].imshow(cf.jaccard_sim, cmap="RdYlBu", aspect="auto")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    axes[1].set_title("用户 Jaccard 相似度矩阵")
    axes[1].set_xlabel("用户索引")
    axes[1].set_ylabel("用户索引")
    plt.tight_layout()
    sim_path = os.path.join(FIG_DIR, "12_similarity_matrices.png")
    plt.savefig(sim_path, dpi=150)
    plt.close()
    print(f"\n[图表] 已保存: {sim_path}")

    # ---- 离线评估 ----
    print("\n" + "=" * 60)
    print("离线评估 (Precision@N / Recall@N)")
    print("=" * 60)

    for N in [5, 10, 20]:
        prec_cos, rec_cos, _, _ = evaluate(cf, cf.cosine_sim, N=N)
        prec_jac, rec_jac, _, _ = evaluate(cf, cf.jaccard_sim, N=N)
        print(f"\n  N={N}:")
        print(f"    余弦相似度:   Precision@{N} = {prec_cos:.4f},  Recall@{N} = {rec_cos:.4f}")
        print(f"    Jaccard:      Precision@{N} = {prec_jac:.4f},  Recall@{N} = {rec_jac:.4f}")

    # ---- Precision/Recall vs N 曲线 ----
    print("\n[评估] 绘制 Precision/Recall vs N 曲线...")
    N_values = list(range(1, 31, 2))
    prec_cos_vals, rec_cos_vals = [], []
    prec_jac_vals, rec_jac_vals = [], []

    for N in N_values:
        pc, rc, _, _ = evaluate(cf, cf.cosine_sim, N=N)
        pj, rj, _, _ = evaluate(cf, cf.jaccard_sim, N=N)
        prec_cos_vals.append(pc)
        rec_cos_vals.append(rc)
        prec_jac_vals.append(pj)
        rec_jac_vals.append(rj)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(N_values, prec_cos_vals, "o-", label="Precision (余弦)", markersize=5)
    ax.plot(N_values, rec_cos_vals, "s--", label="Recall (余弦)", markersize=5)
    ax.plot(N_values, prec_jac_vals, "D-", label="Precision (Jaccard)", markersize=5)
    ax.plot(N_values, rec_jac_vals, "^--", label="Recall (Jaccard)", markersize=5)
    ax.set_xlabel("推荐数量 N")
    ax.set_ylabel("值")
    ax.set_title("Precision@N & Recall@N 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    eval_path = os.path.join(FIG_DIR, "13_evaluation_curves.png")
    plt.savefig(eval_path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {eval_path}")

    # ---- 余弦 vs Jaccard 相似度对比散点 ----
    non_diag = ~np.eye(cf.cosine_sim.shape[0], dtype=bool)
    cos_flat = cf.cosine_sim[non_diag]
    jac_flat = cf.jaccard_sim[non_diag]
    plt.figure(figsize=(6, 6))
    plt.scatter(jac_flat, cos_flat, alpha=0.3, s=8)
    plt.xlabel("Jaccard 相似度")
    plt.ylabel("余弦相似度")
    plt.title("余弦 vs Jaccard 相似度")
    corr_coef = np.corrcoef(cos_flat, jac_flat)[0, 1]
    plt.text(0.05, 0.95, f"r = {corr_coef:.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment="top")
    plt.tight_layout()
    comp_path = os.path.join(FIG_DIR, "14_similarity_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"[图表] 已保存: {comp_path}")

    print("\n" + "=" * 60)
    print("所有任务完成! PNG 图表保存于:", FIG_DIR)
    print("=" * 60)
