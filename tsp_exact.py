# -*- coding: utf-8 -*-
"""
TSP 最优求解（Exact）：Held–Karp / 暴力遍历
--------------------------------------------------
风格与 tsp_ga.py 保持一致：
1) 随机生成 TSP 节点（含仓库 0，seed=42 与 tsp_ga.py 一致）；
2) 使用 Held–Karp（默认）或穷举遍历获得“全局最优”；
3) 保存收敛/过程信息与最优路线图（非交互式后端）。

注意：
- Held–Karp 适合 n<=22 左右；穷举遍历建议 n<=11。
- 若你想与 tsp_ga.py 完全同规模(如 n=40)，请只用 GA/ACO/SA；
  Exact 方法在该规模不可行（指数级）。
"""

import math
import random
from typing import List, Tuple, Dict, Optional
import itertools
import time
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAVE_DIR = "figs_tsp_exact"   # 输出图片目录


# ==============================
# 工具函数（与 tsp_ga.py 风格一致）
# ==============================

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def tour_length(tour: List[int], D: np.ndarray) -> float:
    """tour 仅包含客户 [1..N]，总长度= 0->tour[0] + ... + tour[-1]->0"""
    if not tour:
        return 0.0
    s = D[0, tour[0]]
    for i in range(len(tour) - 1):
        s += D[tour[i], tour[i+1]]
    s += D[tour[-1], 0]
    return s


def generate_random_coords(n_customers: int = 30, plane_size: int = 100, seed: int = 42) -> List[Tuple[float, float]]:
    """与 tsp_ga.py 相同风格/接口；仓库在中心，其余均匀随机；seed=42 与 GA 默认一致"""
    rng = random.Random(seed)
    coords = [(plane_size/2.0, plane_size/2.0)]  # depot 0
    for _ in range(n_customers):
        x = rng.uniform(0, plane_size)
        y = rng.uniform(0, plane_size)
        coords.append((x, y))
    return coords


def plot_route(coords: List[Tuple[float, float]], tour: List[int], title: str = "Exact Best Route"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    xs = [coords[0][0]] + [coords[i][0] for i in tour] + [coords[0][0]]
    ys = [coords[0][1]] + [coords[i][1] for i in tour] + [coords[0][1]]
    plt.figure()
    plt.scatter([c[0] for c in coords], [c[1] for c in coords])
    plt.plot(xs, ys)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i))
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "exact_route.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ==============================
# 1) Held–Karp 动态规划（Exact）
#    返回最优回路长度与客户序列（不含仓库0）
# ==============================

def held_karp_tsp(D: np.ndarray) -> Dict:
    """
    状态 DP[S][j]：从 0 出发，访问 S（客户集合，位集表示），以 j 结尾的最短路长。
    转移：DP[S][j] = min_{i in S-{j}} DP[S-{j}][i] + d(i,j)
    最后加回到 0。
    """
    n = D.shape[0] - 1  # 客户数
    if n <= 0:
        return {"best_distance": 0.0, "best_tour": []}

    # 客户编号 1..n -> 压缩到 0..n-1 用于位运算
    ALL = 1 << n
    INF = float("inf")

    # DP 与前驱
    dp = [ [INF]*(n) for _ in range(ALL) ]
    parent = [ [None]*(n) for _ in range(ALL) ]

    # 初始：只含单个城市 j 的集合 {j}
    for j in range(n):
        mask = 1 << j
        dp[mask][j] = D[0, j+1]  # 0->(j+1)

    # 枚举子集大小 k=2..n
    for k in range(2, n+1):
        for S in range(ALL):
            if bin(S).count("1") != k:
                continue
            # 枚举结尾 j
            for j in range(n):
                if not (S & (1 << j)):
                    continue
                Sj = S ^ (1 << j)
                # 枚举倒数第二个 i
                best_val = dp[S][j]
                best_i = parent[S][j]
                # 若子集只有 j 一个，不必更新
                if Sj == 0:
                    continue
                # 过所有 i
                i_mask = Sj
                while i_mask:
                    i = (i_mask & -i_mask).bit_length() - 1  # 取最低位 idx
                    i_mask ^= (1 << i)
                    cand = dp[Sj][i] + D[i+1, j+1]
                    if cand < best_val:
                        best_val = cand
                        best_i = i
                dp[S][j] = best_val
                parent[S][j] = best_i

    # 终止：min_j dp[ALL-1][j] + d(j->0)
    best_len = INF
    last = None
    S = ALL - 1
    for j in range(n):
        cand = dp[S][j] + D[j+1, 0]
        if cand < best_len:
            best_len = cand
            last = j

    # 还原路径（客户索引转回 1..n）
    tour_rev = []
    cur = last
    curS = S
    while cur is not None:
        tour_rev.append(cur+1)
        nxt = parent[curS][cur]
        curS ^= (1 << cur)
        cur = nxt
    best_tour = tour_rev[::-1]  # 从首到尾

    return {"best_distance": best_len, "best_tour": best_tour}


# ==============================
# 2) 纯暴力遍历（Exact, 小 n）
# ==============================

def brute_force_tsp(D: np.ndarray) -> Dict:
    """
    对客户 1..n 的所有排列做穷举，返回最短回路。
    仅适合 n<=11 左右。更大请用 Held–Karp。
    """
    n = D.shape[0] - 1
    best_len = float("inf")
    best_tour: List[int] = []

    for perm in itertools.permutations(range(1, n+1)):
        L = D[0, perm[0]] + sum(D[perm[i], perm[i+1]] for i in range(n-1)) + D[perm[-1], 0]
        if L < best_len:
            best_len = L
            best_tour = list(perm)
    return {"best_distance": best_len, "best_tour": best_tour}


# ==============================
# 主程序（可直接运行）
# ==============================

def main():
    # —— 与 tsp_ga.py 保持“随机地图实例一致”：
    n_customers = 20            # ★如需 Held–Karp，建议 <=22；若要暴力遍历建议 <=11
    plane_size = 500            # 与你 GA 代码一致的坐标量级
    seed = 42                   # 与 tsp_ga.py 相同 seed，保证地图一致性

    coords = generate_random_coords(n_customers=n_customers, plane_size=plane_size, seed=seed)
    D = build_distance_matrix(coords)

    USE_HELD_KARP = True        # True=Held–Karp；False=暴力遍历
    t0 = time.perf_counter()
    if USE_HELD_KARP:
        res = held_karp_tsp(D)
        tag = "Held-Karp"
    else:
        res = brute_force_tsp(D)
        tag = "BruteForce"
    t1 = time.perf_counter()

    print("\n==== 最优结果（Exact）====")
    print(f"方法：{tag}")
    print(f"客户数：{n_customers}，用时：{t1 - t0:.3f}s")
    print(f"最优距离：{res['best_distance']:.4f}")
    print(f"最优访问顺序（不含仓库0）：{res['best_tour']}")

    # 画最优路线
    plot_route(coords, res["best_tour"], title=f"Exact TSP ({tag}) Best Route")


if __name__ == "__main__":
    main()
