# -*- coding: utf-8 -*-
"""
TSP 模拟退火（SA）完整可运行示例
--------------------------------------------------
风格与 tsp_ga.py / tsp_aco.py 保持一致：
1) 随机生成 TSP 节点（含仓库 0）；
2) 使用 SA 求解最短回路（0 -> 访问全部客户 -> 0）；
3) 绘制收敛曲线与最优路线图；
4) 代码仅依赖 numpy、matplotlib。

运行：
    python tsp_sa.py
或在交互式环境中：
    %run tsp_sa.py
"""

import math
import os
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utlis import generate_random_coords, build_distance_matrix, tour_length
from utlis_vis import save_tsp_figs




# ==============================
# 模拟退火（SA）
# ==============================

def nearest_neighbor_tour(n_customers: int, D: np.ndarray, rng: random.Random) -> List[int]:
    """
    从随机客户出发的最近邻启发式，返回客户排列（不含 0）。
    """
    if n_customers <= 0:
        return []
    start = rng.randrange(1, n_customers + 1)
    unvisited = set(range(1, n_customers + 1))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def random_neighbor(tour: List[int], rng: random.Random) -> List[int]:
    """
    生成邻域解：50% 交换、50% 区段倒置。
    """
    n = len(tour)
    i, j = sorted([rng.randrange(n), rng.randrange(n)])
    if i == j:
        return tour[:]
    child = tour[:]
    if rng.random() < 0.5:
        # 交换变邻域
        child[i], child[j] = child[j], child[i]
    else:
        # 倒置变邻域（2-opt 风格）
        child[i:j+1] = reversed(child[i:j+1])
    return child


def sa_tsp(
    D: np.ndarray,
    iterations: int = 20000,
    T0: float = 100.0,
    cooling_rate: float = 0.999,   # 几何降温 T_k = T0 * cooling_rate^k
    moves_per_temp: int = 1,       # 每次温度下尝试的邻域次数（可设 >1）
    seed: int = 123,
    use_greedy_init: bool = True
) -> dict:
    """
    使用模拟退火求解 TSP。
    返回 dict: {best_tour, best_distance, history}
    """
    rng = random.Random(seed)
    n_customers = D.shape[0] - 1

    # 初始解
    if use_greedy_init:
        cur = nearest_neighbor_tour(n_customers, D, rng)
    else:
        cur = list(range(1, n_customers + 1))
        rng.shuffle(cur)

    cur_len = tour_length(cur, D)
    best = cur[:]
    best_len = cur_len

    history = [best_len]
    T = T0

    # 主循环
    for it in range(iterations):
        for _ in range(moves_per_temp):
            cand = random_neighbor(cur, rng)
            cand_len = tour_length(cand, D)
            delta = cand_len - cur_len
            if delta < 0:
                # 更好，必收
                cur, cur_len = cand, cand_len
            else:
                # 更差，以概率接受：exp(-delta / T)
                accept_prob = math.exp(-delta / max(T, 1e-12))
                if rng.random() < accept_prob:
                    cur, cur_len = cand, cand_len

            if cur_len + 1e-12 < best_len:
                best, best_len = cur[:], cur_len

        # 降温（几何衰减）
        T *= cooling_rate
        history.append(best_len)

        # 进度打印（与 GA/ACO 风格一致）
        if (it + 1) % 100 == 0:
            print(f"[Iter {it+1:6d}] T={T:.4f}, best = {best_len:.4f}")

    return {"best_tour": best, "best_distance": best_len, "history": history}


# ==============================
# 主程序（可直接运行）
# ==============================

def main():
    # 1) 生成随机实例
    n_customers = 20
    plane_size = 500
    coords = generate_random_coords(n_customers=n_customers, plane_size=plane_size, seed=42)
    D = build_distance_matrix(coords)

    # 2) 运行 SA
    res = sa_tsp(
        D,
        iterations=25000,
        T0=200.0,
        cooling_rate=0.9993,
        moves_per_temp=1,
        seed=2025,
        use_greedy_init=True
    )

    print("\n==== 结果 ====")
    print("最优距离：", round(res["best_distance"], 4))
    print("最优访问顺序（不含仓库0）：", res["best_tour"])

    # 3) 绘图改为保存为文件到目录 figs_tsp_sa（不显示图像）
    out_dir = "figs_tsp_sa1"
    os.makedirs(out_dir, exist_ok=True)
    save_tsp_figs(history=res["history"], coords=coords, best_tour=res["best_tour"],
              algo_tag="SA-TSP", save_dir=out_dir, conv_xlabel="Iteration",
              conv_name="sa_convergence.png", route_name="sa_route.png", dpi=150)



if __name__ == "__main__":
    main()
