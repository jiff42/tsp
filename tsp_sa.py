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

import time
import math
import os
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utlis_save_ins import load_tsp_instance
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

    # === 计时与计步（E） ===
    t_start = time.perf_counter()
    E_budget = 0
    E_budget += 1  # 计算了 cur_len 一次
    improve_log = [(0.0, E_budget, best_len)]   

    # 主循环
    for it in range(iterations):
        for _ in range(moves_per_temp):
            cand = random_neighbor(cur, rng)
            cand_len = tour_length(cand, D)
            delta = cand_len - cur_len
            E_budget += 1  # 计算了 cand_len 一次

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
                improve_log.append((time.perf_counter() - t_start, E_budget, best_len))

        # 降温（几何衰减）
        T *= cooling_rate
        history.append(best_len)

        # 进度打印（与 GA/ACO 风格一致）
        if (it + 1) % 100 == 0:
            print(f"[Iter {it+1:6d}] T={T:.4f}, best = {best_len:.4f}")

    return {"best_tour": best, 
            "best_distance": best_len, 
            "history": history,
            "improve_log": improve_log,
            "E_budget": E_budget,
            "time_sec": time.perf_counter() - t_start}


# ==============================
# 主程序（可直接运行）
# ==============================

def main():
    use_benchmark = True

    if use_benchmark == False:
        # 1) 生成随机实例
        n_customers = 40        # 客户数量（可改）
        plane_size = 200         # 坐标范围 0..plane_size
        coords = generate_random_coords(n_customers=n_customers, plane_size=plane_size, seed=42)
        D = build_distance_matrix(coords)

        # 1.5) 检查生成的坐标是否一致
        # 读取随机生成的实例
        json_path = "exp3/tsp_instance_seed42_N40.json"
        coords_loaded = load_tsp_instance(json_path)
        # 检查加载的坐标是否与原始坐标一致
        print(f"coords == coords_loaded: {coords == coords_loaded}")
    elif use_benchmark == True:
        # 1. 读取 TSP 实例
        print("loading eil76.tsp")
        json_path = "exp7/tsp_instance_seed0_N76.json"
        coords = load_tsp_instance(json_path)
        D = build_distance_matrix(coords)

    # 2) 运行 SA
    res = sa_tsp(
        D,
        iterations=10000,
        T0=500.0,
        cooling_rate=0.99,
        moves_per_temp=100,
        seed=2025,
        use_greedy_init=False
    )

    print("\n==== 结果 ====")
    print("最优距离：", round(res["best_distance"], 4))
    print("最优访问顺序（不含仓库0）：", res["best_tour"])

    print(f"总时间: {res['time_sec']:.4f}s, 总E_budget: {res['E_budget']}")

    import csv, os
    os.makedirs("exp7", exist_ok=True)
    with open("exp7/improve_log_sa_N76_seed2025.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "E_budget", "best_distance"])
        w.writerows(res["improve_log"])

    # 3) 绘图改为保存为文件到目录 figs_tsp_sa（不显示图像）
    out_dir = "exp7"
    os.makedirs(out_dir, exist_ok=True)
    save_tsp_figs(history=res["history"], coords=coords, best_tour=res["best_tour"],
              algo_tag="SA-TSP", save_dir=out_dir, conv_xlabel="Iteration",
              conv_name="sa_convergence.png", route_name="sa_route.png", dpi=150)



if __name__ == "__main__":
    main()
