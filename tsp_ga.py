# -*- coding: utf-8 -*-
"""
TSP 遗传算法（GA）完整可运行示例
--------------------------------------------------
功能：
1) 随机生成 TSP 节点（含仓库 0）；
2) 使用 GA 求解最短回路（0 -> 访问全部客户 -> 0）；
3) 绘制收敛曲线与最优路线图；
4) 代码仅依赖 numpy、matplotlib。

运行：
    python tsp_ga.py
或在交互式环境中：
    %run tsp_ga.py
"""

import time
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
# Use a non-interactive backend to ensure saving works without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os



from utlis_save_ins import load_tsp_instance
from utlis import generate_random_coords, build_distance_matrix, tour_length
from utlis_vis import save_tsp_figs

# ==============================
# 遗传算法核心算子（排列编码）
# ==============================

def init_population(pop_size: int, n_customers: int, rng: random.Random) -> List[List[int]]:
    """初始化种群：随机生成若干排列（客户访问顺序）"""
    base = list(range(1, n_customers + 1))
    pop = []
    for _ in range(pop_size):
        tour = base[:]
        rng.shuffle(tour)
        pop.append(tour)
    return pop


def tournament_select(pop: List[List[int]], fitness: List[float], k: int, rng: random.Random) -> List[int]:
    """
    锦标赛选择：随机挑 k 个个体，适应度更优者胜出。
    注意：这里 fitness 是“距离”，数值越小越好。
    """
    idxs = [rng.randrange(len(pop)) for _ in range(k)]
    best_idx = min(idxs, key=lambda i: fitness[i])
    return pop[best_idx][:]


def order_crossover_ox(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """
    OX（顺序交叉）：适用于排列编码，保留相对次序，不会重复城市。
    过程：随机选一段从 p1 复制到子代同位置；其余空位按 p2 的顺序填充未出现的元素。
    """
    n = len(p1)
    a, b = sorted([rng.randrange(n), rng.randrange(n)])
    c1 = [None] * n
    c2 = [None] * n
    # 复制片段
    c1[a:b+1] = p1[a:b+1]
    c2[a:b+1] = p2[a:b+1]

    def fill(child, donor):
        cur = (b + 1) % n
        for g in donor:
            if g not in child:
                child[cur] = g
                cur = (cur + 1) % n
        return child

    return fill(c1, p2), fill(c2, p1)


def mutation_swap(tour: List[int], rng: random.Random) -> None:
    """交换变异：随机交换两个位置"""
    n = len(tour)
    i, j = rng.randrange(n), rng.randrange(n)
    tour[i], tour[j] = tour[j], tour[i]


def mutation_inversion(tour: List[int], rng: random.Random) -> None:
    """倒置变异：随机选一段反序"""
    n = len(tour)
    i, j = sorted([rng.randrange(n), rng.randrange(n)])
    tour[i:j+1] = reversed(tour[i:j+1])


def two_opt(route: List[int], D: np.ndarray) -> List[int]:
    """
    2-opt 局部改进：将交叉边剪开、反转中段，若变短则接受。
    对小中规模很有效，但 O(n^2) 检查，适度使用。
    """
    best = route[:]
    best_len = tour_length(best, D)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_len = tour_length(new_route, D)
                if new_len + 1e-12 < best_len:
                    best, best_len = new_route, new_len
                    improved = True
    return best


# ==============================
# GA 主过程
# ==============================

def ga_tsp(
    D: np.ndarray,
    pop_size: int = 150,
    generations: int = 600,
    pc: float = 0.9,
    pm: float = 0.5,
    tournament_k: int = 3,
    elite_two_opt: bool = True,
    seed: int = 123
) -> dict:
    """
    使用 GA 求解 TSP。
    返回 dict: {best_tour, best_distance, history}
    """
    rng = random.Random(seed)
    n_customers = D.shape[0] - 1

    def fitness_fn(ind: List[int]) -> float:
        return tour_length(ind, D)

    # 初始化
    pop = init_population(pop_size, n_customers, rng)
    fitness = [fitness_fn(ind) for ind in pop]
    best_idx = int(np.argmin(fitness))
    best_tour = pop[best_idx][:]
    best_dist = fitness[best_idx]
    history = [best_dist]

    # 计时与计步
    t_start = time.perf_counter()
    E_budget = 0
    E_budget += len(pop)
    improve_log = [(0.0, E_budget, best_dist)]

    for gen in range(generations):
        new_pop: List[List[int]] = []
        # 精英保留 1 个
        elite = best_tour[:]
        if elite_two_opt:
            elite = two_opt(elite, D)
        new_pop.append(elite)

        # 产生子代，直到达到种群规模
        while len(new_pop) < pop_size:
            # 选择父代
            p1 = tournament_select(pop, fitness, tournament_k, rng)
            p2 = tournament_select(pop, fitness, tournament_k, rng)
            # 交叉
            if rng.random() < pc:
                c1, c2 = order_crossover_ox(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            # 变异
            if rng.random() < pm:
                (mutation_inversion if rng.random() < 0.5 else mutation_swap)(c1, rng)
            if rng.random() < pm:
                (mutation_inversion if rng.random() < 0.5 else mutation_swap)(c2, rng)
            new_pop.extend([c1, c2])

        # 保留前 pop_size 个
        pop = new_pop[:pop_size]
        fitness = [fitness_fn(ind) for ind in pop]
        E_budget += len(pop)

        # 更新最优
        cur_idx = int(np.argmin(fitness))
        if fitness[cur_idx] + 1e-12 < best_dist:
            best_dist = fitness[cur_idx]
            best_tour = pop[cur_idx][:]
            improve_log.append((time.perf_counter() - t_start, E_budget, best_dist))

        history.append(best_dist)

        # 可选：每隔若干代打印进度
        if (gen + 1) % 100 == 0:
            print(f"[Gen {gen+1:4d}] best = {best_dist:.4f}")

    return {"best_tour": best_tour, 
            "best_distance": best_dist, 
            "history": history,
            "time_sec": time.perf_counter() - t_start,
            "E_budget": E_budget,
            "improve_log": improve_log
            }


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

    # 2) 运行 GA
    res = ga_tsp(
        D,
        pop_size=250,
        generations=4000,
        pc=0.9,
        pm=0.5,
        tournament_k=10,
        elite_two_opt=False,
        seed=2025
    )


    print("\n==== 结果 ====")
    print("最优距离：", round(res["best_distance"], 4))
    print("最优访问顺序（不含仓库0）：", res["best_tour"])
    print(f"总时间: {res['time_sec']:.4f}s, 总E_budget: {res['E_budget']}")

    import csv, os
    os.makedirs("exp7", exist_ok=True)
    with open("exp7/improve_log_ga_N76_seed2025.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "E_gudget", "best_distance"])
        w.writerows(res["improve_log"])

    # # 3) 绘图（取消注释以显示图形）
    SAVE_DIR = "exp7"
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_tsp_figs(history=res["history"], coords=coords, best_tour=res["best_tour"],
              algo_tag="GA-TSP", save_dir=SAVE_DIR, conv_xlabel="Generation",
              conv_name="ga_convergence.png", route_name="ga_route.png", dpi=200)


if __name__ == "__main__":
    main()
