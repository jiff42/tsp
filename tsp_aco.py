# -*- coding: utf-8 -*-
"""
TSP 蚁群算法（ACO）完整可运行示例
--------------------------------------------------
风格与 tsp_ga.py 保持一致：
1) 随机生成 TSP 节点（含仓库 0）；
2) 使用 ACO 求解最短回路（0 -> 访问全部客户 -> 0）；
3) 绘制收敛曲线与最优路线图；
4) 代码仅依赖 numpy、matplotlib。

运行：
    python tsp_aco.py
或在交互式环境中：
    %run tsp_aco.py
"""

import math
import random
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utlis import generate_random_coords, build_distance_matrix, tour_length
from utlis_vis import save_tsp_figs

# ==============================
# 蚁群算法（Ant System / Best-so-far variant）
# ==============================

def _choose_next(current: int, allowed: List[int], tau: np.ndarray, eta: np.ndarray,
                 alpha: float, beta: float, rng: random.Random) -> int:
    """按概率选择下一个城市：p ∝ tau^alpha * eta^beta"""
    if not allowed:
        return current
    weights = []
    for j in allowed:
        w = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
        weights.append(max(w, 1e-16))
    total = sum(weights)
    r = rng.random() * total
    cum = 0.0
    for j, w in zip(allowed, weights):
        cum += w
        if cum >= r:
            return j
    return allowed[-1]  # 数值稳定性兜底


def aco_tsp(
    D: np.ndarray,
    num_ants: int = 40,
    iterations: int = 400,
    alpha: float = 1.0,     # 信息素重要性
    beta: float = 5.0,      # 启发函数重要性 (1/d)
    rho: float = 0.5,       # 蒸发率
    Q: float = 100.0,       # 信息素释放强度
    use_best_so_far: bool = True,   # 仅用历史最优强化
    best_weight: float = 2.0,       # 历史最优强化权重
    seed: int = 123
) -> dict:
    """
    使用 ACO 求解 TSP。
    返回 dict: {best_tour, best_distance, history}
    """
    rng = random.Random(seed)
    n = D.shape[0]  # 含仓库
    n_customers = n - 1

    # 预计算启发函数 eta = 1 / d
    with np.errstate(divide='ignore'):
        eta = 1.0 / (D + 1e-12)
    np.fill_diagonal(eta, 0.0)

    # 初始化信息素
    tau0 = 1.0
    tau = np.full_like(D, tau0, dtype=float)
    np.fill_diagonal(tau, 0.0)

    # 记录最好
    best_tour: List[int] = []
    best_dist: float = float('inf')
    history: List[float] = []

    all_customers = list(range(1, n))

    for it in range(iterations):
        ant_tours: List[List[int]] = []
        ant_lengths: List[float] = []

        # ---- 构造每只蚂蚁的解
        for _ in range(num_ants):
            allowed = all_customers[:]
            # 随机选择起点（客户），与 depot 0 无关；回路最后会接回 0
            current = rng.choice(allowed)
            tour = [current]
            allowed.remove(current)

            while allowed:
                nxt = _choose_next(current, allowed, tau, eta, alpha, beta, rng)
                tour.append(nxt)
                allowed.remove(nxt)
                current = nxt

            L = tour_length(tour, D)
            ant_tours.append(tour)
            ant_lengths.append(L)

            if L + 1e-12 < best_dist:
                best_dist = L
                best_tour = tour[:]

        # ---- 信息素更新
        # 蒸发
        tau *= (1.0 - rho)
        # 所有蚂蚁释放
        for tour, L in zip(ant_tours, ant_lengths):
            deposit = Q / max(L, 1e-12)
            # 在回路的边上加信息素：0->tour[0], tour[i]->tour[i+1], tour[-1]->0
            prev = 0
            for c in tour:
                tau[prev, c] += deposit
                tau[c, prev] += deposit
                prev = c
            tau[prev, 0] += deposit
            tau[0, prev] += deposit

        # 历史最优强化（可选）
        if use_best_so_far and best_tour:
            deposit = best_weight * (Q / max(best_dist, 1e-12))
            prev = 0
            for c in best_tour:
                tau[prev, c] += deposit
                tau[c, prev] += deposit
                prev = c
            tau[prev, 0] += deposit
            tau[0, prev] += deposit

        history.append(best_dist)

        # 迭代日志（与 tsp_ga.py 相同风格）
        if (it + 1) % 100 == 0:
            print(f"[Iter {it+1:4d}] best = {best_dist:.4f}")

    return {"best_tour": best_tour, "best_distance": best_dist, "history": history}


# ==============================
# 主程序（可直接运行）
# ==============================


def main():
    # 1) 生成随机实例
    n_customers = 10
    plane_size = 500
    coords = generate_random_coords(n_customers=n_customers, plane_size=plane_size, seed=42)
    D = build_distance_matrix(coords)

    # 2) 运行 ACO
    res = aco_tsp(
        D,
        num_ants=50,
        iterations=500,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100.0,
        use_best_so_far=True,
        best_weight=2.0,
        seed=2025
    )

    print("\n==== 结果 ====")
    print("最优距离：", round(res["best_distance"], 4))
    print("最优访问顺序（不含仓库0）：", res["best_tour"])

    # 3) 绘图（保持与 GA 脚本一致的接口与风格）
    SAVE_DIR = "figs_tsp_aco1"
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_tsp_figs(history=res["history"], coords=coords, best_tour=res["best_tour"],
              algo_tag="ACO-TSP", save_dir=SAVE_DIR, conv_xlabel="Iteration",
              conv_name="aco_convergence.png", route_name="aco_best_route.png", dpi=150)


if __name__ == "__main__":
    main()
