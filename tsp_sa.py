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


# ==============================
# 工具函数
# ==============================

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """欧氏距离"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """根据坐标构建距离矩阵 D[i,j]"""
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def tour_length(tour: List[int], D: np.ndarray) -> float:
    """
    计算 TSP 回路长度。
    tour 仅包含客户节点 [1..N]（不含仓库 0）。
    总长度 = 0->tour[0] + sum(tour[i]->tour[i+1]) + tour[-1]->0
    """
    if not tour:
        return 0.0
    length = D[0, tour[0]]
    for i in range(len(tour) - 1):
        length += D[tour[i], tour[i + 1]]
    length += D[tour[-1], 0]
    return length


# ==============================
# 可视化
# ==============================

def plot_convergence(history: List[float], title: str = "SA Convergence", out_path: str = "sa_convergence.png"):
    """Plot convergence curve and save to file instead of displaying.

    out_path: output filename, defaults to 'sa_convergence.png'.
    """
    plt.figure()
    plt.plot(range(len(history)), history)
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.title(title)
    plt.grid(True)
    # Save figure instead of showing on screen
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_route(
    coords: List[Tuple[float, float]],
    tour: List[int],
    title: str = "Best Route",
    out_path: str = "sa_route.png",
):
    """根据坐标与 best_tour 画路线图"""
    xs = [coords[0][0]] + [coords[i][0] for i in tour] + [coords[0][0]]
    ys = [coords[0][1]] + [coords[i][1] for i in tour] + [coords[0][1]]
    plt.figure()
    plt.scatter([c[0] for c in coords], [c[1] for c in coords])
    plt.plot(xs, ys)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    # Save figure instead of showing on screen
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# ==============================
# 数据生成（随机 TSP 实例）
# ==============================

def generate_random_coords(n_customers: int = 30, plane_size: int = 100, seed: int = 7) -> List[Tuple[float, float]]:
    """
    生成坐标列表（含仓库 0）
    仓库放在平面中心，其余客户随机散布。
    """
    rng = random.Random(seed)
    coords = [(plane_size / 2.0, plane_size / 2.0)]  # depot at center
    for _ in range(n_customers):
        x = rng.uniform(0, plane_size)
        y = rng.uniform(0, plane_size)
        coords.append((x, y))
    return coords


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
    n_customers = 40
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
    out_dir = "figs_tsp_sa"
    os.makedirs(out_dir, exist_ok=True)
    plot_convergence(
        res["history"], title="SA-TSP Convergence",
        out_path=os.path.join(out_dir, "sa_convergence.png")
    )
    plot_route(
        coords, res["best_tour"], title="SA-TSP Best Route",
        out_path=os.path.join(out_dir, "sa_route.png")
    )


if __name__ == "__main__":
    main()
