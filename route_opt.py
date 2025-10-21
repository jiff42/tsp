# -*- coding: utf-8 -*-
"""
路径规划与物流调度 - GA/NSGA-II 参考实现
====================================================
功能包含：
1) TSP (单车) 的 GA 求解；
2) CVRP (多车) 的 GA 求解（巨型序列 + 贪心 Split 解码 + 约束惩罚）；
3) VRP 的 NSGA-II 多目标扩展（f1=总里程, f2=最大路线里程），并保留约束惩罚；
4) 基础图形可视化（路线图/收敛曲线/帕累托前沿）。

依赖：numpy, matplotlib
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: disable GUI
import matplotlib.pyplot as plt
plt.ioff()  # turn off interactive mode so nothing is shown


# ------------------------------
# 数据结构与工具
# ------------------------------

@dataclass
class Customer:
    idx: int
    x: float
    y: float
    demand: int = 0
    ready_time: float = 0.0  # 可选：时间窗开始
    due_time: float = float('inf')  # 可选：时间窗结束
    service_time: float = 0.0  # 可选：服务时长


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def tour_length_tsp(tour: List[int], D: np.ndarray) -> float:
    """tour 仅包含客户(1..N)，0 是仓库。"""
    if not tour:
        return 0.0
    length = D[0, tour[0]]
    for i in range(len(tour) - 1):
        length += D[tour[i], tour[i+1]]
    length += D[tour[-1], 0]
    return length


def route_length(route: List[int], D: np.ndarray) -> float:
    """单条车路线：0 -> route -> 0"""
    if not route:
        return 0.0
    length = D[0, route[0]]
    for i in range(len(route) - 1):
        length += D[route[i], route[i+1]]
    length += D[route[-1], 0]
    return length


def two_opt(route: List[int], D: np.ndarray) -> List[int]:
    """对单条路线做 2-opt 改善（小规模足够）。"""
    improved = True
    best = route[:]
    best_len = route_length(best, D)
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_len = route_length(new_route, D)
                if new_len + 1e-9 < best_len:
                    best, best_len = new_route, new_len
                    improved = True
        # 直到没有改进为止
    return best


# ------------------------------
# 随机实例生成
# ------------------------------

def generate_random_instance(
    n_customers: int = 40,
    plane_size: int = 200,
    demand_range: Tuple[int, int] = (1, 10),
    seed: int = 42
) -> Tuple[List[Customer], np.ndarray]:
    """随机生成：仓库(0) 与 n_customers 个客户(1..N)"""
    rng = random.Random(seed)
    customers = []
    # 仓库
    depot = Customer(idx=0, x=plane_size/2, y=plane_size/2, demand=0)
    customers.append(depot)
    for i in range(1, n_customers + 1):
        x = rng.uniform(0, plane_size)
        y = rng.uniform(0, plane_size)
        d = rng.randint(demand_range[0], demand_range[1])
        customers.append(Customer(idx=i, x=x, y=y, demand=d))
    coords = [(c.x, c.y) for c in customers]
    D = build_distance_matrix(coords)
    return customers, D


# ------------------------------
# 遗传算法通用算子（排列编码）
# ------------------------------

def init_population_permutations(pop_size: int, n_customers: int, rng: random.Random) -> List[List[int]]:
    pop = []
    base = list(range(1, n_customers + 1))
    for _ in range(pop_size):
        tour = base[:]
        rng.shuffle(tour)
        pop.append(tour)
    return pop


def tournament_select(pop: List[List[int]], fitness: List[float], k: int, rng: random.Random) -> List[int]:
    idxs = [rng.randrange(len(pop)) for _ in range(k)]
    best_idx = min(idxs, key=lambda i: fitness[i])
    return pop[best_idx][:]


def order_crossover_ox(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted([rng.randrange(n), rng.randrange(n)])
    c1 = [None] * n
    c2 = [None] * n
    c1[a:b+1] = p1[a:b+1]
    c2[a:b+1] = p2[a:b+1]
    fill = lambda child, donor: _ox_fill(child, donor, a, b)
    return fill(c1, p2), fill(c2, p1)


def _ox_fill(child: List[Optional[int]], donor: List[int], a: int, b: int) -> List[int]:
    n = len(child)
    cur = (b + 1) % n
    for g in donor:
        if g not in child:
            child[cur] = g
            cur = (cur + 1) % n
    return child


def mutation_swap(tour: List[int], rng: random.Random) -> None:
    i, j = rng.randrange(len(tour)), rng.randrange(len(tour))
    tour[i], tour[j] = tour[j], tour[i]


def mutation_inversion(tour: List[int], rng: random.Random) -> None:
    i, j = sorted([rng.randrange(len(tour)), rng.randrange(len(tour))])
    tour[i:j+1] = reversed(tour[i:j+1])


# ------------------------------
# 1) TSP - GA
# ------------------------------

def ga_tsp(
    D: np.ndarray,
    pop_size: int = 150,
    generations: int = 500,
    pc: float = 0.9,
    pm: float = 0.2,
    tournament_k: int = 3,
    use_two_opt_elite: bool = True,
    seed: int = 123
) -> Dict:
    rng = random.Random(seed)
    n_customers = D.shape[0] - 1
    pop = init_population_permutations(pop_size, n_customers, rng)

    def fitness_fn(tour):
        return tour_length_tsp(tour, D)

    fitness = [fitness_fn(ind) for ind in pop]
    best_idx = int(np.argmin(fitness))
    best_tour = pop[best_idx][:]
    best_dist = fitness[best_idx]
    history = [best_dist]

    for gen in range(generations):
        new_pop = []
        # 精英保留 1 个
        elite = best_tour[:]
        if use_two_opt_elite:
            elite = two_opt(elite, D)
        new_pop.append(elite)

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, tournament_k, rng)
            p2 = tournament_select(pop, fitness, tournament_k, rng)
            if rng.random() < pc:
                c1, c2 = order_crossover_ox(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            # 变异
            if rng.random() < pm:
                mutation_inversion(c1, rng) if rng.random() < 0.5 else mutation_swap(c1, rng)
            if rng.random() < pm:
                mutation_inversion(c2, rng) if rng.random() < 0.5 else mutation_swap(c2, rng)
            new_pop.extend([c1, c2])

        pop = new_pop[:pop_size]
        fitness = [fitness_fn(ind) for ind in pop]
        cur_idx = int(np.argmin(fitness))
        if fitness[cur_idx] + 1e-9 < best_dist:
            best_dist = fitness[cur_idx]
            best_tour = pop[cur_idx][:]
        history.append(best_dist)

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history
    }


# ------------------------------
# 2) CVRP - GA (Split 解码 + 约束惩罚)
# ------------------------------

@dataclass
class VRPConfig:
    Q: int                      # 车辆容量
    K_max: Optional[int] = None # 车辆数上限（可选）
    D_max: Optional[float] = None  # 单车最大路线里程（可选）
    penalty_lambda: float = 1000.0  # 违约惩罚系数


def decode_split_routes(
    perm: List[int],
    demands: Dict[int, int],
    D: np.ndarray,
    cfg: VRPConfig
) -> Tuple[List[List[int]], float, Dict[str, float]]:
    """贪心切割：按容量/里程上限拆分，计算总里程与约束违反。"""
    routes: List[List[int]] = []
    cur_route: List[int] = []
    cur_load = 0
    for c in perm:
        d = demands[c]
        # 预估若加入后的负载与里程
        # 里程检查：简单估计，若加入则当前路线增加的边界变化（启发式）
        will_exceed_load = (cur_load + d > cfg.Q)
        if cfg.D_max is not None and cur_route:
            # 粗略估算加入后的路线长度（先加入，再算临时长度）
            tmp = cur_route + [c]
            tmp_len = route_length(tmp, D)
            will_exceed_dist = (tmp_len > cfg.D_max)
        else:
            will_exceed_dist = False

        if (cur_route and (will_exceed_load or will_exceed_dist)):
            routes.append(cur_route)
            cur_route = [c]
            cur_load = d
        else:
            cur_route.append(c)
            cur_load += d
    if cur_route:
        routes.append(cur_route)

    total_dist = sum(route_length(r, D) for r in routes)

    # 约束违反统计（尽量通过切割避免，但仍计算安全检查）
    cv = 0.0
    # 容量：若某路线超载则计惩罚
    for r in routes:
        load = sum(demands[i] for i in r)
        if load > cfg.Q:
            cv += (load - cfg.Q)

    # 车辆数上限违反
    if (cfg.K_max is not None) and (len(routes) > cfg.K_max):
        cv += (len(routes) - cfg.K_max) * 10.0  # 车辆数违反权重

    # 单车路线长度上限违反
    if cfg.D_max is not None:
        for r in routes:
            L = route_length(r, D)
            if L > cfg.D_max:
                cv += (L - cfg.D_max) / max(1.0, cfg.D_max)

    obj = total_dist + cfg.penalty_lambda * cv
    stats = {
        "num_routes": len(routes),
        "total_distance": total_dist,
        "constraint_violation": cv
    }
    return routes, obj, stats


def improve_routes_two_opt(routes: List[List[int]], D: np.ndarray) -> List[List[int]]:
    improved = []
    for r in routes:
        improved.append(two_opt(r, D) if len(r) >= 4 else r[:])
    return improved


def ga_cvrp(
    D: np.ndarray,
    demands: Dict[int, int],
    cfg: VRPConfig,
    pop_size: int = 150,
    generations: int = 400,
    pc: float = 0.9,
    pm: float = 0.25,
    tournament_k: int = 3,
    use_two_opt_after_decode: bool = True,
    seed: int = 123
) -> Dict:
    rng = random.Random(seed)
    n_customers = D.shape[0] - 1
    pop = init_population_permutations(pop_size, n_customers, rng)

    def fitness_fn(ind):
        routes, obj, _ = decode_split_routes(ind, demands, D, cfg)
        return obj

    fitness = [fitness_fn(ind) for ind in pop]
    best_idx = int(np.argmin(fitness))
    best_perm = pop[best_idx][:]
    best_routes, best_obj, best_stats = decode_split_routes(best_perm, demands, D, cfg)
    if use_two_opt_after_decode:
        best_routes = improve_routes_two_opt(best_routes, D)
        # 重新计算
        tmp_perm = sum(best_routes, [])
        best_routes, best_obj, best_stats = decode_split_routes(tmp_perm, demands, D, cfg)

    history = [best_obj]

    for gen in range(generations):
        new_pop = []
        # 精英保留（以最佳个体的“巨型序列”形式）
        new_pop.append(best_perm[:])

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fitness, tournament_k, rng)
            p2 = tournament_select(pop, fitness, tournament_k, rng)
            if rng.random() < pc:
                c1, c2 = order_crossover_ox(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            if rng.random() < pm:
                mutation_inversion(c1, rng) if rng.random() < 0.5 else mutation_swap(c1, rng)
            if rng.random() < pm:
                mutation_inversion(c2, rng) if rng.random() < 0.5 else mutation_swap(c2, rng)
            new_pop.extend([c1, c2])

        pop = new_pop[:pop_size]
        fitness = [fitness_fn(ind) for ind in pop]
        cur_idx = int(np.argmin(fitness))
        if fitness[cur_idx] + 1e-9 < best_obj:
            best_obj = fitness[cur_idx]
            best_perm = pop[cur_idx][:]
            best_routes, _, best_stats = decode_split_routes(best_perm, demands, D, cfg)
            if use_two_opt_after_decode:
                best_routes = improve_routes_two_opt(best_routes, D)
                tmp_perm = sum(best_routes, [])
                best_routes, best_obj, best_stats = decode_split_routes(tmp_perm, demands, D, cfg)
        history.append(best_obj)

    return {
        "best_perm": best_perm,
        "best_routes": best_routes,
        "best_objective": best_obj,
        "best_stats": best_stats,
        "history": history
    }


# ------------------------------
# 3) NSGA-II 多目标 VRP
#    f1 = 总里程（越小越好）
#    f2 = 最大单车路线里程（越小越好）
#    约束通过惩罚影响（可在非支配比较中作为次序因素）
# ------------------------------

def evaluate_vrp_objectives(
    perm: List[int],
    demands: Dict[int, int],
    D: np.ndarray,
    cfg: VRPConfig
) -> Tuple[List[List[int]], float, float, float]:
    routes, obj, stats = decode_split_routes(perm, demands, D, cfg)
    total_dist = stats["total_distance"]
    max_route = max((route_length(r, D) for r in routes), default=0.0)
    cv = stats["constraint_violation"]
    return routes, total_dist, max_route, cv


def fast_non_dominated_sort(objs: List[Tuple[float, float, float]]) -> List[List[int]]:
    """
    objs[i] = (f1, f2, cv)；cv 为约束违反量。
    支配关系：优先比较约束：cv 小者更优；cv 都为 0 时，按 f1,f2 非支配排序。
    """
    n = len(objs)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    def dominates(a, b):
        f1a, f2a, cva = objs[a]
        f1b, f2b, cvb = objs[b]
        # 约束优先：若 a 违反少于 b，则 a 支配 b
        if cva < cvb:
            return True
        if cva > cvb:
            return False
        # 都可行或违反相同，则用 f1,f2 非支配比较
        no_worse = (f1a <= f1b + 1e-12) and (f2a <= f2b + 1e-12)
        strictly_better = (f1a < f1b - 1e-12) or (f2a < f2b - 1e-12)
        return no_worse and strictly_better

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(front: List[int], objs: List[Tuple[float, float, float]]) -> Dict[int, float]:
    if not front:
        return {}
    distances = {i: 0.0 for i in front}
    for m in range(2):  # 只对 f1,f2 计算拥挤度（cv 已在排序优先考虑）
        front_sorted = sorted(front, key=lambda i: objs[i][m])
        distances[front_sorted[0]] = float('inf')
        distances[front_sorted[-1]] = float('inf')
        m_min = objs[front_sorted[0]][m]
        m_max = objs[front_sorted[-1]][m]
        denom = (m_max - m_min) if m_max > m_min else 1.0
        for k in range(1, len(front_sorted)-1):
            i_prev = front_sorted[k-1]
            i_next = front_sorted[k+1]
            distances[front_sorted[k]] += (objs[i_next][m] - objs[i_prev][m]) / denom
    return distances


def nsga2_vrp(
    D: np.ndarray,
    demands: Dict[int, int],
    cfg: VRPConfig,
    pop_size: int = 160,
    generations: int = 300,
    pc: float = 0.9,
    pm: float = 0.25,
    tournament_k: int = 3,
    seed: int = 2024
) -> Dict:
    rng = random.Random(seed)
    n_customers = D.shape[0] - 1
    pop = init_population_permutations(pop_size, n_customers, rng)

    def evaluate_population(popu: List[List[int]]) -> Tuple[List[Tuple[float, float, float]], List[List[List[int]]]]:
        objs = []
        decoded_routes = []
        for ind in popu:
            routes, f1, f2, cv = evaluate_vrp_objectives(ind, demands, D, cfg)
            objs.append((f1, f2, cv))
            decoded_routes.append(routes)
        return objs, decoded_routes

    objs, decoded = evaluate_population(pop)

    for gen in range(generations):
        # ---- 选择（基于等级与拥挤度）
        fronts = fast_non_dominated_sort(objs)
        distances = {}
        for fr in fronts:
            distances.update(crowding_distance(fr, objs))

        def better(i, j):
            # 按前沿等级 + 拥挤度排序
            rank_i = next(r for r, fr in enumerate(fronts) if i in fr)
            rank_j = next(r for r, fr in enumerate(fronts) if j in fr)
            if rank_i != rank_j:
                return rank_i < rank_j
            return distances.get(i, 0.0) > distances.get(j, 0.0)

        # 锦标赛
        parents = []
        for _ in range(pop_size):
            i, j = rng.randrange(pop_size), rng.randrange(pop_size)
            parents.append(pop[i][:] if better(i, j) else pop[j][:])

        # ---- 交叉变异
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i+1) % pop_size]
            if rng.random() < pc:
                c1, c2 = order_crossover_ox(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            if rng.random() < pm:
                mutation_inversion(c1, rng) if rng.random() < 0.5 else mutation_swap(c1, rng)
            if rng.random() < pm:
                mutation_inversion(c2, rng) if rng.random() < 0.5 else mutation_swap(c2, rng)
            offspring.extend([c1, c2])

        # ---- 合并、再排序取前 P
        union = pop + offspring
        objs_u, dec_u = evaluate_population(union)
        fronts_u = fast_non_dominated_sort(objs_u)

        new_pop = []
        new_objs = []
        for fr in fronts_u:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend([union[i] for i in fr])
                new_objs.extend([objs_u[i] for i in fr])
            else:
                cd = crowding_distance(fr, objs_u)
                fr_sorted = sorted(fr, key=lambda i: cd.get(i, 0.0), reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend([union[i] for i in fr_sorted[:needed]])
                new_objs.extend([objs_u[i] for i in fr_sorted[:needed]])
                break
        pop, objs = new_pop, new_objs

    # 最终前沿
    fronts_final = fast_non_dominated_sort(objs)
    F1_idx = fronts_final[0]
    pareto = [{"perm": pop[i], "f1_total_dist": objs[i][0], "f2_max_route": objs[i][1], "cv": objs[i][2]} for i in F1_idx]
    return {
        "pareto": pareto,
        "objs": objs,
        "population": pop,
        "fronts": fronts_final
    }


# ------------------------------
# 可视化与实验脚手架

def _slugify_title(title: str) -> str:
    """Convert a plot title to a safe filename stem (ascii-ish, underscores)."""
    import re
    s = re.sub(r'[^a-zA-Z0-9]+', '_', title.strip().lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'figure'


def _savefig_by_title(title: str, dpi: int = 200):
    """Save current figure into figs/<slug(title)>.png and close the figure."""
    import os
    os.makedirs('figs', exist_ok=True)
    fname = f"{_slugify_title(title)}.png"
    path = os.path.join('figs', fname)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
# ------------------------------

def plot_tsp_route(customers: List[Customer], tour: List[int], D: np.ndarray, title: str = "TSP Route"):
    xs = [customers[0].x] + [customers[i].x for i in tour] + [customers[0].x]
    ys = [customers[0].y] + [customers[i].y for i in tour] + [customers[0].y]
    plt.figure()
    plt.scatter([c.x for c in customers], [c.y for c in customers])
    plt.plot(xs, ys)
    for c in customers:
        plt.text(c.x, c.y, str(c.idx))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    _savefig_by_title(title)


def plot_vrp_routes(customers: List[Customer], routes: List[List[int]], title: str = "VRP Routes"):
    plt.figure()
    xs_all = [c.x for c in customers]
    ys_all = [c.y for c in customers]
    plt.scatter(xs_all, ys_all)
    plt.text(customers[0].x, customers[0].y, "Depot(0)")
    for r in routes:
        xs = [customers[0].x] + [customers[i].x for i in r] + [customers[0].x]
        ys = [customers[0].y] + [customers[i].y for i in r] + [customers[0].y]
        plt.plot(xs, ys)
        for i in r:
            plt.text(customers[i].x, customers[i].y, str(i))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    _savefig_by_title(title)


def plot_history(history: List[float], title: str = "Convergence"):
    plt.figure()
    plt.plot(list(range(len(history))), history)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Objective/Distance")
    plt.grid(True)
    _savefig_by_title(title)


def plot_pareto(pareto: List[Dict], title: str = "Pareto Front (f1 vs f2)"):
    f1 = [p["f1_total_dist"] for p in pareto]
    f2 = [p["f2_max_route"] for p in pareto]
    plt.figure()
    plt.scatter(f1, f2)
    plt.title(title)
    plt.xlabel("Total Distance (min)")
    plt.ylabel("Max Route Distance (min)")
    plt.grid(True)
    _savefig_by_title(title)


# ------------------------------
# 时间窗（可选拓展）惩罚演示（简化）
# ------------------------------

def time_window_penalty(
    routes: List[List[int]],
    customers: List[Customer],
    D: np.ndarray,
    speed: float = 1.0
) -> float:
    """
    简化的时间窗惩罚：若到达时间超过 due_time 则计线性惩罚。
    假设服务在到达后立即开始，服务时长 service_time。
    """
    penalty = 0.0
    for r in routes:
        t = 0.0  # 从仓库 0 出发时间 0
        prev = 0
        for c in r:
            travel = D[prev, c] / max(1e-6, speed)
            t += travel
            # 等待早到 (ready_time) 可不罚；迟到 (t > due_time) 罚
            if t > customers[c].due_time:
                penalty += (t - customers[c].due_time)
            # 服务耗时
            t = max(t, customers[c].ready_time) + customers[c].service_time
            prev = c
        # 返回仓库不计窗罚
    return penalty


# ------------------------------
# 示例 main（可直接运行）
# ------------------------------

if __name__ == "__main__":
    # 生成随机实例
    customers, D = generate_random_instance(n_customers=30, plane_size=100, demand_range=(1, 10), seed=7)
    demands = {c.idx: c.demand for c in customers}

    # ========== 1) 单车 TSP ==========
    print(">> 运行 GA-TSP ...")
    tsp_res = ga_tsp(D, pop_size=120, generations=400, pc=0.9, pm=0.2, seed=1)
    print(f"TSP 最优距离: {tsp_res['best_distance']:.2f}")
    # 绘图
    plot_tsp_route(customers, tsp_res["best_tour"], D, title="Best TSP Route")
    plot_history(tsp_res["history"], title="TSP Convergence")

    # ========== 2) 多车 CVRP ==========
    print("\n>> 运行 GA-CVRP ...")
    cfg = VRPConfig(Q=30, K_max=4, D_max=None, penalty_lambda=500.0)
    cvrp_res = ga_cvrp(D, demands, cfg, pop_size=150, generations=350, pc=0.9, pm=0.25, seed=2)
    print(f"CVRP 最优目标(含惩罚): {cvrp_res['best_objective']:.2f}, "
          f"总距: {cvrp_res['best_stats']['total_distance']:.2f}, "
          f"车辆数: {cvrp_res['best_stats']['num_routes']}, "
          f"违反量: {cvrp_res['best_stats']['constraint_violation']:.2f}")
    # 绘图
    plot_vrp_routes(customers, cvrp_res["best_routes"], title="Best CVRP Routes")
    plot_history(cvrp_res["history"], title="CVRP Convergence")

    # ========== 3) 多目标 NSGA-II ==========
    print("\n>> 运行 NSGA-II VRP (f1:总距, f2:最大路线距) ...")
    cfg_mo = VRPConfig(Q=30, K_max=5, D_max=None, penalty_lambda=800.0)
    nsga_res = nsga2_vrp(D, demands, cfg_mo, pop_size=160, generations=250, pc=0.9, pm=0.25, seed=3)
    pareto = nsga_res["pareto"]
    pareto_sorted = sorted(pareto, key=lambda p: (p["cv"], p["f1_total_dist"]))
    print(f"Pareto 前沿解数量: {len(pareto_sorted)}")
    if pareto_sorted:
        print(f"例: f1={pareto_sorted[0]['f1_total_dist']:.2f}, f2={pareto_sorted[0]['f2_max_route']:.2f}, "
              f"cv={pareto_sorted[0]['cv']:.2f}")
    # 绘图
    plot_pareto(pareto_sorted, title="NSGA-II Pareto Front")

    print("\n完成。可解除注释绘图函数以生成图像并插入报告。")
