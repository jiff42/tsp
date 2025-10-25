# -*- coding: utf-8 -*-
"""
读取 TSPLIB 的 a280.tsp 与 a280.opt.tour，并按现有工程的接口返回/绘图
用法：
    python load_tsplib_a280.py
依赖：
    utlis.py      -> build_distance_matrix, tour_length
    utlis_vis.py  -> save_tsp_figs
与现有 GA/SA/ACO 风格一致：生成 coords、D、best_tour、best_distance，并保存图像。
"""

import os
from typing import List, Tuple, Dict

import numpy as np

from utlis import build_distance_matrix, tour_length
from utlis_vis import save_tsp_figs


def parse_tsplib_tsp(path_tsp: str) -> Dict[int, Tuple[float, float]]:
    """
    解析 TSPLIB .tsp（仅支持 NODE_COORD_SECTION 形式）
    返回：{node_id: (x, y)}，node_id 从 1..N
    """
    coords: Dict[int, Tuple[float, float]] = {}
    with open(path_tsp, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    # 找到 NODE_COORD_SECTION
    try:
        start = lines.index("NODE_COORD_SECTION") + 1
    except ValueError:
        # 有些文件写成 "NODE_COORD_SECTION " 或大小写不同，做个兜底
        start = None
        for i, ln in enumerate(lines):
            if ln.upper().startswith("NODE_COORD_SECTION"):
                start = i + 1
                break
        if start is None:
            raise ValueError("未找到 NODE_COORD_SECTION")

    # 逐行读取直到 EOF
    for ln in lines[start:]:
        if not ln or ln.upper().startswith("EOF"):
            break
        parts = ln.split()
        if len(parts) < 3:
            continue
        idx = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        coords[idx] = (x, y)

    if not coords:
        raise ValueError("未在 .tsp 中解析到坐标")
    return coords


def parse_tsplib_tour(path_tour: str) -> List[int]:
    """
    解析 TSPLIB .tour（TOUR_SECTION 后的一串节点，-1 结束）
    返回：完整回路顺序（长度 N），节点编号为 1..N
    """
    seq: List[int] = []
    with open(path_tour, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    # 找 TOUR_SECTION
    start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("TOUR_SECTION"):
            start = i + 1
            break
    if start is None:
        raise ValueError("未找到 TOUR_SECTION")

    for ln in lines[start:]:
        if not ln:
            continue
        if ln.strip() == "-1" or ln.upper().startswith("EOF"):
            break
        seq.append(int(float(ln.strip())))

    if not seq:
        raise ValueError("未在 .tour 中解析到路径顺序")
    return seq


def tsplib_to_workspace_format(tsplib_coords: Dict[int, Tuple[float, float]],
                               tsplib_tour: List[int]) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    将 TSPLIB 的回路转换为你工程里的格式：
    - coords: 下标 0 为“仓库”，其余 1..N 为客户坐标（沿用 TSPLIB 的编号）
      这里把 0 号仓库坐标设置为 TSPLIB 路线的首城坐标，以复用 tour_length 中的 0->first 和 last->0 逻辑。
    - best_tour: 客户访问序列（不含 0），取 TSPLIB 路线去掉首城后的顺序。
      这样：
        0 -> best_tour[0]   等价于   TSPLIB 首城 -> 第二个城市
        best_tour[-1] -> 0  等价于   TSPLIB 最后一个城市 -> 首城
    """
    N = len(tsplib_coords)
    assert N == len(tsplib_tour), "坐标数与路线长度不一致"

    first_city = tsplib_tour[0]
    depot_xy = tsplib_coords[first_city]

    # coords[0] 放仓库，其余沿用 1..N 的原编号
    # 注意：coords 的长度需要是 N+1，且索引位置与 build_distance_matrix 的访问一致
    coords: List[Tuple[float, float]] = [depot_xy] + [None] * N  # type: ignore
    for i in range(1, N + 1):
        coords[i] = tsplib_coords[i]

    # best_tour 去掉首城（因为仓库已经占了它的位置）
    best_tour = tsplib_tour[1:]  # 长度 N-1，元素范围 1..N
    return coords, best_tour


def main():
    # 文件路径（与脚本同目录或自行修改为绝对路径）
    tsp_path = "tsp_lib/eil76.tsp"
    tour_path = "tsp_lib/eil76.opt.tour"

    # 1) 读取 TSPLIB
    tsplib_coords = parse_tsplib_tsp(tsp_path)
    tsplib_tour = parse_tsplib_tour(tour_path)

    # 2) 转换为现有工程的 coords / best_tour 格式
    coords, best_tour = tsplib_to_workspace_format(tsplib_coords, tsplib_tour)

    # 3) 距离矩阵 + 校验最优距离（用你现有的工具函数）
    D = build_distance_matrix(coords)
    best_distance = tour_length(best_tour, D)

    print("\n==== eil76（TSPLIB）加载结果 ====")
    print(f"节点数（不含 0 号仓库）：{len(coords) - 1}")
    print(f"最优路线首 10 个客户：{best_tour[:10]} ...")
    print(f"按现有 tour_length 计算的最优距离：{best_distance:.4f}")

    # 4) 绘图保存（与 GA/SA/ACO 风格一致；收敛曲线此处给单点）
    save_dir = "figs_tsplib_eil76"
    os.makedirs(save_dir, exist_ok=True)
    history = [best_distance]  # 用单点历史表示“已知最优”
    # algo_tag 写成 "TSPLIB-eil76 (opt)"，x 轴用 "Step"
    from pathlib import Path
    save_tsp_figs(
        history=history,
        coords=coords,
        best_tour=best_tour,
        algo_tag="TSPLIB-eil76 (opt)",
        save_dir=save_dir,
        conv_xlabel="Step",
        conv_name="eil76_opt_convergence.png",
        route_name="eil76_opt_route.png",
        dpi=180
    )

    print(f"图像已保存到：{Path(save_dir).resolve()}")


if __name__ == "__main__":
    main()
