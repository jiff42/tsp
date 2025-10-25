# -*- coding: utf-8 -*-
"""
loader_tsplib.py
----------------
封装一个接口：load_tsplib_instance(tsp_path, tour_path)
读取 TSPLIB 的 .tsp 与 .tour 文件，转换为当前工程通用格式：
    coords: [ (x0,y0), (x1,y1), ... ] ，下标0为仓库（取首城）
    D: 距离矩阵（含仓库）
    best_tour: 不含0号仓库的客户访问序列
    best_distance: 按现有 tour_length 计算的闭环距离
"""

import os
from typing import List, Tuple, Dict

import numpy as np
from utlis import build_distance_matrix, tour_length


def parse_tsplib_tsp(path_tsp: str) -> Dict[int, Tuple[float, float]]:
    """读取 .tsp 文件（支持 NODE_COORD_SECTION）"""
    coords = {}
    with open(path_tsp, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]
    # 找 NODE_COORD_SECTION
    start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("NODE_COORD_SECTION"):
            start = i + 1
            break
    if start is None:
        raise ValueError("未找到 NODE_COORD_SECTION")

    for ln in lines[start:]:
        if not ln or ln.upper().startswith("EOF"):
            break
        parts = ln.split()
        if len(parts) >= 3:
            idx = int(float(parts[0]))
            coords[idx] = (float(parts[1]), float(parts[2]))
    return coords


def parse_tsplib_tour(path_tour: str) -> List[int]:
    """读取 .tour 文件（TOUR_SECTION 后的序列，-1 结束）"""
    seq = []
    with open(path_tour, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]
    start = None
    for i, ln in enumerate(lines):
        if ln.upper().startswith("TOUR_SECTION"):
            start = i + 1
            break
    if start is None:
        raise ValueError("未找到 TOUR_SECTION")
    for ln in lines[start:]:
        if not ln or ln.startswith("-1") or ln.upper().startswith("EOF"):
            break
        seq.append(int(float(ln)))
    return seq


def tsplib_to_workspace_format(tsplib_coords: Dict[int, Tuple[float, float]],
                               tsplib_tour: List[int]) -> Tuple[List[Tuple[float, float]], List[int]]:
    """转换为现有工程格式（仓库+客户）"""
    N = len(tsplib_coords)
    first_city = tsplib_tour[0]
    depot_xy = tsplib_coords[first_city]
    coords = [depot_xy] + [None] * N  # type: ignore
    for i in range(1, N + 1):
        coords[i] = tsplib_coords[i]
    best_tour = tsplib_tour[1:]
    return coords, best_tour


def load_tsplib_instance(path_tsp: str, path_tour: str):
    """
    主函数：读取 + 转换 + 构建距离矩阵 + 返回四元组
    返回：
        coords, D, best_tour, best_distance
    """
    tsplib_coords = parse_tsplib_tsp(path_tsp)
    tsplib_tour = parse_tsplib_tour(path_tour)
    coords, best_tour = tsplib_to_workspace_format(tsplib_coords, tsplib_tour)
    D = build_distance_matrix(coords)
    best_distance = tour_length(best_tour, D)
    return coords, D, best_tour, best_distance


# 示例运行
if __name__ == "__main__":
    tsp_path = "tsp_lib/a280.tsp"
    tour_path = "tsp_lib/a280.opt.tour"
    coords, D, best_tour, best_distance = load_tsplib_instance(tsp_path, tour_path)
    print(f"共 {len(coords)-1} 个城市，最优长度：{best_distance:.4f}")
    print(f"前10个城市：{best_tour[:10]}")
