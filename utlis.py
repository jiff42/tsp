import os
import json
import time
from typing import List, Tuple, Optional, Dict
import random
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import math

# ==============================
# 数据生成（随机 TSP 实例）
# ==============================

def generate_random_coords(n_customers: int = 30, plane_size: int = 100, seed: int = 7) -> List[Tuple[float, float]]:
    """
    生成坐标列表（含仓库 0）仓
    库放在平面中心，其余客户随机散布。
    """
    rng = random.Random(seed)
    coords = [(plane_size / 2.0, plane_size / 2.0)]  # depot at center
    for _ in range(n_customers):
        x = rng.uniform(0, plane_size)
        y = rng.uniform(0, plane_size)
        coords.append((x, y))
    if len(set(coords)) < n_customers + 1:
        raise ValueError("生成的坐标有重复，请调整参数重试。")

    return coords

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