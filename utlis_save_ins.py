# -*- coding: utf-8 -*-
"""
TSP 实例保存/读取模块
--------------------------------------------------
功能：
1) 保存随机生成的 TSP 实例为图片（节点散点图，含仓库 0 标签）；
2) 保存实例坐标为 JSON 数据文件；
3) 从 JSON 文件读取坐标供其他算法脚本复用；
4) 保持与现有脚本风格一致（使用 matplotlib 非交互后端、统一编号标注）。

用法示例（在 tsp_ga.py / tsp_sa.py / tsp_aco.py 中）：

    from tsp_instance_io import save_tsp_instance, load_tsp_instance
    # 生成 coords 后：
    paths = save_tsp_instance(
        coords,
        image_dir="figs_tsp_ga",   # 在 GA 中建议用对应目录
        data_dir="instances",
        seed=42,
        n_customers=40
    )
    # 读取复用：
    coords2 = load_tsp_instance(paths["data_path"])  # 或指定 JSON 路径
    # 然后在各脚本中：
    # D = build_distance_matrix(coords2)

注：默认不保存距离矩阵 D，以避免冗余存储；读取后由各算法脚本自行构建 D。
"""

import os
import json
import time
from typing import List, Tuple, Optional, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from utlis import generate_random_coords, build_distance_matrix, load_tsp_instance

# -----------------------------
# 内部工具
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coords_to_jsonable(coords: List[Tuple[float, float]]) -> List[List[float]]:
    # 将坐标转换为 JSON 可序列化的 [[x,y], ...]
    out: List[List[float]] = []
    for c in coords:
        # 兼容 numpy 数组/列表/元组
        x = float(c[0])
        y = float(c[1])
        out.append([x, y])
    return out


def _default_basename(seed: Optional[int], n_customers: Optional[int]) -> str:
    if seed is not None and n_customers is not None:
        return f"tsp_instance_seed{seed}_N{n_customers}"
    # 回退到时间戳，确保文件名唯一
    ts = int(time.time())
    return f"tsp_instance_{ts}"


# -----------------------------
# 对外 API
# -----------------------------

def plot_instance(
    coords: List[Tuple[float, float]],
    title: str = "TSP Instance",
    out_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    filename: str = "instance.png",
    dpi: int = 200,
) -> str:
    """
    保存实例散点图：包含仓库 0 与客户 1..N 的坐标及编号。

    - out_path：若提供则直接保存到此路径；
    - image_dir + filename：若未提供 out_path，则在 image_dir 下保存为 filename；
    - 返回最终保存的图片路径。
    """
    if out_path is None:
        img_dir = image_dir or "figs_tsp_instances"
        _ensure_dir(img_dir)
        out_path = os.path.join(img_dir, filename)
    else:
        _ensure_dir(os.path.dirname(out_path) or ".")

    # 准备数据
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # 画图（风格与现有脚本一致）：散点 + 编号
    plt.figure()
    # 区分仓库 0 与客户：
    plt.scatter(xs[1:], ys[1:], c="tab:blue", label="Customers")
    plt.scatter([xs[0]], [ys[0]], c="tab:red", marker="*", s=150, label="Depot 0")

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return out_path


def save_instance_json(
    coords: List[Tuple[float, float]],
    out_path: Optional[str] = None,
    data_dir: str = "instances",
    seed: Optional[int] = None,
    n_customers: Optional[int] = None,
    meta: Optional[Dict] = None,
) -> str:
    """
    将坐标保存为 JSON 文件，便于其他脚本读取复用。
    结构：{"coords": [[x0,y0], [x1,y1], ...], "meta": {...}}

    - out_path：若提供则直接保存到此路径；
    - data_dir：默认保存到 "instances" 目录下；
    - 文件名默认包含 seed 与 N（若提供），否则使用时间戳。
    - 返回最终保存的 JSON 路径。
    """
    if out_path is None:
        _ensure_dir(data_dir)
        base = _default_basename(seed, n_customers)
        out_path = os.path.join(data_dir, base + ".json")
    else:
        _ensure_dir(os.path.dirname(out_path) or ".")

    data = {
        "coords": _coords_to_jsonable(coords),
        "meta": {
            "seed": seed,
            "n_customers": n_customers,
            "depot_index": 0,
            **(meta or {}),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path

def save_tsp_instance(
    coords: List[Tuple[float, float]],
    image_dir: str = "figs_tsp_instances",
    data_dir: str = "instances",
    seed: Optional[int] = None,
    n_customers: Optional[int] = None,
) -> Dict[str, str]:
    """
    一次性保存图片与数据文件，返回路径字典：
    {"image_path": ..., "data_path": ...}

    - image_dir：建议在 GA/SA/ACO 脚本中分别传入对应目录，例如：
        GA -> "figs_tsp_ga"，SA -> "figs_tsp_sa"，ACO -> "figs_tsp_aco"。
    - data_dir：统一数据目录，默认 "instances"。
    - 文件名含 seed/N（若提供），避免多次运行覆盖；否则使用时间戳。
    """
    base = _default_basename(seed, n_customers)

    # 保存图片
    _ensure_dir(image_dir)
    image_path = os.path.join(image_dir, base + ".png")
    plot_instance(coords, title="TSP Instance", out_path=image_path)

    # 保存 JSON 数据
    data_path = save_instance_json(
        coords,
        out_path=os.path.join(data_dir, base + ".json"),
        data_dir=data_dir,
        seed=seed,
        n_customers=n_customers,
    )

    return {"image_path": image_path, "data_path": data_path}



if __name__ == "__main__":
    # 1) 生成随机实例
    n_customers = 40
    plane_size = 500
    coords = generate_random_coords(n_customers=n_customers, plane_size=plane_size, seed=42)
    D = build_distance_matrix(coords)

    # 读取随机生成的实例
    json_path = "instances/tsp_instance_seed42_N40.json"
    coords_loaded = load_tsp_instance(json_path)

    # 检查加载的坐标是否与原始坐标一致
    assert coords == coords_loaded, "加载的坐标与原始坐标不一致"

    # 2) 保存实例
    res = save_tsp_instance(
        coords,
        image_dir="figs_tsp_instances",
        data_dir="instances",
        seed=42,
        n_customers=n_customers,
    )
    print(res)