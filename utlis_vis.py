# utils_vis.py
import os
from typing import List, Tuple, Iterable, Optional
import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_convergence(
    history: Iterable[float],
    *,
    title: str = "Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Best Distance",
    save_dir: str = "figs",
    filename: str = "convergence.png",
    dpi: int = 200,
    tight: bool = True,
) -> str:
    """保存收敛曲线到 save_dir/filename，并返回保存路径。"""
    _ensure_dir(save_dir)
    y = list(history)
    plt.figure()
    plt.plot(range(len(y)), y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if tight:
        plt.tight_layout()
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    return out_path

def plot_route(
    coords: List[Tuple[float, float]],
    tour: List[int],
    *,
    title: str = "Best Route",
    save_dir: str = "figs",
    filename: str = "route.png",
    dpi: int = 200,
    annotate: bool = True,
    tight: bool = True,
) -> str:
    """根据坐标与 best_tour 画路线图并保存到 save_dir/filename，返回保存路径。"""
    _ensure_dir(save_dir)
    xs = [coords[0][0]] + [coords[i][0] for i in tour] + [coords[0][0]]
    ys = [coords[0][1]] + [coords[i][1] for i in tour] + [coords[0][1]]

    plt.figure()
    plt.scatter([c[0] for c in coords], [c[1] for c in coords])
    plt.plot(xs, ys)
    if annotate:
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, str(i))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    if tight:
        plt.tight_layout()
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    return out_path

def save_tsp_figs(
    *,
    history: Iterable[float],
    coords: List[Tuple[float, float]],
    best_tour: List[int],
    algo_tag: str,           # 例如 "ACO-TSP" / "GA-TSP" / "SA-TSP"
    save_dir: str,           # 输出目录
    conv_xlabel: str,        # "Iteration" 或 "Generation"
    conv_name: str = None,   # 文件名可覆写
    route_name: str = None,  # 文件名可覆写
    dpi: int = 200,
) -> Tuple[str, str]:
    """一键保存收敛曲线和路线图，返回(收敛图路径, 路线图路径)。"""
    conv_name = conv_name or f"{algo_tag.lower().replace(' ', '_')}_convergence.png"
    route_name = route_name or f"{algo_tag.lower().replace(' ', '_')}_route.png"

    conv_path = plot_convergence(
        history,
        title=f"{algo_tag} Convergence",
        xlabel=conv_xlabel,
        save_dir=save_dir,
        filename=conv_name,
        dpi=dpi,
    )
    route_path = plot_route(
        coords,
        best_tour,
        title=f"{algo_tag} Best Route",
        save_dir=save_dir,
        filename=route_name,
        dpi=dpi,
    )
    return conv_path, route_path
