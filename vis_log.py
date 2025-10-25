# vis_improve_log.py
import os
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_algo_from_path(path: str) -> str:
    """
    根据文件路径返回算法名（假设路径中包含 improve_log_<algo>_...）
    例如：
        exp6/improve_log_aco_N10_seed2025.csv  ->  'aco'
    """
    filename = os.path.basename(path)
    # filename: improve_log_aco_N10_seed2025.csv
    parts = filename.split("_")
    if len(parts) >= 3 and parts[0] == "improve" and parts[1] == "log":
        algo = parts[2]
    elif len(parts) >= 2 and parts[0].startswith("improve") and parts[1]:
        # 备用匹配：improve_log_aco_... 或 improveaco_...
        algo = parts[1]
    else:
        raise ValueError(f"无法从文件名推断算法名: {filename}")
    return algo

def _plot_xy(
    x: Iterable[float],
    y: Iterable[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Best Distance",
    save_dir: str = "figs",
    filename: str = "plot.png",
    dpi: int = 200,
    tight: bool = True,
) -> str:
    """通用的折线图绘制函数：保存到 save_dir/filename，并返回保存路径。"""
    _ensure_dir(save_dir)
    x = list(x)
    y = list(y)

    plt.figure()
    plt.plot(x, y)
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


def plot_convergence_by_budget(
    df: pd.DataFrame,
    *,
    save_dir: str = "figs",
    filename: str = "aco_convergence_budget.png",
    dpi: int = 200,
    algo: str = "aco",
) -> str:
    """
    绘制 best_distance 随 E_budget 的收敛曲线。
    期望列: ['E_budget', 'best_distance']
    """
    filename = f"{algo}_convergence_budget.png"
    if not {"E_budget", "best_distance"}.issubset(df.columns):
        miss = {"E_budget", "best_distance"} - set(df.columns)
        raise ValueError(f"缺少必要列: {sorted(miss)}")

    return _plot_xy(
        df["E_budget"].to_list(),
        df["best_distance"].to_list(),
        title=f"{algo.upper()}-TSP Convergence (by E_budget)",
        xlabel="E_budget",
        ylabel="Best Distance",
        save_dir=save_dir,
        filename=filename,
        dpi=dpi,
    )


def plot_convergence_by_time(
    df: pd.DataFrame,
    *,
    save_dir: str = "figs",
    filename: str = "aco_convergence_time.png",
    dpi: int = 200,
    algo: str = "aco",
) -> str:
    """
    绘制 best_distance 随 time_sec 的变化曲线。
    期望列: ['time_sec', 'best_distance']
    """
    filename = f"{algo}_convergence_time.png"
    if not {"time_sec", "best_distance"}.issubset(df.columns):
        miss = {"time_sec", "best_distance"} - set(df.columns)
        raise ValueError(f"缺少必要列: {sorted(miss)}")

    return _plot_xy(
        df["time_sec"].to_list(),
        df["best_distance"].to_list(),
        title=f"{algo.upper()}-TSP Best Distance over Time",
        xlabel="Time (s)",
        ylabel="Best Distance",
        save_dir=save_dir,
        filename=filename,
        dpi=dpi,
    )


def load_log_csv(path: str) -> pd.DataFrame:
    """读取改进日志 CSV。默认会保持原始行序。"""
    df = pd.read_csv(path)
    # 若需要可在此处做排序/去重等预处理：
    # df = df.sort_values("E_budget", kind="stable")
    return df


def save_improve_log_figs(
    csv_path: str = "/mnt/data/improve_log_aco_N10_seed2025.csv",
    *,
    save_dir: str = "figs",
    dpi: int = 200,
) -> Tuple[str, str]:
    """
    一键读取 CSV 并保存两张图：
    - 按 E_budget 的收敛曲线
    - 按 time_sec 的收敛曲线
    返回(按预算图路径, 按时间图路径)。
    """
    df = load_log_csv(csv_path)

    # 从路径中提取算法名
    algo = get_algo_from_path(csv_path)

    path_budget = plot_convergence_by_budget(
        df, save_dir=save_dir, filename=f"{algo}_convergence_budget.png", dpi=dpi, algo=algo
    )
    path_time = plot_convergence_by_time(
        df, save_dir=save_dir, filename=f"{algo}_convergence_time.png", dpi=dpi, algo=algo
    )
    return path_budget, path_time


if __name__ == "__main__":
    budget_fig, time_fig = save_improve_log_figs(
        csv_path="exp1/improve_log_ga_N15_seed2025.csv",
        save_dir="exp1",
        dpi=200,
    )
    print("Saved:", budget_fig)
    print("Saved:", time_fig)
