# -*- coding: utf-8 -*-
"""
plot_one_csv.py
---------------
读取一个 CSV (time_sec, E, best_distance) 并绘制收敛曲线。
支持两种横轴：
1) 时间 Time (s)
2) 评估步数 E
"""

import csv
import matplotlib.pyplot as plt

def plot_one_csv(csv_path, x_axis="time"):
    """
    csv_path: 日志文件路径，例如 'logs/improve_log_ga_N50_seed2025.csv'
    x_axis:   'time' 或 'E'，决定横轴用时间还是评估次数
    """
    # --- 读取 CSV ---
    t_list, E_list, dist_list = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            t_list.append(float(row[0]))
            E_list.append(int(row[1]))
            dist_list.append(float(row[2]))

    # --- 绘图 ---
    plt.figure()
    if x_axis == "time":
        plt.plot(t_list, dist_list, linewidth=2)
        plt.xlabel("Time (s)")
    else:
        plt.plot(E_list, dist_list, linewidth=2)
        plt.xlabel("Evaluation steps (E)")

    plt.ylabel("Best tour length")
    plt.title(f"Convergence Curve ({x_axis}-axis)")
    plt.grid(True)
    plt.tight_layout()

    # 自动保存
    out_name = csv_path.replace(".csv", f"_{x_axis}_curve.png")
    plt.savefig(out_name, dpi=200)
    plt.close()
    print(f"✅ 已保存图像: {out_name}")


# 示例用法
if __name__ == "__main__":
    # 你只需改这里的路径
    csv_path = "logs/improve_log_ga_N50_seed2025.csv"
    plot_one_csv(csv_path, x_axis="time")   # 按时间绘制
    plot_one_csv(csv_path, x_axis="E")      # 按 E 绘制
