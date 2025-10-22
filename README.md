# 项目说明

本仓库包含若干个用于路径规划/组合优化的示例脚本，覆盖 TSP（旅行商问题）与 VRP（带容量约束的路径问题）等典型模型，示例均可直接运行并将图像结果保存到本地目录。

## 环境依赖
- Python 3.8+
- numpy, matplotlib

安装依赖（示例）：
```bash
pip install numpy matplotlib
```

## 文件说明
- `tsp_ga.py`：使用遗传算法（GA）求解 TSP。
  - 随机生成含仓库 `0` 的节点；输出收敛曲线与最优路线图到 `figs_tsp_ga/`。
  - 运行：`python tsp_ga.py`
- `tsp_sa.py`：使用模拟退火（SA）求解 TSP。
  - 保存收敛曲线与最优路线图到 `figs_tsp_sa/`。
  - 运行：`python tsp_sa.py`
- `tsp_aco.py`：使用蚁群算法（ACO）求解 TSP。
  - 保存收敛曲线与最优路线图到 `figs_tsp_aco/`。
  - 运行：`python tsp_aco.py`
- `route_opt.py`：路径规划与物流调度参考实现（GA/NSGA-II）。
  - 包含：TSP 的 GA、CVRP 的 GA（巨型序列 + 贪心 Split 解码 + 约束惩罚）、VRP 的 NSGA-II 多目标（f1=总里程, f2=最大单车里程），以及基础可视化。
  - 运行：`python route_opt.py`（示例会生成并保存图像到 `figs/`）。

## 目录说明
- `figs/`：`route_opt.py` 生成的图（如 `best_tsp_route.png`, `tsp_convergence.png`, `best_cvrp_routes.png`, `cvrp_convergence.png`, `nsga_ii_pareto_front.png`）。
- `figs_tsp_ga/`：`tsp_ga.py` 的输出图（`convergence.png`, `route.png`）。
- `figs_tsp_sa/`：`tsp_sa.py` 的输出图（`sa_convergence.png`, `sa_route.png`）。
- `figs_tsp_aco/`：`tsp_aco.py` 的输出图（`aco_convergence.png`, `aco_best_route.png`）。
- `__pycache__/`：Python 缓存目录。
- `.vscode/`：VS Code 配置。

## 备注
- 各脚本会随机生成问题实例：仓库索引为 `0`，客户为 `1..N`，路线均为 `0 -> ... -> 0`。
- 运行后不弹出图形窗口，图片会保存到对应目录中。
- 具体参数（客户数量、迭代次数、温度/蚂蚁数/种群规模等）可在脚本内修改。
