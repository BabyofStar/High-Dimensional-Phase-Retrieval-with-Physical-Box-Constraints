# 带物理盒约束的高维相位恢复

《最优化理论》期末作业。本项目从多张编码衍射图样（Coded Diffraction Patterns，CDP）的强度观测中恢复实值振幅透射率，比较两种盒约束优化方法在无噪声恢复、含噪恢复和大规模计算中的表现。

## 问题与方法

设 $A$ 为由已知相位掩模和二维傅里叶变换构成的编码衍射算子，$y_j$ 为用于拟合的非负观测强度。待恢复图像满足物理盒约束 $x\in[0,1]^n$。项目求解以下平滑振幅最小二乘问题。

```math
\min_{x\in[0,1]^n}
\frac{1}{2m}\sum_{j=1}^{m}
\left(
\sqrt{\lvert(Ax)_j\rvert^2+\varepsilon^2}
-\sqrt{y_j+\varepsilon^2}
\right)^2
```

- **SP-NMPG** 采用截断谱初始化、BB1 步长、盒投影和窗口为 5 的非单调 Armijo 线搜索。
- **曲率阻尼双度量投影 L-BFGS** 在自由坐标上使用有限记忆拟牛顿更新，在积极集上使用对角度量，并用 Powell 阻尼修正弱曲率对。项目还实现了无约束对照和关闭阻尼的版本。

推导过程见[确定模型与推导](确定模型与推导.md)，随机种子、停止条件和指标口径见[实验协议](实验协议.md)。

## 实验用例

| 用例 | 测试图像 | 决策变量数 | 掩模数 | 观测条件 |
| --- | --- | ---: | ---: | --- |
| Case 1 | Shepp–Logan phantom | 16,384 | 6 | 无噪声 |
| Case 2 | camera | 16,384 | 3 | 泊松噪声，目标 SNR 20 dB |
| Case 3 | 灰度 astronaut | 65,536 | 4 | 高斯噪声，目标 SNR 30 dB |

同一用例的两种盒约束算法共享掩模、观测和谱初值。Case 1 另用五个均匀随机初值运行 SP-NMPG。Powell 阻尼消融复用 Case 2 的观测，对照双方共用 1% 幅度的谱初值。

## 复现步骤

以下命令以 **PowerShell 和 Python 3.12** 为例。

### 1. 克隆仓库并安装依赖

```powershell
git clone https://github.com/BabyofStar/High-Dimensional-Phase-Retrieval-with-Physical-Box-Constraints.git
cd High-Dimensional-Phase-Retrieval-with-Physical-Box-Constraints
python -m pip install -r requirements.txt
```

### 2. 下载测试原图

三张原图来自 [scikit-image v0.25.2](https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data)。程序会根据 `experiment_protocol.toml` 校验文件的 SHA-256，并在加载时完成灰度化和下采样。

```powershell
New-Item -ItemType Directory -Path data -Force | Out-Null
$source = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data"
foreach ($name in @("phantom.png", "camera.png", "astronaut.png")) {
    Invoke-WebRequest -Uri "$source/$name" -OutFile "data/$name"
}
```

### 3. 运行实验并生成图表

```powershell
python -m experiments --case all --output reproduced_results
python -m profile_case3 reproduced_results/case3
python -m plot_results reproduced_results --output reproduced_figures
```

`--case` 也可指定 `case1`、`case2`、`case3` 或 `ablation`。复跑时请选择新的实验输出目录。单独重绘某一用例时，可给绘图命令添加 `--case case2` 等选项。

下载原图后，可运行 `python -m unittest -v` 检查 CDP 算子、梯度、求解器和实验协议。

## 输出文件

| 输出 | 内容 |
| --- | --- |
| `reproduced_results/<用例>/inputs.npz` | 真值、掩模、无噪与原始观测、谱初值 |
| `reproduced_results/<用例>/summary.json` | 配置、输入哈希、运行环境及指标 |
| `reproduced_results/<用例>/<运行>/iterations.csv` | 目标值、残差、评估次数、回溯和用时 |
| `reproduced_results/<用例>/<运行>/reconstruction.npz` | 该次运行的初值和重构 |
| `reproduced_results/case3/profile.json` | 峰值内存和单次梯度计算开销 |
| `reproduced_figures/` | 初始化、收敛、重构及消融图，另含 `run_summary.csv` |

## 主要结果

- **Case 1** 的七次无噪声运行均达到相对重构误差低于 $10^{-7}$、KKT 残差低于 $10^{-8}$ 的验收阈值。
- **Case 2** 的实测 SNR 为 19.989 dB。盒约束 L-BFGS 的 PSNR 比无约束版本高约 0.104 dB，越界比例从 1.361% 降至 0。
- **Case 3** 的实测 SNR 约为 30.000 dB。投影 L-BFGS 用 53 次目标评估达到停止阈值，SP-NMPG 用 97 次。本次运行中，SP-NMPG 的实测求解耗时和峰值内存较低。
- **Powell 阻尼消融**中，开启阻尼的版本修正了一个负曲率对，关闭阻尼的版本跳过了该曲率对。两版均回溯 2 次并达到停止阈值。

上述结果对应仓库中固定的实验配置。耗时和峰值内存随硬件与运行环境而变化。

## 源码入口

- [`cdp.py`](cdp.py)　CDP 算子、损失与梯度、噪声模型及评估指标。
- [`sp_nmpg.py`](sp_nmpg.py)　截断谱初始化与非单调投影梯度求解器。
- [`projected_lbfgs.py`](projected_lbfgs.py)　双度量投影 L-BFGS 与 Powell 阻尼。
- [`experiment_protocol.toml`](experiment_protocol.toml)　实验参数、随机种子和评估预算。
- [`experiments.py`](experiments.py)、[`profile_case3.py`](profile_case3.py)、[`plot_results.py`](plot_results.py)　实验运行、资源测量和图表生成。
