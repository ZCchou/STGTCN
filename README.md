# STGTCN

本项目为论文 **STGTCN** 的官方代码实现，面向无人机飞行日志中的多源传感器时间序列异常检测与诊断。项目包含 ALFA 与 GPSAttack 两个数据集的实验与消融实验配置。

## 论文摘要

With the routine deployment of unmanned aerial vehicles (UAVs) in inspection, agriculture, and emergency response, multi-source sensor time-series data recorded in flight logs have become critical evidence for fault troubleshooting and risk diagnosis. However, the scarcity of anomalous samples, pronounced noise and distribution drift, rapid operating-condition switches, and complex inter-sensor couplings make offline anomaly detection prone to false alarms and performance fluctuations when generalizing across flights. To address these challenges, we propose a spatio-temporal forecasting model termed STGTCN. On the spatial side, we construct a density-constrained static graph prior based on the maximal information coefficient (MIC), and fuse it with an attention-induced dynamic graph to jointly achieve stable regularization and adaptive relation updating. On the temporal side, a causal TCN and a causal-masked Transformer are combined to model both local and long-range dependencies, while cross-branch alignment is introduced to alleviate spatio-temporal semantic misalignment. On the decision side, we aggregate prediction residuals and apply causal smoothing to obtain an anomaly score, and then estimate in parallel a median absolute deviation (MAD)-based global robust threshold and a flight-level adaptive threshold, where a more conservative boundary is adopted to enable stable alarming. Experiments on the ALFA fault dataset and the GPSAttack attack dataset demonstrate the effectiveness of the proposed method: STGTCN achieves mean F1 scores of 85.11 and 99.03, respectively, and exhibits more consistent advantages in precision and false-alarm control. Code release \url{https://github.com/ZCchou/STGTCN}.

## 项目结构说明

| 路径 | 说明 |
| --- | --- |
| `README.md` | 项目说明与快速开始文档。 |
| `train_alfa.py` | ALFA 数据集训练与评估主脚本（默认读取 `alfadata/No_Failure` 与 `alfadata/Failure`）。 |
| `train_gpsbest.py` | GPSAttack 数据集训练与评估主脚本（默认读取 `gpsdata/No_Failure` 与 `gpsdata/Failure`）。 |
| `ablationalfa.py` | ALFA 数据集消融实验脚本（支持关闭时空分支等开关）。 |
| `ablationgps.py` | GPSAttack 数据集消融实验脚本。 |
| `model/` | 模型实现：STGTCN 主体、消融版、预测版与基线预测模型。 |
| `out_static_graph_alfa/` | ALFA 静态图构建脚本与输出（静态图先验）。 |
| `out_static_graph_gpsatt/` | GPSAttack 静态图构建脚本与输出（静态图先验）。 |
| `alfadata.tar.gz` | 预处理后的 ALFA 数据集压缩包。 |
| `gpsdata.tar.gz` | 预处理后的 GPSAttack 数据集压缩包。 |
| `data/` | 空目录（需自行创建），建议用于存放解压后的数据或模型检查点。 |

## 依赖与安装

请在项目根目录执行：

```bash
pip install -r reaquirement.txt
```

`reaquirement.txt` 中仅包含运行脚本所需的最小依赖，主要包括：

- numpy
- pandas
- torch
- tqdm
- matplotlib
- scipy
- minepy（用于 MIC 计算，缺失时脚本会提示或自动降级）

## 数据集获取

- **ALFA dataset**: https://theairlab.org/alfa-dataset/
- **UAV GPSAttack dataset**: https://ieee-dataport.org/open-access/uav-attack-dataset

## 快速开始

### 1) 准备数据

使用本仓库提供的预处理数据（推荐快速复现）：

```bash
tar -xzf alfadata.tar.gz
tar -xzf gpsdata.tar.gz
```

解压后将得到 `alfadata/` 与 `gpsdata/` 目录，结构与脚本默认路径匹配：

- `alfadata/No_Failure/`、`alfadata/Failure/`
- `gpsdata/No_Failure/`、`gpsdata/Failure/`

如果你希望从原始数据集自行预处理，请先从上方链接下载原始数据，再按照数据集说明整理成上述目录结构。

### 2) （可选）构建静态图

```bash
python out_static_graph_alfa/Buildgraph.py --help
python out_static_graph_gpsatt/Bulidgraphgpsatt.py --help
```

静态图构建完成后，输出会保存在对应的 `out_static_graph_*` 目录，供训练脚本调用。

### 3) 训练与评估

```bash
python train_alfa.py
python train_gpsbest.py
```

### 4) 消融实验

```bash
python ablationalfa.py
python ablationgps.py
```

> 注：脚本内置默认配置，可按需修改脚本中的参数（如数据路径、训练轮次、模型规模等）。
