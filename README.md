# STGTCN

本项目为论文 **STGTCN** 的代码实现，面向无人机飞行日志中的多源传感器时间序列异常检测。项目包含两个数据集（ALFA 与 GPSAttack）的实验与消融实验配置。

## 项目结构说明

- `out_static_graph_alfa/`、`out_static_graph_gpsatt/`：用于构建静态图结构的脚本与输出（静态图先验）。
- `*.tar.gz`：两个**已预处理**的数据集压缩包（`alfadata.tar.gz` 与 `gpsdata.tar.gz`）。
- `data/`：数据目录（默认为空），可用于存放解压后的数据集。

## 论文摘要

With the routine deployment of unmanned aerial vehicles (UAVs) in inspection, agriculture, and emergency response, multi-source sensor time-series data recorded in flight logs have become critical evidence for fault troubleshooting and risk diagnosis. However, the scarcity of anomalous samples, pronounced noise and distribution drift, rapid operating-condition switches, and complex inter-sensor couplings make offline anomaly detection prone to false alarms and performance fluctuations when generalizing across flights. To address these challenges, we propose a spatio-temporal forecasting model termed STGTCN. On the spatial side, we construct a density-constrained static graph prior based on the maximal information coefficient (MIC), and fuse it with an attention-induced dynamic graph to jointly achieve stable regularization and adaptive relation updating. On the temporal side, a causal TCN and a causal-masked Transformer are combined to model both local and long-range dependencies, while cross-branch alignment is introduced to alleviate spatio-temporal semantic misalignment. On the decision side, we aggregate prediction residuals and apply causal smoothing to obtain an anomaly score, and then estimate in parallel a median absolute deviation (MAD)-based global robust threshold and a flight-level adaptive threshold, where a more conservative boundary is adopted to enable stable alarming. Experiments on the ALFA fault dataset and the GPSAttack attack dataset demonstrate the effectiveness of the proposed method: STGTCN achieves mean F1 scores of 85.11 and 99.03, respectively, and exhibits more consistent advantages in precision and false-alarm control. Code release \url{https://github.com/ZCchou/STGTCN}.
