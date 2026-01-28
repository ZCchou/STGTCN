# -*- coding: utf-8 -*-
"""
可视化 GraphBuild 产出的邻接矩阵 adjacency_dense.csv 为热力图。
- 支持排序方式：none（原序）、degree（按度降序）、spectral（谱排序，Fiedler 向量）
- 对角线方向：默认从左上到右下（origin='upper'）
- 自动调节刻度密度，避免标签过多挤在一起

用法：
    python visualize_adjacency.py \
        --adj_csv out_static_graph_gpsatt/adjacency_dense.csv \
        --order spectral \
        --out_png out_static_graph_gpsatt/adjacency_heatmap.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj_csv", default="out_static_graph_gpsatt/adjacency_dense.csv",
                    help="邻接矩阵 CSV（GraphBuild 输出）")
    ap.add_argument("--order", choices=["none", "degree", "spectral"], default="spectral",
                    help="节点重排方式：none/degree/spectral，默认 spectral")
    ap.add_argument("--out_png", default="out_static_graph_gpsatt/adjacency_heatmap.png",
                    help="输出图片路径（.png/.pdf/.svg 均可）")
    ap.add_argument("--max_labels", type=int, default=50,
                    help="坐标轴最多显示的标签数（默认 50）")
    return ap.parse_args()


def _spectral_order(A: np.ndarray) -> np.ndarray:
    """谱排序：按 Laplacian 第二小特征向量（Fiedler 向量）排序；失败则退化为度排序。"""
    n = A.shape[0]
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    try:
        evals, evecs = np.linalg.eigh(L)  # 对称矩阵的特征分解
        if len(evals) >= 2:
            fiedler = evecs[:, 1]
            return np.argsort(fiedler)
        else:
            return np.argsort(-deg)
    except Exception:
        return np.argsort(-deg)


def _degree_order(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    return np.argsort(-deg)


def reorder(A: np.ndarray, labels: list, mode: str):
    if mode == "none":
        idx = np.arange(A.shape[0])
    elif mode == "degree":
        idx = _degree_order(A)
    else:  # spectral
        idx = _spectral_order(A)
    return A[np.ix_(idx, idx)], [labels[i] for i in idx], idx


def main():
    args = parse_args()
    adj_path = Path(args.adj_csv)
    assert adj_path.exists(), f"找不到邻接矩阵文件：{adj_path}"

    # 读矩阵（第一列为索引）
    df = pd.read_csv(adj_path, index_col=0)
    labels = df.index.astype(str).tolist()
    A = df.values.astype(float)

    # 清理 & 对称化
    A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
    A = 0.5 * (A + A.T)
    A = np.clip(A, 0.0, 1.0)

    # 重排
    A_ord, labels_ord, idx = reorder(A, labels, args.order)

    n = A_ord.shape[0]
    figsize = (min(0.2 * n + 4, 20), min(0.2 * n + 4, 20))

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    # 关键调整：origin='upper' → 对角线从左上到右下
    im = ax.imshow(A_ord, origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("weight", rotation=90)

    # 轴标签稀疏化
    step = max(1, n // max(1, args.max_labels))
    xticks = np.arange(0, n, step)
    yticks = np.arange(0, n, step)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([labels_ord[i] for i in xticks], rotation=90, fontsize=8)
    ax.set_yticklabels([labels_ord[i] for i in yticks], fontsize=8)

    ax.set_xlabel(f"nodes ({args.order} ordered)")
    ax.set_ylabel(f"nodes ({args.order} ordered)")
    ax.set_title(f"Adjacency Heatmap ({args.order}), n={n}")

    fig.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[OK] 保存热力图：{out_path}")


if __name__ == "__main__":
    main()
