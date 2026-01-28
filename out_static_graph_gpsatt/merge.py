# -*- coding: utf-8 -*-


from pathlib import Path
import pandas as pd
import numpy as np

# 根目录（请按需修改）
BASE_DIR = r"Live GPS Spoofing and Jamming"

# 需要处理的子目录名
SUBFOLDERS = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]

# 打标签阈值（度），与原仓库一致（约公里级）
DEG_THRESHOLD = 0.03


def read_csv_sorted(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name} 缺少 'timestamp' 列")
    return df.sort_values("timestamp").reset_index(drop=True)


def find_topic_csv(folder: Path, keys: list[str]) -> Path | None:
    """
    在 folder 内查找包含任一 key 的 CSV 文件（如 *vehicle_gps_position_0*.csv）
    返回匹配到的第一份。若未找到，返回 None。
    """
    for key in keys:
        for p in sorted(folder.glob(f"*{key}*.csv")):
            return p
    return None


def merge_chain_outer(dfs: list[pd.DataFrame | None]) -> pd.DataFrame:
    """
    依次用 outer-merge(on='timestamp') 链式合并。
    """
    final = pd.DataFrame()
    for df in dfs:
        if df is None or df.empty:
            continue
        if final.empty:
            final = df
        else:
            final = final.merge(df, how="outer", on="timestamp")
    if final.empty:
        raise RuntimeError("合并结果为空，请检查四个话题 CSV 是否存在。")
    return final


def linear_interpolate_on_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    timestamp 排序 → 设为索引 → 线性插值（双向补齐）→ 恢复列
    """
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.interpolate(axis=0, method="linear", limit_direction="both")
    return df.reset_index()


def drop_extra_timestamp_like_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除除主 'timestamp' 外，任何列名包含 'timestamp' 的列（例如合并产生的 timestamp_x 等）。
    """
    droplist = [c for c in df.columns if ("timestamp" in c and c != "timestamp")]
    if droplist:
        df = df.drop(columns=droplist, errors="ignore")
    return df


def label_like_original(df: pd.DataFrame,
                        lat_col: str = "lat_x",
                        lon_col: str = "lon_x",
                        deg_threshold: float = DEG_THRESHOLD) -> pd.DataFrame:
    """
    以首行 lat_x/lon_x 为参考点，任何一行若经/纬偏离超过 ±deg_threshold → malicious，否则 benign。
    若缺少 lat_x/lon_x（或全 NaN），则全部标为 benign 并给出提示。
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"[WARN] 未发现 {lat_col}/{lon_col}，label 全设为 'benign'")
        df["label"] = "benign"
        return df

    lat = pd.to_numeric(df[lat_col], errors="coerce").ffill().bfill()
    lon = pd.to_numeric(df[lon_col], errors="coerce").ffill().bfill()

    if lat.isna().all() or lon.isna().all():
        print(f"[WARN] {lat_col}/{lon_col} 全为 NaN，label 全设为 'benign'")
        df["label"] = "benign"
        return df

    lat0, lon0 = lat.iloc[0], lon.iloc[0]
    cond = (
        (lat > lat0 + deg_threshold) | (lat < lat0 - deg_threshold) |
        (lon > lon0 + deg_threshold) | (lon < lon0 - deg_threshold)
    )
    df["label"] = np.where(cond, "malicious", "benign")
    vc = df["label"].value_counts(dropna=False)
    print(f"[INFO] 标签分布：{vc.to_dict()}")
    return df


def process_one_folder(folder: Path) -> None:
    print(f"\n==== 处理目录：{folder} ====")
    if not folder.exists():
        print(f"[WARN] 目录不存在，跳过：{folder}")
        return

    # 话题关键字（兼容 vehicle_* 与省略前缀）
    topic_keys = {
        "att":  ["vehicle_attitude_0", "attitude_0"],
        "gpos": ["vehicle_global_position_0", "global_position_0"],
        "gps":  ["vehicle_gps_position_0", "gps_position_0"],
        "lpos": ["vehicle_local_position_0", "local_position_0"],
    }

    # 查找 4 张 CSV（严格按顺序：att → gpos → gps → lpos）
    att_path  = find_topic_csv(folder, topic_keys["att"])
    gpos_path = find_topic_csv(folder, topic_keys["gpos"])
    gps_path  = find_topic_csv(folder, topic_keys["gps"])
    lpos_path = find_topic_csv(folder, topic_keys["lpos"])

    for name, p in [("attitude", att_path),
                    ("global_position", gpos_path),
                    ("gps_position", gps_path),
                    ("local_position", lpos_path)]:
        print(f"[INFO] {name:16s}: {p.name if p else '未找到'}")

    # 读取（缺哪个就 None，仍会继续 outer 合并）
    att_df  = read_csv_sorted(att_path)  if att_path  else None
    gpos_df = read_csv_sorted(gpos_path) if gpos_path else None
    gps_df  = read_csv_sorted(gps_path)  if gps_path  else None
    lpos_df = read_csv_sorted(lpos_path) if lpos_path else None

    # 外连接合并
    final_df = merge_chain_outer([att_df, gpos_df, gps_df, lpos_df])

    # 排序 + 线性插值
    final_df = linear_interpolate_on_timestamp(final_df)

    # 删除多余 timestamp* 列
    final_df = drop_extra_timestamp_like_cols(final_df)

    # 按原规则打标签（lat_x/lon_x）
    final_df = label_like_original(final_df, lat_col="lat_x", lon_col="lon_x", deg_threshold=DEG_THRESHOLD)

    # 输出
    out_dir = folder / "CSVs" / "Condensed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder.name}.csv"   # 与原作者命名风格一致：<目录名>.csv
    final_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] 写出：{out_path}  行数={len(final_df)}, 列数={final_df.shape[1]}")


def main():
    base = Path(BASE_DIR)
    for sub in SUBFOLDERS:
        process_one_folder(base / sub)


if __name__ == "__main__":
    main()
