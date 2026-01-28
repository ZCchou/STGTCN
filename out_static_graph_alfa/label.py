# -*- coding: utf-8 -*-
"""
Label ALFA flights in `alldata/` using failure_status under `ALFA/processed/<flight>/`.

Usage:
    python label_alfa_from_failure_status.py \
        --alldata_dir ./alldata \
        --alfa_processed_dir ./ALFA/processed \
        --out_dir ./alldata_labeled

说明：
- 自动识别并解析两种时间戳格式：
  * 数值型（自动判断 ns/us/ms/s）
  * 字符串型（如 '2018-09-11 18:59:38.100'）
- 对齐策略：将 failure_status 作为“状态序列”，对齐到 alldata 时间戳，以前向填充为主（最前段补 0）。
- 若同一航班下存在多个 failure_status 文件，则按并集（max）合并。
- no_failure 文件（文件名包含 'no_failure'）直接打全 0。
- 对数据空值做时间插值 + ffill/bfill 兜底。
"""

import argparse
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # 安静模式

def detect_time_unit_from_int(x: int) -> str:
    """
    根据整数时间戳数量级推断单位：ns/us/ms/s
    """
    # 取绝对值位数
    d = len(str(abs(int(x))))
    # 常见范围（粗略判断）
    if d >= 18:   # 1_000_000_000_000_000_000 (ns)
        return "ns"
    elif d >= 16: # 1_000_000_000_000_000 (us 可能)
        return "us"
    elif d >= 13: # 1_000_000_000_000 (ms)
        return "ms"
    else:
        return "s"

def to_datetime_series(s: pd.Series) -> pd.Series:
    """
    将 %time 列统一转换为 pandas datetime（无时区、纳秒精度）.
    兼容：
      - 字符串时间（例如 '2018-09-11 18:59:38.100'）
      - 数值型时间（ns/us/ms/s），自动判别单位
    """
    # 先尝试数值解析
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        # 以中位数判断单位
        med = s_num.dropna().median()
        unit = detect_time_unit_from_int(int(med))
        dt = pd.to_datetime(s_num.astype("Int64"), unit=unit, utc=True)
        return dt.dt.tz_convert(None)  # 去掉时区，转为 naive
    else:
        # 字符串解析
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
        # 若全是 NaT，尝试 dayfirst/指定格式两次兜底
        if dt.isna().all():
            dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt

def read_alldata_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    if "%time" not in df.columns:
        raise ValueError(f"[ERROR] {fp.name} 缺少 '%time' 列")
    # 解析时间列
    df["_dt"] = to_datetime_series(df["%time"])
    if df["_dt"].isna().all():
        raise ValueError(f"[ERROR] {fp.name} 的 '%time' 无法解析为 datetime")
    # 保持原 %time 字符串列，另建 _dt 用于对齐、插值
    return df

def read_failure_status_csv(fp: Path) -> pd.Series:
    """
    读取一个 failure_status CSV，返回以 datetime 为索引的状态序列（0/1 int）。
    期望列：%time, field.data
    """
    df = pd.read_csv(fp, low_memory=False, comment="#")
    # 有些文件可能存在 BOM 或奇怪表头空白，做一层列名清洗
    df.columns = [str(c).strip() for c in df.columns]
    # 容错：允许 field 列名出现轻微变体
    field_col = None
    for cand in ["field.data", "field.data[0]", "field.data[1]", "data", "value", "status"]:
        if cand in df.columns:
            field_col = cand
            break
    if field_col is None and len(df.columns) >= 2:
        # 退一步：取第2列作为状态列
        field_col = df.columns[1]

    if "%time" not in df.columns:
        # 退一步：尝试匹配包含 time 的列
        tcol = [c for c in df.columns if "%time" in c or "time" == c.lower()]
        if not tcol:
            raise ValueError(f"[ERROR] {fp} 未找到时间列")
        tcol = tcol[0]
    else:
        tcol = "%time"

    # 解析时间列
    df["_dt"] = to_datetime_series(df[tcol])
    df = df.loc[df["_dt"].notna()].copy()
    if df.empty:
        raise ValueError(f"[ERROR] {fp} 时间列解析为空")

    # 将状态转为 0/1
    s = df[field_col].copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({
            "1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0
        }).fillna(0)
    else:
        s = pd.to_numeric(s, errors="coerce").fillna(0)

    s = (s > 0).astype(int)
    s.index = df["_dt"].values
    s = s.sort_index()
    # 去重（同一时刻多条以 max 保守处理）
    s = s.groupby(level=0).max()
    return s

def collect_failure_status_series(flight_dir: Path) -> pd.Series:
    """
    汇总一个航班目录下的所有 failure_status 文件，返回合并后的状态序列（按逐时刻 max 合并）。
    若未找到，返回空的 Series。
    """
    patterns = [
        "*-failure_status*.csv",       # 常见模式
        "*failure_status*.csv",        # 兜底
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(flight_dir.glob(pat)))
    files = list(dict.fromkeys(files))  # 去重保序

    combined = None
    for f in files:
        try:
            s = read_failure_status_csv(f)
            combined = s if combined is None else pd.concat([combined, s], axis=1).max(axis=1)
        except Exception as e:
            print(f"[WARN] 读取 {f.name} 失败：{e}")

    if combined is None:
        return pd.Series(dtype="int64")  # 空
    return combined.astype(int)

def align_status_to_alldata_times(status_s: pd.Series, alldata_times: pd.Series) -> pd.Series:
    """
    将合并后的 failure_status（以 datetime 为索引）对齐到 alldata 的时间戳序列（同为 datetime）。
    策略：reindex -> ffill；最前段 NaN -> 0。
    """
    if status_s.empty:
        return pd.Series(0, index=alldata_times, dtype="int64")
    status_s = status_s.sort_index()
    # 对齐到 alldata 的时间戳
    s_aligned = status_s.reindex(alldata_times, method="pad")  # ffill
    s_aligned = s_aligned.fillna(0).astype(int)
    s_aligned.index = alldata_times
    return s_aligned

def sanitize_and_interpolate(df: pd.DataFrame, time_col: str = "_dt") -> pd.DataFrame:
    """
    对数值列按时间做插值；再 ffill/bfill 兜底；不处理 'label' 与 '%time' 原始列。
    """
    work = df.copy()
    if time_col not in work.columns:
        return work
    work = work.sort_values(time_col)
    work = work.set_index(time_col)

    protected = {"%time", "label"}
    numeric_cols = [c for c in work.columns if c not in protected and pd.api.types.is_numeric_dtype(work[c])]
    # 对于非数值列不改动（可能是字符串传感器状态等）
    if numeric_cols:
        try:
            work[numeric_cols] = work[numeric_cols].interpolate(method="time", limit_direction="both")
        except Exception:
            # 如果索引不是 DatetimeIndex 或失败，退化为线性插值
            work[numeric_cols] = work[numeric_cols].interpolate(limit_direction="both")
        work[numeric_cols] = work[numeric_cols].fillna(method="ffill").fillna(method="bfill")

    work = work.reset_index()
    return work

def main():
    ap = argparse.ArgumentParser(description="Label ALFA alldata using failure_status under ALFA/processed.")
    ap.add_argument("--alldata_dir", type=str, default="./alldata", help="目录：别人处理好的每航班一个 CSV")
    ap.add_argument("--alfa_processed_dir", type=str, default="./ALFA/processed", help="目录：ALFA 原始 processed")
    ap.add_argument("--out_dir", type=str, default="./alldata_labeled", help="输出目录")
    ap.add_argument("--overwrite", action="store_true", help="已存在目标文件是否覆盖")
    args = ap.parse_args()

    alldata_dir = Path(args.alldata_dir).resolve()
    processed_dir = Path(args.alfa_processed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not alldata_dir.exists():
        print(f"[ERROR] alldata_dir 不存在：{alldata_dir}")
        sys.exit(1)
    if not processed_dir.exists():
        print(f"[ERROR] alfa_processed_dir 不存在：{processed_dir}")
        sys.exit(1)

    csvs = sorted(alldata_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] {alldata_dir} 下未发现 CSV")
        sys.exit(0)

    print(f"[INFO] 将处理 {len(csvs)} 个航班文件，输出至：{out_dir}")

    for i, fp in enumerate(csvs, 1):
        flight_name = fp.stem  # 与 processed/<flight_name>/ 对应
        out_fp = out_dir / f"{flight_name}.csv"

        if out_fp.exists() and not args.overwrite:
            print(f"[SKIP {i}/{len(csvs)}] {fp.name} → 目标已存在（跳过）。")
            continue

        print(f"[{i}/{len(csvs)}] 处理航班：{fp.name}")

        try:
            df = read_alldata_csv(fp)
        except Exception as e:
            print(f"[ERROR] 读取 alldata 失败：{fp.name} | {e}")
            continue

        # no_failure 直接打 0
        is_no_failure = ("no_failure" in flight_name.lower())
        flight_dir = processed_dir / flight_name

        if is_no_failure:
            print(f"  - 检测到 no_failure，整段标 0")
            label = pd.Series(0, index=df["_dt"], dtype="int64")
        else:
            if not flight_dir.exists():
                print(f"  [WARN] 未找到对应原始目录：{flight_dir.name}，将整段标 0")
                label = pd.Series(0, index=df["_dt"], dtype="int64")
            else:
                status_s = collect_failure_status_series(flight_dir)
                if status_s.empty:
                    print(f"  [WARN] 未找到任何 failure_status 文件，整段标 0")
                    label = pd.Series(0, index=df["_dt"], dtype="int64")
                else:
                    print(f"  - 已汇总 failure_status，共 {status_s.shape[0]} 条状态时间点")
                    label = align_status_to_alldata_times(status_s, df["_dt"])

        # 附加标签列（0/1）
        df["label"] = label.values.astype(np.int64)

        # 空值处理（不动 %time 和 label）
        df_out = sanitize_and_interpolate(df, time_col="_dt")

        # 保留原列顺序：将 _dt 放后或移除（可选）
        # 这里保留 %time（原样字符串），保留 _dt（对齐/插值用），也可按需注释掉下一行移除 _dt
        df_out = df_out.drop(columns=["_dt"])

        # 保存
        try:
            df_out.to_csv(out_fp, index=False)
            n1 = int(df_out["label"].sum())
            print(f"  -> 保存：{out_fp.name} | 总行数={len(df_out)} | 异常标记数(1)={n1}")
        except Exception as e:
            print(f"[ERROR] 保存失败：{out_fp} | {e}")

    print("[DONE] 全部处理完成。")

if __name__ == "__main__":
    main()
