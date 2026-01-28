# -*- coding: utf-8 -*-
"""
对三份合并后的 CSV（Benign Flight / GPS Jamming / GPS Spoofing）进行数据清洗，
并仅保留三者共有的列，分别输出到脚本同目录的 preprocessed/ 文件夹。
额外：将 label 列统一转换为 0/1（benign→0，malicious→1）。

输入默认位置（与前一步脚本一致的输出约定）：
  <BASE_DIR>/<SubFolder>/CSVs/Condensed/gpsonly.csv
  或 <BASE_DIR>/<SubFolder>/CSVs/Condensed/<SubFolder>.csv
  若均不存在，则取 Condensed 目录下第一个 CSV

输出：
  ./preprocessed/Benign Flight.csv
  ./preprocessed/GPS Jamming.csv
  ./preprocessed/GPS Spoofing.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict

# ======== 配置：根目录与子目录名 ========
BASE_DIR = r"Live GPS Spoofing and Jamming"
SUBFOLDERS = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]
# =====================================

def find_condensed_csv(folder: Path) -> Optional[Path]:
    """
    在 <folder>/CSVs/Condensed 下，依次尝试：
      1) gpsonly.csv
      2) <folder.name>.csv
      3) 目录下第一个 *.csv
    找不到返回 None
    """
    con = folder / "Processed"
    if not con.exists():
        return None
    cand1 = con / "gpsonly.csv"
    if cand1.exists():
        return cand1
    cand2 = con / f"{folder.name}.csv"
    if cand2.exists():
        return cand2
    csvs = sorted(con.glob("*.csv"))
    return csvs[0] if csvs else None

def as_int_ts(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def coerce_bool_like(col: pd.Series) -> pd.Series:
    """
    将布尔/布尔字符串统一到 0/1；其它类型保持原状或尽量转数值。
    """
    if col.dtype == bool:
        return col.astype(np.int8)
    if col.dtype == object:
        lower = col.astype(str).str.strip().str.lower()
        mask_true  = lower.isin(["true","t","yes","y","1"])
        mask_false = lower.isin(["false","f","no","n","0"])
        if (mask_true | mask_false).any():
            out = col.copy()
            out[mask_true]  = 1
            out[mask_false] = 0
            try:
                return pd.to_numeric(out, errors="ignore")
            except Exception:
                return out
    return col

def coerce_numeric_except_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    除 'label' 外，尽量将 object 列转为数值；并把布尔类列转为 0/1。
    """
    for c in df.columns:
        if c == "label":
            continue
        df[c] = coerce_bool_like(df[c])
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 label 列稳健地映射为 0/1：
      - 'benign','normal','0','false','no','n' -> 0
      - 'malicious','attack','1','true','yes','y' -> 1
      - 可解析为数字的字符串 -> 非0为1，0为0
      - 其它无法识别的值 -> 按 0 处理（保守）
    若无 label 列则原样返回。
    """
    if "label" not in df.columns:
        return df

    s = df["label"]
    # 已是数字：非0为1
    if pd.api.types.is_numeric_dtype(s):
        df["label"] = (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype(np.int8)
        return df

    # 字符串映射
    lower = s.astype(str).str.strip().str.lower()
    mapping = {
        "benign": 0, "normal": 0, "0": 0, "false": 0, "f": 0, "no": 0, "n": 0,
        "malicious": 1, "attack": 1, "1": 1, "true": 1, "t": 1, "yes": 1, "y": 1
    }
    m = lower.map(mapping)

    # 对无法直接映射的值，尝试按数字解析
    num = pd.to_numeric(lower, errors="coerce")
    m = m.where(~m.isna(), num)
    # 最终默认未知→0（保守）
    df["label"] = m.fillna(0).astype(np.int8)
    return df

def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    温和清洗：
      - 统一 timestamp 数值化、去重、排序
      - 替换 inf/-inf
      - 删除全空列
      - 尽量数值化、布尔统一
      - label→0/1
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} 缺少 'timestamp' 列")

    # 统一时间戳
    df["timestamp"] = as_int_ts(df["timestamp"])
    before_rows = len(df)
    df = df.dropna(subset=["timestamp"]).copy()
    if len(df) < before_rows:
        print(f"[{name}] 丢弃 timestamp 为 NaN 的行：{before_rows - len(df)}")

    # 去重 + 排序
    dup_cnt = df.duplicated(subset=["timestamp"]).sum()
    if dup_cnt > 0:
        print(f"[{name}] 去重重复 timestamp 行：{dup_cnt}")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 替换 inf，再按列删除全 NaN 列
    df = df.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"[{name}] 删除全空列（{len(all_nan_cols)}）：{all_nan_cols[:8]}{'...' if len(all_nan_cols)>8 else ''}")
        df = df.drop(columns=all_nan_cols)

    # 尝试类型统一（除 label）
    df = coerce_numeric_except_label(df)

    # label → 0/1
    df = normalize_label_column(df)

    # 再次删除清洗后变成全空的列
    all_nan_cols2 = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols2:
        print(f"[{name}] 再次删除全空列（{len(all_nan_cols2)}）：{all_nan_cols2[:8]}{'...' if len(all_nan_cols2)>8 else ''}")
        df = df.drop(columns=all_nan_cols2)

    return df

def main():
    base = Path(BASE_DIR)
    assert base.exists(), f"BASE_DIR 不存在：{base}"

    # 读取三份 CSV
    raw: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}

    for sub in SUBFOLDERS:
        folder = base / sub
        p = find_condensed_csv(folder)
        if p is None:
            raise FileNotFoundError(f"未在 {folder}/CSVs/Condensed 下找到任何 CSV")
        print(f"[LOAD] {sub}: {p}")
        df = pd.read_csv(p)
        raw[sub] = df
        paths[sub] = p

    # 清洗
    cleaned: Dict[str, pd.DataFrame] = {}
    for sub in SUBFOLDERS:
        cleaned[sub] = clean_df(raw[sub], name=sub)
        # 输出 label 分布便于核对
        if "label" in cleaned[sub].columns:
            vc = cleaned[sub]["label"].value_counts(dropna=False).to_dict()
            print(f"[LABEL] {sub} 分布：{vc}")
        print(f"[CLEAN] {sub}: 行数={len(cleaned[sub])}, 列数={cleaned[sub].shape[1]}")

    # 取三者共有列（列名交集）
    cols_sets = [set(cleaned[sub].columns) for sub in SUBFOLDERS]
    common_cols = set.intersection(*cols_sets)
    if not common_cols:
        raise RuntimeError("清洗后三者没有公共列。请检查上游合并或清洗规则。")

    # 统一列顺序：以 Benign Flight 的列顺序为基准（timestamp 放第一）
    benign_cols = list(cleaned["Benign Flight"].columns)
    ordered_common = [c for c in benign_cols if c in common_cols]
    if "timestamp" in ordered_common:
        ordered_common = ["timestamp"] + [c for c in ordered_common if c != "timestamp"]

    print(f"[COMMON] 共有列数={len(ordered_common)}，示例前20列={ordered_common[:20]}")

    # 输出目录：脚本同级的 preprocessed/
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path.cwd()
    out_dir = script_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 对每个数据集：只保留公共列，按一致顺序写出
    for sub in SUBFOLDERS:
        df = cleaned[sub].copy()
        # 防御性：确保所有公共列存在
        for c in ordered_common:
            if c not in df.columns:
                df[c] = np.nan
        df = df[ordered_common]

        out_path = out_dir / f"{sub}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[SAVE] {sub} → {out_path} 行数={len(df)}, 列数={df.shape[1]}")

if __name__ == "__main__":
    main()
