# -*- coding: utf-8 -*-
"""
alfa_merge_flights.py  (numeric-key version: avoid Timedelta OOB)

- 全程使用浮点秒 t 做 merge_asof 键，不再把 t 转为 Timedelta，规避 OutOfBoundsDatetime
- 每轮合并前显式按 t 排序，避免 "keys must be sorted"
- 消除了 to_numeric FutureWarning 的用法
"""

import argparse
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # silence copy warnings


# -------------------------------
# Topic -> columns whitelist (used when --keep_all_columns is OFF)
# -------------------------------

KNOWN_TOPICS = {
    r"failure_status": [
        "%time", "field.data", "data", "value", "status", "state", "field.state", "field.value"
    ],
    r"nav_info-airspeed": [
        "field.commanded", "field.measured", "commanded", "measured", "airspeed",
    ],
    r"nav_info-roll": [
        "field.commanded", "field.measured", "roll_commanded", "roll_measured",
    ],
    r"nav_info-pitch": [
        "field.commanded", "field.measured", "pitch_commanded", "pitch_measured",
    ],
    r"nav_info-yaw": [
        "field.commanded", "field.measured", "yaw_commanded", "yaw_measured",
    ],
    r"nav_info-velocity": [
        "field.vx", "field.vy", "field.vz", "vx", "vy", "vz", "field.measured", "field.commanded"
    ],
    r"nav_info-errors": [
        "field.alt_error", "field.aspd_error", "field.xtrack_error",
        "alt_error", "aspd_error", "xtrack_error",
    ],
    r"imu-data": [
        "field.orientation.x", "field.orientation.y", "field.orientation.z", "field.orientation.w",
        "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z",
        "field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
        "orientation.x", "orientation.y", "orientation.z", "orientation.w",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "field.temperature", "temperature"
    ],
    r"imu-data_raw": [
        "field.raw_angular_velocity.x", "field.raw_angular_velocity.y", "field.raw_angular_velocity.z",
        "field.raw_linear_acceleration.x", "field.raw_linear_acceleration.y", "field.raw_linear_acceleration.z",
        "raw_angular_velocity.x", "raw_angular_velocity.y", "raw_angular_velocity.z",
        "raw_linear_acceleration.x", "raw_linear_acceleration.y", "raw_linear_acceleration.z",
    ],
    r"imu-atm_pressure": [
        "field.fluid_pressure", "field.variance", "fluid_pressure", "variance",
    ],
    r"imu-mag": [
        "field.magnetic_field.x", "field.magnetic_field.y", "field.magnetic_field.z",
        "magnetic_field.x", "magnetic_field.y", "magnetic_field.z",
    ],
    r"imu-temperature": [
        "field.temperature", "temperature",
    ],
    r"local_position-velocity": [
        "field.twist.angular.x", "field.twist.angular.y", "field.twist.angular.z",
        "field.twist.linear.x",  "field.twist.linear.y",  "field.twist.linear.z",
        "twist.angular.x", "twist.angular.y", "twist.angular.z",
        "twist.linear.x", "twist.linear.y", "twist.linear.z",
    ],
    r"local_position-pose": [
        "field.pose.position.x", "field.pose.position.y", "field.pose.position.z",
        "field.pose.orientation.x", "field.pose.orientation.y", "field.pose.orientation.z", "field.pose.orientation.w",
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w",
    ],
    r"local_position-odom": [
        "field.pose.pose.position.x", "field.pose.pose.position.y", "field.pose.pose.position.z",
        "field.twist.twist.linear.x", "field.twist.twist.linear.y", "field.twist.twist.linear.z",
        "field.twist.twist.angular.x", "field.twist.twist.angular.y", "field.twist.twist.angular.z",
    ],
    r"global_position-global": [
        "field.latitude", "field.longitude", "field.altitude", "latitude", "longitude", "altitude",
    ],
    r"global_position-local": [
        "field.pose.position.x", "field.pose.position.y", "field.pose.position.z",
        "pose.position.x", "pose.position.y", "pose.position.z",
    ],
    r"global_position-rel_alt": [
        "field.data", "data", "rel_alt"
    ],
    r"global_position-compass_hdg": [
        "field.data", "data", "heading", "compass_hdg"
    ],
    r"global_position-raw-fix": [
        "field.status.status", "field.latitude", "field.longitude", "field.altitude",
        "status.status", "latitude", "longitude", "altitude",
    ],
    r"global_position-raw-gps_vel": [
        "field.twist.linear.x", "field.twist.linear.y", "field.twist.linear.z",
        "twist.linear.x", "twist.linear.y", "twist.linear.z",
    ],
    r"battery": [
        "field.voltage", "field.current", "field.percentage", "field.remaining",
        "voltage", "current", "percentage", "remaining",
    ],
    r"wind_estimation": [
        "field.wind_speed", "field.wind_direction", "wind_speed", "wind_direction",
    ],
    r"rc-in": [
        "field.channels.0","field.channels.1","field.channels.2","field.channels.3",
        "field.channels.4","field.channels.5","field.channels.6","field.channels.7",
        "channels.0","channels.1","channels.2","channels.3","channels.4","channels.5","channels.6","channels.7",
    ],
    r"rc-out": [
        "field.channels.0","field.channels.1","field.channels.2","field.channels.3",
        "field.channels.4","field.channels.5","field.channels.6","field.channels.7",
        "channels.0","channels.1","channels.2","channels.3","channels.4","channels.5","channels.6","channels.7",
    ],
    r"setpoint_raw-local": [
        "field.position.x","field.position.y","field.position.z",
        "field.velocity.x","field.velocity.y","field.velocity.z",
        "position.x","position.y","position.z","velocity.x","velocity.y","velocity.z",
    ],
    r"setpoint_raw-target_global": [
        "field.latitude","field.longitude","field.altitude","latitude","longitude","altitude",
    ],
    r"state": [
        "field.armed","field.guided","field.mode","armed","guided","mode"
    ],
    r"time_reference": [
        "field.time_ref","field.source","time_ref","source"
    ],
    r"vfr_hud": [
        "field.airspeed","field.groundspeed","field.throttle","field.heading","field.climb",
        "airspeed","groundspeed","throttle","heading","climb",
    ],
    r"mavctrl-path_dev": [
        "field.xtrack_error","field.alt_error","xtrack_error","alt_error"
    ],
    r"mavctrl-rpy": [
        "field.roll","field.pitch","field.yaw","roll","pitch","yaw"
    ],
    r"diagnostics": [
        "level","hardware_id","name","message","values"
    ],
    r"mavlink-from": [
        # keep only time by default unless keep_all_columns
    ],
    r"mission-reached": [
        "field.seq","seq"
    ],
    r"emergency_responder-traj_file": [
        "field.des_x","field.des_y","field.des_z","field.meas_x","field.meas_y","field.meas_z",
        "des_x","des_y","des_z","meas_x","meas_y","meas_z",
    ],
}

TIME_CANDIDATES = [
    "%time", "time", "timestamp", "rosbagTimestamp",
    "header.stamp", "header.stamp.secs", "header.stamp.nsecs",
    "stamp.secs", "stamp.nsecs", "secs", "nsecs",
    "TimeUS", "time_boot_ms",
]


def _coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def build_time_columns(df: pd.DataFrame):
    cols = set(df.columns)
    if "%time" in cols:
        t_abs = _coerce_numeric(df["%time"])
        t = t_abs - t_abs.iloc[0]
        return t.astype(float).values, t_abs.astype(float).values

    pairs = [
        ("header.stamp.secs","header.stamp.nsecs"),
        ("stamp.secs","stamp.nsecs"),
        ("secs","nsecs"),
        ("sec","nsec"),
    ]
    for s_col, ns_col in pairs:
        if s_col in cols and ns_col in cols:
            secs = _coerce_numeric(df[s_col])
            nsecs = _coerce_numeric(df[ns_col]).fillna(0)
            t_abs = secs.astype(float) + nsecs.astype(float) * 1e-9
            t = t_abs - t_abs.iloc[0]
            return t.values, t_abs.values

    if "TimeUS" in cols:
        t_abs = _coerce_numeric(df["TimeUS"]) * 1e-6
        t = t_abs - t_abs.iloc[0]
        return t.values, t_abs.values

    if "time_boot_ms" in cols:
        t_rel = _coerce_numeric(df["time_boot_ms"]) * 1e-3
        t = t_rel - t_rel.iloc[0]
        return t.values, None

    if "rosbagTimestamp" in cols:
        t_abs = _coerce_numeric(df["rosbagTimestamp"])
        t = t_abs - t_abs.iloc[0]
        return t.values, t_abs.values

    if "time" in cols:
        t_abs = _coerce_numeric(df["time"])
        if t_abs.max() > 1e6:
            t = t_abs - t_abs.iloc[0]
            return t.values, t_abs.values
        else:
            t = t_abs
            return t.values, None

    # fallback: artificial 100 Hz counter
    n = len(df)
    t = np.arange(n, dtype=float) / 100.0
    return t, None


def clean_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")]


def pick_columns(df: pd.DataFrame, topic_key: str, keep_all: bool) -> pd.DataFrame:
    if keep_all:
        return df
    chosen = []
    topic_key_lower = topic_key.lower()
    for pattern, cols in KNOWN_TOPICS.items():
        if re.search(pattern, topic_key_lower):
            for c in cols:
                if c in df.columns:
                    chosen.append(c)
            for tcol in TIME_CANDIDATES:
                if tcol in df.columns and tcol not in chosen:
                    chosen.append(tcol)
            break
    if not chosen:
        for tcol in TIME_CANDIDATES:
            if tcol in df.columns:
                chosen.append(tcol)
    if not chosen:
        return df
    return df.loc[:, chosen]


def add_prefix(df: pd.DataFrame, prefix: str, exclude=("t","t_abs")) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        if c in exclude:
            continue
        rename_map[c] = f"{prefix}.{c}"
    return df.rename(columns=rename_map)


def read_topic_csv(csv_path: Path, topic_key: str, keep_all: bool, verbose: bool=False) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to read {csv_path}: {e}")
        return None
    if df.empty:
        return None

    df = clean_unnamed_columns(df)
    df = pick_columns(df, topic_key, keep_all)

    t, t_abs = build_time_columns(df)
    df.insert(0, "t", t.astype(float))
    if t_abs is not None and "t_abs" not in df.columns:
        df.insert(1, "t_abs", t_abs.astype(float))

    # 尝试把 object 列转数值（失败就保留原值）
    for c in df.columns:
        if c in ("t","t_abs"):
            continue
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

    df = df.sort_values("t").drop_duplicates(subset=["t"])
    df = add_prefix(df, topic_key, exclude=("t","t_abs"))
    return df


def discover_flight_dirs(input_root: Path, verbose: bool=False):
    flights = []
    for d in sorted(p for p in input_root.iterdir() if p.is_dir()):
        if any(f.suffix.lower()==".csv" for f in d.glob("*.csv")):
            flights.append(d)
    if not flights:
        for d in sorted(input_root.rglob("*")):
            if d.is_dir() and any(f.suffix.lower()==".csv" for f in d.glob("*.csv")):
                flights.append(d)
    seen = set()
    unique = []
    for d in flights:
        if d.as_posix() not in seen:
            seen.add(d.as_posix())
            unique.append(d)
    if verbose:
        print(f"[INFO] Found {len(unique)} flight directories under: {input_root}")
    return unique


def topic_key_from_filename(flight_name: str, file_path: Path) -> str:
    stem = file_path.stem
    if stem.startswith(flight_name + "-"):
        return stem[len(flight_name)+1:]
    return stem


def merge_topics_for_flight(flight_dir: Path, resample_hz: float, tolerance_ms: float,
                            keep_all_cols: bool, compute_deltas: bool, verbose: bool=False) -> pd.DataFrame:
    flight_name = flight_dir.name
    csv_files = sorted(f for f in flight_dir.glob("*.csv"))
    if not csv_files:
        if verbose:
            print(f"[WARN] No CSV in {flight_dir}")
        return None

    topic_dfs = []
    failure_cols = []

    for f in csv_files:
        topic_key = topic_key_from_filename(flight_name, f)
        df = read_topic_csv(f, topic_key, keep_all_cols, verbose=verbose)
        if df is None or df.empty:
            continue
        topic_dfs.append(df)

        if topic_key.startswith("failure_status"):
            cand = [c for c in df.columns if c.endswith(".field.data") or c.endswith(".data")
                    or c.endswith(".value") or c.endswith(".state")]
            failure_cols.extend(cand)

    if not topic_dfs:
        if verbose:
            print(f"[WARN] No valid topic DataFrames in {flight_dir}")
        return None

    # Base time axis
    if resample_hz and resample_hz > 0:
        t_min = min(df["t"].iloc[0] for df in topic_dfs)
        t_max = max(df["t"].iloc[-1] for df in topic_dfs)
        dt = 1.0 / float(resample_hz)
        base_t = np.arange(t_min, t_max + dt/2.0, dt)
        base = pd.DataFrame({"t": base_t})
    else:
        all_t = np.concatenate([df["t"].values for df in topic_dfs])
        base_t = np.unique(np.round(all_t, 9))
        base = pd.DataFrame({"t": np.sort(base_t)})

    # Attach t_abs if any (numeric asof on t)
    abs_sources = [df[["t","t_abs"]] for df in topic_dfs if "t_abs" in df.columns]
    tol_sec = float(tolerance_ms)/1000.0 if tolerance_ms is not None else None
    if abs_sources:
        abs_df = (pd.concat(abs_sources, ignore_index=True)
                    .dropna(subset=["t"])
                    .drop_duplicates(subset=["t"])
                    .sort_values("t"))
        kwargs = dict(on="t", direction="nearest")
        if tol_sec is not None:
            kwargs["tolerance"] = tol_sec
        base = pd.merge_asof(
            base.sort_values("t"),
            abs_df.sort_values("t")[["t","t_abs"]],
            **kwargs
        )
    else:
        base["t_abs"] = np.nan

    merged = base.sort_values("t").reset_index(drop=True)

    # Progressive numeric asof using 't'
    for df in topic_dfs:
        right = df.copy()
        # 避免 t_abs_x/y 混乱：右侧不再提供 t_abs
        right = right.drop(columns=["t_abs"], errors="ignore")

        kwargs = dict(on="t", direction="nearest")
        if tol_sec is not None:
            kwargs["tolerance"] = tol_sec

        merged = pd.merge_asof(
            merged.sort_values("t"),
            right.sort_values("t"),
            **kwargs
        )

    # failure_status.any
    if failure_cols:
        sub_cols = [c for c in failure_cols if c in merged.columns]
        if sub_cols:
            sub = merged[sub_cols].copy()
            for c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            merged["failure_status.any"] = (sub.fillna(0) != 0).any(axis=1).astype(int)
        else:
            merged["failure_status.any"] = 0
    else:
        merged["failure_status.any"] = 0

    if compute_deltas:
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("t","t_abs")]
        dt = merged["t"].diff().values
        dt[dt == 0] = np.nan
        for c in numeric_cols:
            merged[c + "_dot"] = merged[c].diff().values / dt

    return merged.sort_values("t").reset_index(drop=True)


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Merge ALFA flight topics into one raw CSV per flight (with failure_status).")
    parser.add_argument("--input_root", type=str, default="ALFA/processed",
                        help="Root directory containing flight subfolders (default: ALFA/processed)")
    parser.add_argument("--output_dir", type=str, default="ALFA_merged_raw",
                        help="Where to save merged CSVs (default: ALFA_merged_raw)")
    parser.add_argument("--resample_hz", type=float, default=0.0,
                        help="If >0, resample base timeline at this Hz before asof matching (default 0 = raw union)")
    parser.add_argument("--tolerance_ms", type=float, default=50.0,
                        help="merge_asof nearest-neighbor tolerance in milliseconds (default 50 ms)")
    parser.add_argument("--keep_all_columns", action="store_true",
                        help="If set, keep ALL columns from each topic CSV (otherwise use KNOWN_TOPICS whitelist).")
    parser.add_argument("--compute_deltas", action="store_true",
                        help="If set, compute first derivative (_dot) for numeric columns.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if not input_root.exists():
        print(f"[ERR] Input root does not exist: {input_root}")
        sys.exit(1)

    flights = discover_flight_dirs(input_root, verbose=args.verbose)
    if not flights:
        print(f"[ERR] No flight directories found under: {input_root}")
        sys.exit(1)

    print(f"[INFO] Will process {len(flights)} flights from: {input_root}")
    for i, fdir in enumerate(flights, 1):
        print(f"[{i}/{len(flights)}] Merging flight: {fdir.name}")
        merged = merge_topics_for_flight(
            fdir,
            resample_hz=args.resample_hz,
            tolerance_ms=args.tolerance_ms,
            keep_all_cols=args.keep_all_columns,
            compute_deltas=args.compute_deltas,
            verbose=args.verbose
        )
        if merged is None or merged.empty:
            print(f"   [WARN] Skipped (no data): {fdir.name}")
            continue

        out_path = output_dir / f"{fdir.name}.merged.raw.csv"
        merged.to_csv(out_path, index=False)
        print(f"   [OK] Saved: {out_path}")

    print("[DONE] All flights processed.")


if __name__ == "__main__":
    main()
