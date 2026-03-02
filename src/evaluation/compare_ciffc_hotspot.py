"""
compare_ciffc_hotspot.py
========================
对 CIFFC 里的每一条火灾记录，在 CWFIS 卫星热点（hotspot）数据中寻找
时空最近的匹配点，并量化两者的空间距离误差。

数据源说明
----------
CIFFC（Canadian Interagency Forest Fire Centre）:
    人工上报的火灾状态记录。每条记录 = 一次状态更新（一个 lat/lon 点代表整场火灾）。
    包含火灾面积（公顷）、控制状态、火因等丰富属性，时间精确到秒。

Hotspot（CWFIS / VIIRS 卫星热像素）:
    卫星自动探测的地表热异常像素（约 375m 分辨率）。
    每条记录 = 一个像素的探测，仅有坐标+日期，无面积/状态信息。
    每日数量远多于 CIFFC（~3,000 条/天 vs ~26 条/天）。

关键差异：
    1. 颗粒度   CIFFC 一条 = 整场火灾；hotspot 一条 = 375m 单个像素
    2. 面积     CIFFC 有 field_fire_size（ha）；hotspot 无
    3. 时间精度 CIFFC 精确到秒；hotspot 仅日期
    4. 数量     Hotspot 比 CIFFC 多约 100 倍
    5. 来源     CIFFC 人工；hotspot 卫星自动

Usage
-----
# 显式路径
python -m src.evaluation.compare_ciffc_hotspot \\
    --ciffc_csv    path/to/ciffc.csv \\
    --hotspot_csv  path/to/hotspot.csv \\
    --output_csv   ciffc_hotspot_match.csv \\
    --window_days  7 \\
    --match_km     10

# 使用 config（读取 paths.ciffc_csv / paths.hotspot_csv）
python -m src.evaluation.compare_ciffc_hotspot \\
    --config configs/default.yaml \\
    --window_days 7 --match_km 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── 项目内部工具（可选；脚本也支持不依赖 config 直接指定路径）──────────────
try:
    from src.config import add_config_argument, get_path, load_config
    from src.data_ops.processing.rasterize_fires import load_ciffc_data
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(
    lat1: float,
    lon1: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    从一个点 (lat1, lon1) 到一组点 (lats, lons) 的 Haversine 大圆距离，单位 km。

    Parameters
    ----------
    lat1, lon1 : float
        参考点（CIFFC 火灾位置）
    lats, lons : np.ndarray  shape (N,)
        候选 hotspot 坐标数组

    Returns
    -------
    np.ndarray  shape (N,)  每个候选点距参考点的距离（km）
    """
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _load_hotspot_raw(hotspot_path: str) -> pd.DataFrame:
    """
    加载 hotspot CSV，保留原始列名。
    返回列：latitude, longitude, acq_date (date object)
    """
    df = pd.read_csv(hotspot_path)
    df = df.dropna(subset=["acq_date"])
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    return df


def _load_ciffc_raw(ciffc_path: str) -> pd.DataFrame:
    """
    加载 CIFFC CSV，解析日期列。
    如果项目环境可用则复用 load_ciffc_data()，否则自行解析。
    """
    if _HAS_PROJECT:
        return load_ciffc_data(ciffc_path)
    df = pd.read_csv(ciffc_path)
    df["date"] = pd.to_datetime(df["field_situation_report_date"]).dt.date
    return df


def _build_hotspot_index(
    hotspot_df: pd.DataFrame,
) -> dict[date, np.ndarray]:
    """
    将 hotspot DataFrame 预建为日期索引字典。

    Returns
    -------
    dict  {date → np.ndarray shape (N, 2)}   每个元素 = [[lat, lon], ...]
    """
    print("  建立 hotspot 日期索引（一次性）...", flush=True)
    idx: dict[date, np.ndarray] = {}
    for d, g in hotspot_df.groupby("acq_date"):
        idx[d] = g[["latitude", "longitude"]].values.astype(np.float64)
    print(f"  索引建立完成，共 {len(idx):,} 个不同日期", flush=True)
    return idx


def _print_data_summary(ciffc_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> None:
    """打印两个数据源的内容对比摘要。"""
    cif_dates = pd.to_datetime(ciffc_df["field_situation_report_date"])
    hot_dates = hotspot_df["acq_date"]

    print()
    print("=" * 65)
    print("  数据源对比：CIFFC vs CWFIS Hotspot")
    print("=" * 65)

    print()
    print("【CIFFC — 人工上报火灾记录】")
    print(f"  记录数     : {len(ciffc_df):>10,} 条")
    print(f"  时间范围   : {cif_dates.min().date()} → {cif_dates.max().date()}")
    print(f"  列（12）   : field_agency_fire_id, field_agency_code,")
    print(f"               field_situation_report_date (精确到秒),")
    print(f"               field_stage_of_control_status, field_system_fire_cause,")
    print(f"               field_response_type,")
    print(f"               field_fire_size (公顷), field_latitude, field_longitude")
    if "field_fire_size" in ciffc_df.columns:
        sz = ciffc_df["field_fire_size"].dropna()
        print(f"  fire_size  : min={sz.min():.1f} ha, median={sz.median():.1f} ha, "
              f"max={sz.max():,.0f} ha")
    status_str = ", ".join(
        f"{k}:{v}"
        for k, v in ciffc_df["field_stage_of_control_status"].value_counts().items()
    ) if "field_stage_of_control_status" in ciffc_df.columns else "N/A"
    print(f"  控制状态   : {status_str}")
    print(f"  每条含义   : 一场火灾在某日的状态汇报，一个 lat/lon 代表整场火灾")

    print()
    print("【Hotspot — CWFIS 卫星热像素】")
    print(f"  记录数     : {len(hotspot_df):>10,} 条")
    print(f"  时间范围   : {hot_dates.min()} → {hot_dates.max()}")
    print(f"  列（3）    : latitude, longitude, acq_date (仅日期，无时刻)")
    lat_r = f"{hotspot_df['latitude'].min():.2f} → {hotspot_df['latitude'].max():.2f}"
    lon_r = f"{hotspot_df['longitude'].min():.2f} → {hotspot_df['longitude'].max():.2f}"
    print(f"  lat 范围   : {lat_r}")
    print(f"  lon 范围   : {lon_r}")
    print(f"  每条含义   : 卫星（VIIRS/MODIS ~375m）识别到的单个热像素")

    print()
    print("【关键差异】")
    rows = [
        ("数据来源",   "人工上报（省/联邦机构）",        "卫星自动探测（CWFIS/VIIRS）"),
        ("每条记录",   "整场火灾的一次状态汇报",         "一个 375m 卫星热像素"),
        ("面积信息",   "有（field_fire_size, 公顷）",   "无"),
        ("状态属性",   "有（OUT/UC/OC/BH/H）",         "无"),
        ("时间精度",   "秒（ISO 8601 时间戳）",          "日（YYYY-MM-DD）"),
        ("单日记录量", "~26 条（仅活跃火灾）",           "~3,000 条（全部热异常）"),
        ("数量级",    f"{len(ciffc_df):,} 条（2年）",   f"{len(hotspot_df):,} 条"),
    ]
    col1_w = max(len(r[0]) for r in rows) + 2
    for label, cif_val, hot_val in rows:
        print(f"  {label:<{col1_w}} CIFFC: {cif_val}")
        print(f"  {'':<{col1_w}}         Hotspot: {hot_val}")

    # 时间重叠检查
    cif_date_set = set(
        str(pd.to_datetime(ciffc_df["field_situation_report_date"]).dt.date)
        if False else  # pragma: no cover
        pd.to_datetime(ciffc_df["field_situation_report_date"]).dt.date.astype(str)
    )
    hot_date_set = set(hotspot_df["acq_date"].astype(str))
    overlap = cif_date_set & hot_date_set
    print()
    if overlap:
        print(f"  [OK] 时间重叠 : {len(overlap):,} 个共同日期（可进行匹配）")
    else:
        print(f"  [WARN] 时间无重叠：CIFFC {cif_dates.min().year}–{cif_dates.max().year}，"
              f"Hotspot {hot_dates.min()}–{hot_dates.max()}")
        print(f"     -> 用服务器上覆盖 2023–2025 的完整 hotspot 文件运行才能看到真实匹配率")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 核心匹配函数
# ─────────────────────────────────────────────────────────────────────────────

def match_ciffc_to_hotspot(
    ciffc_df: pd.DataFrame,
    hotspot_idx: dict[date, np.ndarray],
    window_days: int,
    match_km: float,
    date_field: str = "situation",
) -> pd.DataFrame:
    """
    对 ciffc_df 每一行，在 hotspot_idx 中查找最近的热点。

    Parameters
    ----------
    ciffc_df     : CIFFC DataFrame（含 date / field_latitude / field_longitude）
    hotspot_idx  : {date → np.ndarray(N,2)} 日期索引
    window_days  : 时间窗口 ±N 天
    match_km     : "已匹配"判定阈值（km）
    date_field   : "situation"=field_situation_report_date, "status"=field_status_date

    Returns
    -------
    DataFrame：原始 CIFFC 列 + 新增匹配列
    """
    # 确定使用哪个日期列
    if date_field == "status" and "field_status_date" in ciffc_df.columns:
        ciffc_df = ciffc_df.copy()
        ciffc_df["date"] = pd.to_datetime(
            ciffc_df["field_status_date"]
        ).dt.date
    # 若 load_ciffc_data 已解析 date 则直接用
    dates = ciffc_df["date"].values
    lats  = ciffc_df["field_latitude"].values.astype(float)
    lons  = ciffc_df["field_longitude"].values.astype(float)

    n = len(ciffc_df)
    # 结果列（预分配 NaN）
    same_day_nearest_km        = np.full(n, np.nan)
    window_nearest_km          = np.full(n, np.nan)
    window_nearest_day_offset  = np.full(n, np.nan)
    nearest_hotspot_lat        = np.full(n, np.nan)
    nearest_hotspot_lon        = np.full(n, np.nan)
    no_hotspots_in_window      = np.ones(n, dtype=bool)

    print(f"[匹配中] {n:,} 条 CIFFC 记录，时间窗口 ±{window_days} 天，"
          f"匹配阈值 {match_km} km ...", flush=True)

    for i in range(n):
        if i % 500 == 0 and i > 0:
            pct = 100 * i / n
            print(f"  {i:,}/{n:,} ({pct:.0f}%)...", flush=True)

        ciffc_date = dates[i]
        lat1 = lats[i]
        lon1 = lons[i]

        # --- 收集窗口内所有 hotspot 点 ---
        window_pts_list:  list[np.ndarray] = []
        day_offsets_list: list[np.ndarray] = []

        for offset in range(-window_days, window_days + 1):
            d = ciffc_date + timedelta(days=offset)
            pts = hotspot_idx.get(d)
            if pts is not None and len(pts) > 0:
                window_pts_list.append(pts)
                day_offsets_list.append(np.full(len(pts), offset, dtype=np.int32))

        if not window_pts_list:
            # 时间窗口内完全没有 hotspot（文件不覆盖该时段）
            continue

        no_hotspots_in_window[i] = False

        # --- 同日匹配 ---
        same_day_pts = hotspot_idx.get(ciffc_date)
        if same_day_pts is not None and len(same_day_pts) > 0:
            dists = _haversine_km(lat1, lon1, same_day_pts[:, 0], same_day_pts[:, 1])
            same_day_nearest_km[i] = float(dists.min())

        # --- 窗口最近匹配 ---
        combined  = np.vstack(window_pts_list)    # (M, 2)
        offsets   = np.concatenate(day_offsets_list)  # (M,)
        dists_all = _haversine_km(lat1, lon1, combined[:, 0], combined[:, 1])
        best_idx  = int(dists_all.argmin())
        window_nearest_km[i]         = float(dists_all[best_idx])
        window_nearest_day_offset[i] = int(offsets[best_idx])
        nearest_hotspot_lat[i]       = float(combined[best_idx, 0])
        nearest_hotspot_lon[i]       = float(combined[best_idx, 1])

    print(f"  Matching done.", flush=True)

    # 拼接结果
    out = ciffc_df.copy()
    out["ciffc_date"]                = [str(d) for d in dates]
    out["no_hotspots_in_window"]     = no_hotspots_in_window
    out["same_day_nearest_km"]       = same_day_nearest_km
    out["window_nearest_km"]         = window_nearest_km
    out["window_nearest_day_offset"] = window_nearest_day_offset  # float; NaN where no hotspot in window
    out["nearest_hotspot_lat"]       = nearest_hotspot_lat
    out["nearest_hotspot_lon"]       = nearest_hotspot_lon
    out["matched_same_day"]          = (
        ~np.isnan(same_day_nearest_km) & (same_day_nearest_km < match_km)
    )
    out["matched_window"]            = (
        ~np.isnan(window_nearest_km) & (window_nearest_km < match_km)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 统计摘要
# ─────────────────────────────────────────────────────────────────────────────

def _print_match_summary(out: pd.DataFrame, window_days: int, match_km: float) -> None:
    n = len(out)
    no_win  = out["no_hotspots_in_window"].sum()
    has_win = n - no_win

    same_day_has = out["same_day_nearest_km"].notna().sum()
    matched_sd   = out["matched_same_day"].sum()
    matched_win  = out["matched_window"].sum()

    print()
    print("=" * 65)
    print(f"  匹配统计 (window=±{window_days}天, threshold={match_km}km)")
    print("=" * 65)
    print(f"  CIFFC 记录总数         : {n:>8,}")
    print(f"  窗口内无 hotspot       : {no_win:>8,}  ({100*no_win/n:.1f}%)")
    print(f"  窗口内有 hotspot       : {has_win:>8,}  ({100*has_win/n:.1f}%)")
    print(f"  同日有 hotspot         : {same_day_has:>8,}  ({100*same_day_has/n:.1f}%)")
    print(f"  同日匹配 (<{match_km:.0f}km)    : {matched_sd:>8,}  ({100*matched_sd/n:.1f}%)")
    print(f"  窗口内匹配 (<{match_km:.0f}km)  : {matched_win:>8,}  ({100*matched_win/n:.1f}%)")

    # 距离分布（只有有数据时才打印）
    wkm = out["window_nearest_km"].dropna()
    if len(wkm) > 0:
        print()
        print("  窗口内最近 hotspot 距离分布（km）：")
        for pct, val in [
            (0,   wkm.min()),
            (10,  wkm.quantile(0.10)),
            (25,  wkm.quantile(0.25)),
            (50,  wkm.median()),
            (75,  wkm.quantile(0.75)),
            (90,  wkm.quantile(0.90)),
            (100, wkm.max()),
        ]:
            print(f"    P{pct:>3}  {val:>10.2f} km")

    doff = out["window_nearest_day_offset"].dropna()
    if len(doff) > 0:
        print()
        print("  最近 hotspot 日期偏移分布（天，负=hotspot 早于 CIFFC）：")
        for pct, val in [
            (0,   doff.min()),
            (25,  doff.quantile(0.25)),
            (50,  doff.median()),
            (75,  doff.quantile(0.75)),
            (100, doff.max()),
        ]:
            print(f"    P{pct:>3}  {val:>+8.0f} 天")

    if no_win == n:
        print()
        print("  [WARN] 所有 CIFFC 记录在 hotspot 文件时间范围内均无数据。")
        print("     -> 请在服务器使用覆盖 2023–2025 年的完整 hotspot 文件重新运行。")
    print("=" * 65)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="对 CIFFC 每条记录在 CWFIS hotspot 中查找最近匹配点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    if _HAS_PROJECT:
        add_config_argument(ap)
    ap.add_argument("--ciffc_csv",   type=str, default=None,
                    help="CIFFC CSV 路径（覆盖 config 中的 ciffc_csv）")
    ap.add_argument("--hotspot_csv", type=str, default=None,
                    help="Hotspot CSV 路径（覆盖 config 中的 hotspot_csv）")
    ap.add_argument("--output_csv",  type=str, default="ciffc_hotspot_match.csv",
                    help="输出 CSV 路径（默认 ciffc_hotspot_match.csv）")
    ap.add_argument("--window_days", type=int, default=7,
                    help="时间窗口 ±N 天（默认 7）")
    ap.add_argument("--match_km",    type=float, default=10.0,
                    help="判定匹配的最大距离 km（默认 10.0）")
    ap.add_argument("--date_field",  choices=["situation", "status"],
                    default="situation",
                    help="使用 field_situation_report_date(situation) 或 "
                         "field_status_date(status)（默认 situation）")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # ── 解析路径 ────────────────────────────────────────────────────────────
    ciffc_path   = args.ciffc_csv
    hotspot_path = args.hotspot_csv

    if _HAS_PROJECT and hasattr(args, "config"):
        cfg = load_config(args.config)
        if ciffc_path is None:
            ciffc_path = get_path(cfg, "ciffc_csv")
        if hotspot_path is None:
            hotspot_path = get_path(cfg, "hotspot_csv")

    if not ciffc_path or not hotspot_path:
        sys.exit(
            "错误：请通过 --ciffc_csv / --hotspot_csv 或 --config 指定数据路径。"
        )

    ciffc_path   = str(ciffc_path)
    hotspot_path = str(hotspot_path)

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 1] 加载 CIFFC  ← {ciffc_path}")
    ciffc_df = _load_ciffc_raw(ciffc_path)
    print(f"  {len(ciffc_df):,} 条记录")

    print(f"\n[STEP 2] 加载 Hotspot ← {hotspot_path}")
    hotspot_df = _load_hotspot_raw(hotspot_path)
    print(f"  {len(hotspot_df):,} 条记录")

    # ── 数据对比摘要 ─────────────────────────────────────────────────────────
    _print_data_summary(ciffc_df, hotspot_df)

    # ── 建立 hotspot 日期索引 ─────────────────────────────────────────────────
    print("[STEP 3] 建立 hotspot 日期索引...")
    hotspot_idx = _build_hotspot_index(hotspot_df)

    # ── 逐条匹配 ─────────────────────────────────────────────────────────────
    print("\n[STEP 4] 开始匹配...")
    out_df = match_ciffc_to_hotspot(
        ciffc_df     = ciffc_df,
        hotspot_idx  = hotspot_idx,
        window_days  = args.window_days,
        match_km     = args.match_km,
        date_field   = args.date_field,
    )

    # ── 统计摘要 ─────────────────────────────────────────────────────────────
    _print_match_summary(out_df, args.window_days, args.match_km)

    # ── 保存输出 CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)
    print(f"[完成] 结果已保存到：{out_path.resolve()}")
    print(f"  共 {len(out_df):,} 行，新增列：")
    new_cols = [
        "ciffc_date", "no_hotspots_in_window",
        "same_day_nearest_km", "window_nearest_km",
        "window_nearest_day_offset",
        "nearest_hotspot_lat", "nearest_hotspot_lon",
        "matched_same_day", "matched_window",
    ]
    for c in new_cols:
        print(f"    {c}")
    print()


if __name__ == "__main__":
    main()
