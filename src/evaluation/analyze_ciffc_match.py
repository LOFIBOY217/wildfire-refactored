"""
analyze_ciffc_match.py
======================
对 compare_ciffc_hotspot.py 的输出进行深度分析，回答三个问题：

  1. 总体探测率：CIFFC 里的火，卫星有没有看到？
  2. 误差量化：看到的那些，时间差多少？空间差多少？
  3. 未探测特征：没看到的那些火，有什么共同特点？

Usage:
    python -m src.evaluation.analyze_ciffc_match ^
        --match_csv ciffc_hotspot_match.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, total: float = 100.0, width: int = 25) -> str:
    """生成 ASCII 进度条，value/total 映射到 width 个字符。"""
    filled = int(round(width * value / total)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{n:>7,}  ({100*n/total:>5.1f}%)" if total > 0 else f"{n:>7,}  ( N/A )"


def _section(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _subsection(title: str) -> None:
    print()
    print(f"  ── {title}")
    print(f"  {'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# 分析函数
# ─────────────────────────────────────────────────────────────────────────────

def section1_detection_rate(df: pd.DataFrame) -> None:
    """【1】总体探测率：卫星有没有看到这场火？"""
    _section("1. 总体探测率")

    n          = len(df)
    detected   = df[~df["no_hotspots_in_window"]]    # 窗口内有 hotspot
    undetected = df[df["no_hotspots_in_window"]]      # 窗口内无 hotspot
    n_det  = len(detected)
    n_und  = len(undetected)

    print(f"\n  CIFFC 总记录数            : {n:>7,}")
    print(f"  ✅ 卫星有探测（±7天窗口） : {_pct(n_det, n)}   {_bar(n_det, n)}")
    print(f"  ❌ 卫星无探测             : {_pct(n_und, n)}   {_bar(n_und, n)}")

    # 进一步拆分"有探测"里的同日探测
    same_day = (~df["same_day_nearest_km"].isna()).sum()
    print(f"\n  其中，同日即有卫星记录    : {_pct(same_day, n)}")


def section2_time_error(df: pd.DataFrame) -> None:
    """【2】时间误差：卫星和人工记录差了几天？"""
    _section("2. 时间误差（有卫星探测的记录）")

    det = df[~df["no_hotspots_in_window"]].copy()
    offsets = det["window_nearest_day_offset"].dropna()

    if len(offsets) == 0:
        print("\n  无数据")
        return

    print(f"\n  样本量：{len(offsets):,} 条（窗口内有 hotspot 的记录）")
    print(f"\n  偏移 = hotspot日期 − CIFFC日期（负 = 卫星先于人工上报）\n")

    # 逐天分布（-7 到 +7）
    print(f"  {'偏移天数':>6}  {'记录数':>8}  {'占比':>6}  分布")
    print(f"  {'─'*55}")
    for d in range(-7, 8):
        cnt = int((offsets == d).sum())
        pct = 100 * cnt / len(offsets)
        bar = _bar(pct, 20.0, 20)
        tag = " ← 同日" if d == 0 else ""
        print(f"  {d:>+6}天  {cnt:>8,}  {pct:>5.1f}%  {bar}{tag}")

    print(f"\n  分位数摘要：")
    for q, label in [(0, "最小"), (0.25, "P25"), (0.5, "中位"), (0.75, "P75"), (1.0, "最大")]:
        print(f"    {label} : {offsets.quantile(q):>+.0f} 天")


def section3_distance_error(df: pd.DataFrame) -> None:
    """【3】距离误差：按火灾面积分组，误差有多大？"""
    _section("3. 距离误差（按火灾面积分组）")

    det = df[~df["no_hotspots_in_window"]].copy()
    det = det[det["window_nearest_km"].notna()]

    if len(det) == 0:
        print("\n  无数据")
        return

    # 去掉明显异常值（>5000 km，不可能的距离）
    n_outlier = (det["window_nearest_km"] > 5000).sum()
    if n_outlier > 0:
        print(f"\n  ⚠  去除 {n_outlier} 条距离 >5000km 的异常记录（坐标明显有误）")
        det = det[det["window_nearest_km"] <= 5000]

    print(f"\n  逻辑：火灾面积越大，CIFFC 汇报点与卫星热点的距离自然越远")
    print(f"  （CIFFC 记录一个点，卫星记录整片燃烧区域）\n")

    # 面积分组
    bins   = [0, 1, 10, 100, 1_000, 10_000, float("inf")]
    labels = ["极小 <1ha", "小 1–10ha", "中 10–100ha",
              "大 100–1000ha", "超大 1000–10000ha", "巨大 >10000ha"]

    if "field_fire_size" not in det.columns:
        print("  （缺少 field_fire_size 列，跳过分组分析）")
        _print_dist(det["window_nearest_km"], "全部")
        return

    det["_size_grp"] = pd.cut(
        det["field_fire_size"].fillna(0).clip(lower=0),
        bins=bins, labels=labels, right=False
    )

    print(f"  {'面积分组':<18} {'记录数':>6}  {'中位距离':>8}  "
          f"{'<10km':>6}  {'<50km':>7}  {'<100km':>7}")
    print(f"  {'─'*70}")

    for grp in labels:
        sub = det[det["_size_grp"] == grp]["window_nearest_km"]
        if len(sub) == 0:
            continue
        med    = sub.median()
        lt10   = 100 * (sub < 10).mean()
        lt50   = 100 * (sub < 50).mean()
        lt100  = 100 * (sub < 100).mean()
        print(f"  {grp:<18} {len(sub):>6,}  {med:>7.1f}km  "
              f"{lt10:>5.1f}%  {lt50:>6.1f}%  {lt100:>6.1f}%")

    # 全体分位数
    print(f"\n  全体距离分位数（去异常后 {len(det):,} 条）：")
    for q, label in [(0, "P0（最小）"), (0.1, "P10"), (0.25, "P25"),
                     (0.5, "P50（中位）"), (0.75, "P75"), (0.9, "P90"), (1.0, "P100（最大）")]:
        print(f"    {label:<12} : {det['window_nearest_km'].quantile(q):>8.2f} km")


def _print_dist(series: pd.Series, label: str) -> None:
    print(f"\n  {label} 距离分布：")
    for q, ql in [(0, "P0"), (0.25, "P25"), (0.5, "P50"), (0.75, "P75"), (1.0, "P100")]:
        print(f"    {ql}: {series.quantile(q):.2f} km")


def section4_undetected_profile(df: pd.DataFrame) -> None:
    """【4】未探测记录的特征：没被卫星看到的火有什么规律？"""
    _section("4. 未被卫星探测到的火灾特征分析")

    det = df[~df["no_hotspots_in_window"]]
    und = df[df["no_hotspots_in_window"]]
    n_und = len(und)
    n_det = len(det)

    if n_und == 0:
        print("\n  所有记录都有卫星探测，无需分析。")
        return

    print(f"\n  分析对象：{n_und:,} 条无卫星探测的记录\n")

    # ── 4a. 火灾面积 ──────────────────────────────────────────────────
    _subsection("a. 火灾面积（公顷）")

    if "field_fire_size" in df.columns:
        und_sz = und["field_fire_size"].dropna()
        det_sz = det["field_fire_size"].dropna()

        print(f"  {'指标':<20} {'未探测':>12}  {'已探测':>12}")
        print(f"  {'─'*48}")
        for label, q in [("最小值", 0), ("P10", 0.1), ("中位数", 0.5),
                          ("P90", 0.9), ("最大值", 1.0)]:
            u = und_sz.quantile(q) if len(und_sz) > 0 else float("nan")
            d = det_sz.quantile(q) if len(det_sz) > 0 else float("nan")
            print(f"  {label:<20} {u:>12.2f}  {d:>12.2f}")

        # 极小火（<1ha）占比
        u_tiny = 100 * (und_sz < 1).mean() if len(und_sz) > 0 else 0
        d_tiny = 100 * (det_sz < 1).mean() if len(det_sz) > 0 else 0
        print(f"\n  面积 < 1 ha 占比：未探测 {u_tiny:.1f}% vs 已探测 {d_tiny:.1f}%")
        print(f"  → 小火更难被卫星探测，这符合预期（卫星分辨率约 375m）")

    # ── 4b. 控制状态 ──────────────────────────────────────────────────
    _subsection("b. 控制状态（火灾是否仍在燃烧？）")

    status_col = "field_stage_of_control_status"
    if status_col in df.columns:
        status_map = {
            "OUT": "OUT（已扑灭）",
            "BH":  "BH（受控中）",
            "UC":  "UC（失控）",
            "OC":  "OC（观察中）",
            "H":   "H（控制）",
        }
        print(f"\n  {'状态':<18} {'未探测数':>9}  {'未探测%':>8}  {'已探测数':>9}  {'已探测%':>8}")
        print(f"  {'─'*60}")

        all_statuses = df[status_col].dropna().unique()
        for s in sorted(all_statuses):
            u = (und[status_col] == s).sum()
            d = (det[status_col] == s).sum()
            u_p = 100 * u / n_und if n_und > 0 else 0
            d_p = 100 * d / n_det if n_det > 0 else 0
            name = status_map.get(s, s)
            print(f"  {name:<18} {u:>9,}  {u_p:>7.1f}%  {d:>9,}  {d_p:>7.1f}%")

        out_und = 100 * (und[status_col] == "OUT").sum() / n_und if n_und > 0 else 0
        out_det = 100 * (det[status_col] == "OUT").sum() / n_det if n_det > 0 else 0
        print(f"\n  → 未探测组 OUT 占比 {out_und:.1f}% vs 已探测组 {out_det:.1f}%")
        print(f"     OUT = 火已扑灭 → 无热量 → 卫星看不到，符合预期")

    # ── 4c. 火因 ─────────────────────────────────────────────────────
    _subsection("c. 火灾原因")

    cause_col = "field_system_fire_cause"
    if cause_col in df.columns:
        cause_map = {"N": "N（自然/雷电）", "H": "H（人为）", "U": "U（未知）"}
        print(f"\n  {'火因':<20} {'未探测%':>8}  {'已探测%':>8}")
        print(f"  {'─'*40}")
        for c in sorted(df[cause_col].dropna().unique()):
            u_p = 100 * (und[cause_col] == c).sum() / n_und if n_und > 0 else 0
            d_p = 100 * (det[cause_col] == c).sum() / n_det if n_det > 0 else 0
            name = cause_map.get(c, c)
            print(f"  {name:<20} {u_p:>7.1f}%  {d_p:>7.1f}%")

    # ── 4d. 省份/机构 ─────────────────────────────────────────────────
    _subsection("d. 省份/机构（未探测率最高的前10个）")

    agency_col = "field_agency_code"
    if agency_col in df.columns:
        grp = df.groupby(agency_col).agg(
            total    = (agency_col, "count"),
            undet    = ("no_hotspots_in_window", "sum"),
        )
        grp["undet_pct"] = 100 * grp["undet"] / grp["total"]
        grp = grp.sort_values("undet_pct", ascending=False)

        print(f"\n  {'省份':>6}  {'总记录':>8}  {'未探测':>8}  {'未探测率':>8}")
        print(f"  {'─'*38}")
        for agency, row in grp.head(10).iterrows():
            print(f"  {str(agency):>6}  {int(row['total']):>8,}  "
                  f"{int(row['undet']):>8,}  {row['undet_pct']:>7.1f}%")

    # ── 4e. 时间分布 ──────────────────────────────────────────────────
    _subsection("e. 时间分布（未探测集中在哪几个月？）")

    if "ciffc_date" in und.columns:
        und_copy = und.copy()
        und_copy["_month"] = pd.to_datetime(und_copy["ciffc_date"]).dt.month
        month_counts = und_copy.groupby("_month").size()
        month_names  = {5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
        total_und    = month_counts.sum()

        print(f"\n  {'月份':<6} {'记录数':>7}  {'占比':>6}  分布")
        print(f"  {'─'*45}")
        for m in range(5, 11):
            cnt = int(month_counts.get(m, 0))
            pct = 100 * cnt / total_und if total_und > 0 else 0
            bar = _bar(pct, 30.0, 20)
            print(f"  {month_names.get(m, str(m)):<6} {cnt:>7,}  {pct:>5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="分析 CIFFC vs hotspot 匹配结果（compare_ciffc_hotspot 的输出）"
    )
    ap.add_argument("--match_csv", type=str, default="ciffc_hotspot_match.csv",
                    help="compare_ciffc_hotspot.py 输出的 CSV（默认 ciffc_hotspot_match.csv）")
    ap.add_argument("--window_days", type=int, default=7,
                    help="当时使用的时间窗口天数（仅用于展示，默认 7）")
    args = ap.parse_args(argv)

    csv_path = args.match_csv
    if not Path(csv_path).exists():
        sys.exit(f"[ERROR] 找不到文件：{csv_path}\n"
                 f"请先运行 compare_ciffc_hotspot.py 生成匹配文件。")

    print(f"\n{'='*65}")
    print(f"  CIFFC vs Hotspot 深度分析报告")
    print(f"  输入文件：{csv_path}")
    print(f"  时间窗口：±{args.window_days} 天")
    print(f"{'='*65}")

    df = pd.read_csv(csv_path)
    print(f"\n  已加载 {len(df):,} 条记录")

    section1_detection_rate(df)
    section2_time_error(df)
    section3_distance_error(df)
    section4_undetected_profile(df)

    print(f"\n{'='*65}")
    print(f"  分析完成")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
