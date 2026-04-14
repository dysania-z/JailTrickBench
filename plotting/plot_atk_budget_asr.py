#!/usr/bin/env python3
"""
读取 test_atk_budget 实验结果，绘制双子图：
  (a) PAIR：n_iterations ∈ {5,10,20,40}
  (b) AutoDAN：gcg_attack_budget ∈ {50,100,200,400}
攻击者与目标均为 Vicuna-13B。优先 is_JB_Agent。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

DEFENSE_DIR = "1None_defense"
ATTACK_AUTODAN = "1AutoDAN"
ATTACK_PAIR = "2PAIR"


def _jb_success(item: Dict[str, Any]) -> bool:
    ja = item.get("is_JB_Agent")
    if ja not in (None, "None"):
        return bool(ja)
    jj = item.get("is_JB_Judge")
    if jj not in (None, "None"):
        return bool(jj)
    return bool(item.get("is_JB", False))


def _load_latest_json(
    result_dir: str, attack_subdir: str, exp_name_prefix: str
) -> Optional[List[Dict]]:
    d = os.path.join(result_dir, DEFENSE_DIR, attack_subdir)
    if not os.path.isdir(d):
        return None
    candidates = [
        f
        for f in glob.glob(os.path.join(d, exp_name_prefix + "_*.json"))
        if os.path.isfile(f)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    return data


def _asr(data: Optional[List[Dict]]) -> float:
    if not data:
        return float("nan")
    ok = sum(1 for x in data if _jb_success(x))
    return 100.0 * ok / len(data)


def _scan_budget_order(
    result_dir: str, attack_subdir: str, prefix: str, value_key: str, palette: List[str]
) -> List[Tuple[str, str, str]]:
    d = os.path.join(result_dir, DEFENSE_DIR, attack_subdir)
    if not os.path.isdir(d):
        return []
    pat = re.compile(rf"^({re.escape(prefix)}{re.escape(value_key)}(\d+))_")
    values = set()
    for path in glob.glob(os.path.join(d, "*.json")):
        fname = os.path.basename(path)
        m = pat.match(fname)
        if m:
            values.add(int(m.group(2)))
    ordered_values = sorted(values)
    order = []
    for idx, val in enumerate(ordered_values):
        exp_prefix = f"{prefix}{value_key}{val}"
        order.append((str(val), exp_prefix, palette[idx % len(palette)]))
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_dir",
        default=os.path.join(_REPO_ROOT, "exp_results", "test_atk_budget"),
        help="与 --save_result_path 一致",
    )
    ap.add_argument("--out", default=None, help="输出 png 路径")
    args = ap.parse_args()
    result_dir = os.path.abspath(args.result_dir)
    out_path = args.out or os.path.join(result_dir, "atk_budget_asr.png")

    pair_palette = ["#6b9bd1", "#5a8bc4", "#4a7db8", "#3a6fac", "#315f96", "#284f80"]
    ad_palette = ["#c17a5c", "#b06a4c", "#9f5a3c", "#8e4a2c", "#7d3b1d", "#6c2d10"]
    pair_order = _scan_budget_order(
        result_dir, ATTACK_PAIR, "budget_pair_", "it", pair_palette
    )
    ad_order = _scan_budget_order(
        result_dir, ATTACK_AUTODAN, "budget_autodan_", "b", ad_palette
    )
    if not pair_order:
        pair_order = [
            ("5", "budget_pair_it5", pair_palette[0]),
            ("10", "budget_pair_it10", pair_palette[1]),
            ("20", "budget_pair_it20", pair_palette[2]),
            ("40", "budget_pair_it40", pair_palette[3]),
        ]
    if not ad_order:
        ad_order = [
            ("1", "budget_autodan_b1", ad_palette[0]),
            ("5", "budget_autodan_b5", ad_palette[1]),
            ("10", "budget_autodan_b10", ad_palette[2]),
            ("50", "budget_autodan_b50", ad_palette[3]),
        ]

    pair_vals = [_asr(_load_latest_json(result_dir, ATTACK_PAIR, p)) for _, p, _ in pair_order]
    ad_vals = [_asr(_load_latest_json(result_dir, ATTACK_AUTODAN, p)) for _, p, _ in ad_order]

    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as e:
        raise SystemExit("需要 matplotlib：pip install matplotlib\n" + str(e)) from e

    mpl.rcParams.update(
        {
            "figure.facecolor": "#1a1a1e",
            "axes.facecolor": "#1a1a1e",
            "axes.edgecolor": "#444",
            "axes.labelcolor": "#ddd",
            "text.color": "#eee",
            "xtick.color": "#ccc",
            "ytick.color": "#ccc",
            "grid.color": "#444",
            "grid.linestyle": "--",
            "font.size": 10,
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)

    def draw_ax(
        ax,
        order: List[Tuple[str, str, str]],
        vals: List[float],
        title: str,
        xlabel: str,
    ) -> None:
        labels = [x[0] for x in order]
        colors = [x[2] for x in order]
        x = range(len(labels))
        ymax = min(
            100.0,
            max(50.0, max((v for v in vals if v == v), default=50.0) * 1.15),
        )
        bars = ax.bar(x, vals, color=colors, width=0.65, edgecolor="#222")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel, color="#bbb")
        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, ymax)
        ax.set_title(title, color="#eee", fontsize=12)
        ax.grid(axis="y", alpha=0.5)
        for b, v in zip(bars, vals):
            if v != v:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    ymax * 0.02,
                    "N/A",
                    ha="center",
                    va="bottom",
                    color="#888",
                    fontsize=8,
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    min(v + ymax * 0.02, ymax - 1),
                    f"{v:.0f}%",
                    ha="center",
                    va="bottom",
                    color="#ddd",
                    fontsize=9,
                )

    draw_ax(
        ax0,
        pair_order,
        pair_vals,
        "(a) PAIR (Vicuna-13B attacker & target)",
        "n_iterations (attack budget)",
    )
    draw_ax(
        ax1,
        ad_order,
        ad_vals,
        "(b) AutoDAN (Vicuna-13B attacker & target)",
        "gcg_attack_budget",
    )

    plt.suptitle(
        "Attack budget vs ASR — Vicuna-13B / None_defense / harmful_bench_15",
        color="#ccc",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
