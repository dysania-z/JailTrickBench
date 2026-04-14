#!/usr/bin/env python3
"""
从 target_size_pair 实验目录读取 JSON 结果，绘制 (a) AutoDAN (b) PAIR 双子图柱状图（ASR %）。
优先使用 is_JB_Agent（GPT）；否则 PAIR 用 is_JB_Judge，再否则 is_JB。

在仓库根目录执行：python plotting/plot_target_size_asr.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# 与 utils.test_utils.attack_rename 一致
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


def _load_latest_json(result_dir: str, attack_subdir: str, exp_name_prefix: str) -> Optional[List[Dict]]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_dir",
        default=os.path.join(_REPO_ROOT, "exp_results", "target_size_pair"),
        help="与 --save_result_path 一致（默认：仓库下 exp_results/target_size_pair）",
    )
    ap.add_argument("--out", default=None, help="输出 png 路径")
    args = ap.parse_args()
    result_dir = os.path.abspath(args.result_dir)
    out_path = args.out or os.path.join(result_dir, "target_size_asr.png")

    # 与 target_size_pair.sh 中 exp_name 一致
    order: List[Tuple[str, str, str]] = [
        ("Llama2-7B", "size_llama2-7b", "#a85c50"),
        ("Llama2-13B", "size_llama2-13b", "#a85c50"),
        ("Vicuna-7B", "size_vicuna-7b", "#2d6a4f"),
        ("Vicuna-13B", "size_vicuna-13b", "#2d6a4f"),
    ]

    autodan_vals: List[float] = []
    pair_vals: List[float] = []
    for label, prefix, _c in order:
        da = _load_latest_json(result_dir, ATTACK_AUTODAN, f"{prefix}_autodan")
        dp = _load_latest_json(result_dir, ATTACK_PAIR, f"{prefix}_pair")
        autodan_vals.append(_asr(da))
        pair_vals.append(_asr(dp))

    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as e:
        raise SystemExit(
            "需要 matplotlib：pip install matplotlib\n" + str(e)
        ) from e

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

    labels = [x[0] for x in order]
    colors = [x[2] for x in order]
    x = range(len(labels))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)

    def draw_ax(ax, vals: List[float], title: str, ymax: float) -> None:
        bars = ax.bar(x, vals, color=colors, width=0.65, edgecolor="#222")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, ymax)
        ax.set_title(title, color="#eee", fontsize=12)
        ax.grid(axis="y", alpha=0.5)
        for b, v in zip(bars, vals):
            if v != v:  # nan
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

    ad_max = max((v for v in autodan_vals if v == v), default=50.0)
    pr_max = max((v for v in pair_vals if v == v), default=50.0)
    draw_ax(ax0, autodan_vals, "(a) AutoDAN", min(100.0, max(50.0, ad_max * 1.15)))
    draw_ax(ax1, pair_vals, "(b) PAIR", min(100.0, max(50.0, pr_max * 1.15)))

    from matplotlib.patches import Patch

    legend_el = [
        Patch(facecolor="#a85c50", label="Llama 2"),
        Patch(facecolor="#2d6a4f", label="Vicuna"),
    ]
    fig.legend(handles=legend_el, loc="upper center", ncol=2, frameon=False, labelcolor="#ccc")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
