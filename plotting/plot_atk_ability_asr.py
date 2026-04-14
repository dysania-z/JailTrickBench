#!/usr/bin/env python3
"""
从 test_atk_ability_pair 实验目录读取 JSON 结果，绘制 PAIR 攻击者能力 ASR 柱状图。
三种攻击者（Vicuna-7B / Vicuna-13B / DeepSeek-Chat）固定攻击 Vicuna-13B。
优先使用 is_JB_Agent；否则 is_JB_Judge，再否则 is_JB。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

DEFENSE_DIR = "1None_defense"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_dir",
        default=os.path.join(_REPO_ROOT, "exp_results", "test_atk_ability_pair"),
        help="与 --save_result_path 一致",
    )
    ap.add_argument("--out", default=None, help="输出 png 路径")
    args = ap.parse_args()
    result_dir = os.path.abspath(args.result_dir)
    out_path = args.out or os.path.join(result_dir, "atk_ability_asr.png")

    order: List[Tuple[str, str, str]] = [
        ("Vicuna-7B", "atk_vicuna7b_pair", "#5b8c85"),
        ("Vicuna-13B", "atk_vicuna13b_pair", "#3a6b5e"),
        ("DeepSeek-Chat", "atk_deepseek_pair", "#c06040"),
    ]

    pair_vals: List[float] = []
    for label, prefix, _c in order:
        dp = _load_latest_json(result_dir, ATTACK_PAIR, prefix)
        pair_vals.append(_asr(dp))

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
            "font.size": 11,
        }
    )

    labels = [x[0] for x in order]
    colors = [x[2] for x in order]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=120)
    bars = ax.bar(x, pair_vals, color=colors, width=0.55, edgecolor="#222")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("ASR (%)")
    pr_max = max((v for v in pair_vals if v == v), default=50.0)
    ymax = min(100.0, max(50.0, pr_max * 1.15))
    ax.set_ylim(0, ymax)
    ax.set_title(
        "PAIR: Attacker Ability vs ASR  (Target = Vicuna-13B)",
        color="#eee",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.5)

    for b, v in zip(bars, pair_vals):
        if v != v:  # nan
            ax.text(
                b.get_x() + b.get_width() / 2,
                ymax * 0.02,
                "N/A",
                ha="center",
                va="bottom",
                color="#888",
                fontsize=9,
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(v + ymax * 0.02, ymax - 1),
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                color="#ddd",
                fontsize=10,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
