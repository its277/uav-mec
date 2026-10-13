#!/usr/bin/env python3
"""Summarize DDPG grid-search results and generate plots/report.

Run this script from the directory containing ddpg_grid_results.csv, or place
that CSV beside this script. Works with older Matplotlib versions as well.
"""
import csv
import itertools
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "ddpg_grid_results.csv")

REQUIRED_COLUMNS = (
    "trial", "episodes", "actor_lr", "critic_lr", "batch_size",
    "eval_mean_return", "eval_std_return", "last_100_train_return",
)
INTEGER_COLUMNS = ("trial", "episodes", "batch_size")
FLOAT_COLUMNS = (
    "actor_lr", "critic_lr", "eval_mean_return",
    "eval_std_return", "last_100_train_return",
)


def load_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Results CSV not found: {}\nRun ddpg_analysis.py first, or place "
            "ddpg_grid_results.csv beside summarize_ddpg.py.".format(path)
        )

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The results CSV is empty or has no header.")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                "CSV is missing required column(s): {}. Found: {}".format(
                    ", ".join(missing), ", ".join(reader.fieldnames)
                )
            )
        rows = []
        for line_number, source_row in enumerate(reader, start=2):
            # Ignore entirely blank lines, but report malformed result rows.
            if not any((value or "").strip() for value in source_row.values()):
                continue
            row = {}
            try:
                for key in INTEGER_COLUMNS:
                    row[key] = int(source_row[key])
                for key in FLOAT_COLUMNS:
                    row[key] = float(source_row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid value in CSV line {}: {}".format(line_number, exc)
                ) from exc
            if not all(math.isfinite(row[key]) for key in FLOAT_COLUMNS):
                raise ValueError("Non-finite numeric value in CSV line {}.".format(line_number))
            rows.append(row)

    if not rows:
        raise ValueError("The results CSV contains no trial results.")
    return rows


def fmt_lr(value):
    return "{:.4g}".format(value)


def factor_levels(rows):
    return {
        "episodes": sorted({r["episodes"] for r in rows}),
        "actor_lr": sorted({r["actor_lr"] for r in rows}),
        "critic_lr": sorted({r["critic_lr"] for r in rows}),
        "batch_size": sorted({r["batch_size"] for r in rows}),
    }


def summarize(rows, field, levels):
    result = []
    for value in levels[field]:
        values = np.asarray(
            [r["eval_mean_return"] for r in rows if r[field] == value],
            dtype=float,
        )
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append((value, float(values.mean()), sd, len(values)))
    return result


def save_main_effects(rows, levels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_specs = [
        ("episodes", "Training episodes", "Episodes"),
        ("actor_lr", "Actor learning rate", "Actor LR"),
        ("critic_lr", "Critic learning rate", "Critic LR"),
        ("batch_size", "Batch size", "Batch size"),
    ]
    for ax, (field, title, xlabel) in zip(axes.flat, plot_specs):
        data = summarize(rows, field, levels)
        x = np.arange(len(data))
        means = [item[1] for item in data]
        spread = [item[2] for item in data]
        labels = [
            fmt_lr(item[0]) if field in ("actor_lr", "critic_lr") else str(item[0])
            for item in data
        ]
        ax.errorbar(x, means, yerr=spread, marker="o", capsize=4, linewidth=2)
        # Separate calls keep this compatible with older Matplotlib releases.
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean evaluation return (higher is better)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "DDPG UAV grid search: marginal effects "
        "(error bars = SD across configurations)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(ROOT, "ddpg_grid_main_effects.png"), dpi=180)
    plt.close(fig)


def save_heatmaps(rows, levels):
    episodes = levels["episodes"]
    batches = levels["batch_size"]
    actors = levels["actor_lr"]
    critics = levels["critic_lr"]
    expected = len(episodes) * len(batches) * len(actors) * len(critics)

    lookup = {
        (r["episodes"], r["batch_size"], r["actor_lr"], r["critic_lr"]):
        r["eval_mean_return"] for r in rows
    }
    if len(lookup) < expected:
        print(
            "Warning: incomplete grid ({} unique configurations; {} possible). "
            "Missing heatmap cells will be shown as blank.".format(len(lookup), expected),
            file=sys.stderr,
        )

    # A 2x6 grid made each panel too narrow for its title, tick labels, and
    # cell annotations. Use four columns with room for each 3x3 matrix.
    panel_count = len(episodes) * len(batches)
    ncols = min(4, panel_count)
    nrows = int(math.ceil(panel_count / float(ncols)))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols + 1.2, 3.7 * nrows + 1.0),
        squeeze=False,
        constrained_layout=True,
    )
    values = [r["eval_mean_return"] for r in rows]
    vmin, vmax = min(values), max(values)
    images = []
    panel = 0
    for episode_count in episodes:
        for batch_size in batches:
            ax = axes.flat[panel]
            panel += 1
            matrix = np.full((len(actors), len(critics)), np.nan)
            for ai, actor_lr in enumerate(actors):
                for ci, critic_lr in enumerate(critics):
                    key = (episode_count, batch_size, actor_lr, critic_lr)
                    if key in lookup:
                        matrix[ai, ci] = lookup[key]
            image = ax.imshow(
                np.ma.masked_invalid(matrix), cmap="viridis",
                vmin=vmin, vmax=vmax, aspect="auto",
            )
            images.append(image)
            ax.set_xticks(range(len(critics)))
            ax.set_xticklabels([fmt_lr(v) for v in critics])
            ax.set_yticks(range(len(actors)))
            ax.set_yticklabels([fmt_lr(v) for v in actors])
            ax.set_xlabel("Critic LR")
            ax.set_ylabel("Actor LR")
            ax.set_title("{} episodes | batch {}".format(episode_count, batch_size),
                         fontsize=11, pad=8)
            midpoint = (vmin + vmax) / 2.0
            for ai in range(matrix.shape[0]):
                for ci in range(matrix.shape[1]):
                    if np.isfinite(matrix[ai, ci]):
                        color = "white" if matrix[ai, ci] < midpoint else "black"
                        ax.text(
                            ci, ai, "{:.0f}".format(matrix[ai, ci]),
                            ha="center", va="center", color=color, fontsize=9,
                        )
    for ax in axes.flat[panel_count:]:
        ax.set_visible(False)
    fig.colorbar(
        images[0], ax=axes.ravel().tolist(), shrink=0.82,
        label="Mean evaluation return (higher is better)",
    )
    fig.suptitle("Evaluation return by actor and critic learning rate", fontsize=16)
    fig.savefig(
        os.path.join(ROOT, "ddpg_grid_heatmaps.png"),
        dpi=180, bbox_inches="tight",
    )
    plt.close(fig)


def save_top10(rows):
    top = rows[:10]
    ordered = list(reversed(top))
    labels = [
        "{} ep | A={} | C={} | B={}".format(
            r["episodes"], fmt_lr(r["actor_lr"]),
            fmt_lr(r["critic_lr"]), r["batch_size"],
        )
        for r in ordered
    ]
    scores = [r["eval_mean_return"] for r in ordered]
    errors = [r["eval_std_return"] for r in ordered]
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(ordered))))
    ax.errorbar(scores, y, xerr=errors, fmt="none", ecolor="#222222",
                capsize=3, linewidth=1.2, zorder=2)
    ax.scatter(scores, y, s=55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean total reward across evaluation episodes (higher is better)")
    ax.set_title("Top {} DDPG hyperparameter configurations".format(len(top)))
    spread = max(errors) if errors else 0.0
    low, high = min(scores), max(scores)
    padding = max(spread, (high - low) * 0.05, 1.0)
    ax.set_xlim(low - padding - 1, high + padding + 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "ddpg_grid_top10.png"), dpi=180)
    plt.close(fig)


def write_report(rows, levels):
    ranked = sorted(rows, key=lambda r: r["eval_mean_return"], reverse=True)
    unique_count = len({
        (r["episodes"], r["actor_lr"], r["critic_lr"], r["batch_size"])
        for r in rows
    })
    expected = (
        len(levels["episodes"]) * len(levels["actor_lr"])
        * len(levels["critic_lr"]) * len(levels["batch_size"])
    )
    report = [
        "# UAV-DDPG hyperparameter grid search",
        "",
        "Loaded {} result rows representing {} unique configurations. "
        "The observed grid contains {} possible combinations of the factor levels "
        "present in this CSV.".format(len(rows), unique_count, expected),
        "Evaluation score is cumulative reward per episode; higher (less negative) "
        "returns are better when reward represents negative delay.",
        "The evaluation SD is copied from the CSV and describes evaluation-episode "
        "variation as computed by the training script; it is not variation across "
        "independent training seeds.",
        "",
        "## Search levels",
        "",
        "- Training episodes: {}".format(", ".join(map(str, levels["episodes"]))),
        "- Actor learning rate: {}".format(", ".join(fmt_lr(v) for v in levels["actor_lr"])),
        "- Critic learning rate: {}".format(", ".join(fmt_lr(v) for v in levels["critic_lr"])),
        "- Batch size: {}".format(", ".join(map(str, levels["batch_size"]))),
        "",
        "## Best observed configuration",
        "",
        "| Episodes | Actor LR | Critic LR | Batch | Eval mean | Eval SD | Last 100 training return |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked[:5]:
        report.append(
            "| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                r["episodes"], fmt_lr(r["actor_lr"]), fmt_lr(r["critic_lr"]),
                r["batch_size"], r["eval_mean_return"], r["eval_std_return"],
                r["last_100_train_return"],
            )
        )

    best = ranked[0]
    report += [
        "",
        "**Best row in the CSV:** {} episodes, actor LR {}, critic LR {}, "
        "batch size {}; evaluation mean {:.3f} (SD {:.3f}).".format(
            best["episodes"], fmt_lr(best["actor_lr"]), fmt_lr(best["critic_lr"]),
            best["batch_size"], best["eval_mean_return"], best["eval_std_return"],
        ),
        "",
        "## Average evaluation return by factor level",
        "",
        "These marginal means average over the other factor settings represented in "
        "the CSV. The SD is across result rows at that factor level.",
        "",
    ]
    for field, name in [
        ("episodes", "Training episodes"),
        ("actor_lr", "Actor learning rate"),
        ("critic_lr", "Critic learning rate"),
        ("batch_size", "Batch size"),
    ]:
        report += [
            "### {}".format(name),
            "",
            "| Level | Mean return | SD across configurations | Rows |",
            "|---:|---:|---:|---:|",
        ]
        for value, mean, sd, count in summarize(rows, field, levels):
            label = fmt_lr(value) if field in ("actor_lr", "critic_lr") else str(value)
            report.append("| {} | {:.3f} | {:.3f} | {} |".format(label, mean, sd, count))
        report.append("")

    report += [
        "## Complete ranked results",
        "",
        "| Rank | Episodes | Actor LR | Critic LR | Batch | Eval mean | Eval SD | Last 100 train return |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(ranked, 1):
        report.append(
            "| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                rank, r["episodes"], fmt_lr(r["actor_lr"]), fmt_lr(r["critic_lr"]),
                r["batch_size"], r["eval_mean_return"], r["eval_std_return"],
                r["last_100_train_return"],
            )
        )
    report += [
        "",
        "## Generated files",
        "",
        "- `ddpg_grid_results.csv`: source results file.",
        "- `ddpg_grid_main_effects.png`: marginal effects of each factor.",
        "- `ddpg_grid_heatmaps.png`: actor/critic learning-rate heatmaps by episode and batch settings.",
        "- `ddpg_grid_top10.png`: top configurations with evaluation SD error bars.",
        "",
        "Interpret marginal summaries cautiously: parameter interactions can make the "
        "best individual configuration differ from the best average factor level. "
        "If the CSV contains only one training seed per configuration, these results "
        "do not quantify run-to-run training variability.",
    ]
    report_path = os.path.join(ROOT, "ddpg_grid_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")


def main():
    rows = load_results(CSV_PATH)
    rows.sort(key=lambda row: row["eval_mean_return"], reverse=True)
    levels = factor_levels(rows)
    unique_count = len({
        (r["episodes"], r["actor_lr"], r["critic_lr"], r["batch_size"])
        for r in rows
    })
    expected = (
        len(levels["episodes"]) * len(levels["actor_lr"])
        * len(levels["critic_lr"]) * len(levels["batch_size"])
    )
    print("Loaded {} rows ({} unique configurations; {} combinations possible).".format(
        len(rows), unique_count, expected
    ))
    if unique_count != expected:
        print("Warning: results may be incomplete or contain duplicate configurations.",
              file=sys.stderr)

    save_main_effects(rows, levels)
    save_heatmaps(rows, levels)
    save_top10(rows)
    write_report(rows, levels)

    print("Top 10 configurations:")
    for rank, row in enumerate(rows[:10], 1):
        print(
            "{}. episodes={} actor_lr={} critic_lr={} batch={} eval={:.3f} +/- {:.3f}".format(
                rank, row["episodes"], row["actor_lr"], row["critic_lr"],
                row["batch_size"], row["eval_mean_return"], row["eval_std_return"],
            )
        )
    print("Wrote report and three charts to {}".format(ROOT))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        sys.exit(1)
