"""Tạo lại hai hình thực nghiệm từ các kết quả đã xác minh trong kho mã."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parent
LOGS = FIGURES.parents[1] / "handoff" / "logs"

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})


def load_log(relative_path):
    return json.loads((LOGS / relative_path).read_text(encoding="utf-8"))


def plot_fps():
    data = load_log("review109_nano80/results/score_verified.json")["throughput"]
    groups = ["bdd", "ua", "all"]
    labels = ["BDD100K\n40 đoạn", "UA-DETRAC\n40 đoạn", "Gộp\n80 đoạn"]
    fig, ax = plt.subplots(figsize=(6.25, 3.0), layout="constrained")
    configurations = [
        ("v21combo", "v21combo", -0.19, "#6c7884"),
        ("overlap", "Cấu hình tối ưu R109", 0.19, "#176480"),
    ]
    for key, label, offset, color in configurations:
        values = [data[group][key]["mean_fps"] for group in groups]
        bars = ax.bar([i + offset for i in range(3)], values, 0.32, label=label, color=color)
        ax.bar_label(bars, [f"{value:.2f}".replace(".", ",") for value in values], padding=3)
    ax.axhline(15, linewidth=0.9, color="#30363a", linestyle="--", label="Mốc 15 FPS")
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 23)
    ax.set_ylabel("FPS trung bình")
    ax.grid(axis="y", color="#dce1e5", linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncols=3, frameon=False)
    fig.savefig(FIGURES / "fps-jetson-nano.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_trajectories():
    data = load_log("review111_roundabout/turn_episodes.json")
    examples = [("imagesc001", 371, 6298), ("imagesc002", 226, 4665)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.95), layout="constrained")
    for ax, (camera, gt_id, first) in zip(axes, examples):
        segment = next(
            episode for episode in data["episodes"]
            if (episode["camera"], episode["gt_id"], episode["first"]) == (camera, gt_id, first)
        )
        _, x, y = zip(*segment["path"])
        ax.plot(x, y, color="#176480", marker=".", linewidth=1.7, markersize=4)
        ax.scatter(x[0], y[0], marker="o", color="#176480", zorder=4, s=39)
        ax.scatter(x[-1], y[-1], marker="s", color="#954520", zorder=4, s=32)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.grid(color="#dce1e5", linewidth=0.6)
        idsw = ", ".join(
            f"{name.upper()} {segment['metrics'][name]['idsw']}"
            for name in ("kf", "ekf", "ukf")
        )
        camera_number = int(camera[-3:])
        ax.set_title(
            f"Camera {camera_number}, GT {gt_id}\n"
            f"Khung {first}–{segment['last']}\n"
            f"IDSW {idsw}",
            fontsize=10,
        )
    fig.savefig(FIGURES / "quy-dao-roundabout.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_fps()
    plot_trajectories()
