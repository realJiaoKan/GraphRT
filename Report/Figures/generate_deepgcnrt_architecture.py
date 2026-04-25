from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "deepgcnrt_architecture.png"
PDF_PATH = OUT_DIR / "deepgcnrt_architecture.pdf"

COLORS = {
    "input": "#DBEAFE",
    "encoder": "#E0F2FE",
    "stack": "#FDE68A",
    "stack_shadow": "#FCD34D",
    "readout": "#DCFCE7",
    "head": "#EDE9FE",
    "output": "#FCE7F3",
    "badge": "#F8FAFC",
    "edge": "#334155",
    "text": "#0F172A",
    "muted": "#475569",
}


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines,
    facecolor,
    edgecolor=COLORS["edge"],
    title_size=14,
    text_size=10.5,
    rounding=14,
    lw=1.8,
    zorder=2,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h - 5.5,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        zorder=zorder + 1,
    )

    if lines:
        start_y = y + h - 13
        line_gap = max(4.7, min(6.2, (h - 18) / max(len(lines), 1)))
        for idx, line in enumerate(lines):
            ax.text(
                x + 2.5,
                start_y - idx * line_gap,
                line,
                ha="left",
                va="top",
                fontsize=text_size,
                color=COLORS["text"],
                zorder=zorder + 1,
            )
    return box


def add_arrow(ax, start, end, color=COLORS["edge"], lw=2.0, style="-|>", rad=0.0, zorder=3):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_badge(ax, x, y, w, text):
    badge = FancyBboxPatch(
        (x, y),
        w,
        7.5,
        boxstyle="round,pad=0.02,rounding_size=7",
        linewidth=1.0,
        edgecolor="#CBD5E1",
        facecolor=COLORS["badge"],
        zorder=1,
    )
    ax.add_patch(badge)
    ax.text(
        x + w / 2,
        y + 3.75,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
        zorder=2,
    )


def draw_input_graph(ax, x, y, w, h):
    node_points = [
        (x + 4.4, y + 10.0),
        (x + 7.8, y + 15.0),
        (x + 12.8, y + 14.2),
        (x + 11.0, y + 7.6),
        (x + 6.4, y + 5.8),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    for src, dst in edges:
        x0, y0 = node_points[src]
        x1, y1 = node_points[dst]
        ax.plot([x0, x1], [y0, y1], color="#64748B", lw=1.6, zorder=4)
    for px, py in node_points:
        ax.add_patch(
            Circle(
                (px, py),
                radius=0.9,
                facecolor="#2563EB",
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
            )
        )

    ax.text(
        x + w / 2,
        y + 3.2,
        "graph from SMILES",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
        zorder=6,
    )


def draw_stack(ax, x, y, w, h):
    offsets = [(0.0, 0.0), (-1.7, 1.6), (-3.4, 3.2)]
    for idx, (dx, dy) in enumerate(reversed(offsets)):
        face = COLORS["stack_shadow"] if idx < 2 else COLORS["stack"]
        add_box(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            "",
            [],
            facecolor=face,
            edgecolor=COLORS["edge"],
            rounding=16,
            lw=1.5,
            zorder=1 + idx,
        )

    ax.text(
        x + w / 2,
        y + h - 5.3,
        "16 x Residual Edge-Aware GCN Layer",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h - 11.8,
        r"$\mathbf{h}^{(l)}_u, \mathbf{h}^{(l)}_{e_{uv}} \in \mathbb{R}^{200}$",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
        zorder=5,
    )

    add_box(
        ax,
        x + 3.0,
        y + 8.5,
        w - 6.0,
        h - 25.0,
        "One layer",
        [
            r"$m_{uv} = h_u + e_{uv}$",
            r"$\alpha_{uv} = \mathrm{softmax}(m_{uv})$ over incoming edges",
            r"$m_v = \sum_u \alpha_{uv} \odot m_{uv}$",
            "Linear(200 -> 200) + ReLU + Dropout(0.1)",
            r"Residual add: $h_v^{(l+1)} \leftarrow h_v^{(l+1)} + h_v^{(l)}$",
            "Output norm = Identity",
        ],
        facecolor="#FEF3C7",
        title_size=12.5,
        text_size=10.2,
        rounding=12,
        lw=1.4,
        zorder=4,
    )

    ax.text(
        x + w - 5.5,
        y + 4.8,
        "repeat",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
        zorder=5,
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        60,
        95,
        "DeepGCN-RT Architecture",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        60,
        90.5,
        "PyTorch Geometric reimplementation used in this report",
        ha="center",
        va="center",
        fontsize=12.5,
        color=COLORS["muted"],
    )

    input_box = add_box(
        ax,
        4,
        34,
        16,
        32,
        "Molecular Graph",
        [
            r"atom features $x \in \mathbb{R}^{164}$",
            r"bond features $e \in \mathbb{R}^{11}$",
        ],
        facecolor=COLORS["input"],
    )
    draw_input_graph(ax, 4, 34, 16, 32)

    node_box = add_box(
        ax,
        24,
        56,
        18,
        17,
        "Node Encoder",
        [r"Linear$(164 \rightarrow 200)$"],
        facecolor=COLORS["encoder"],
        text_size=11,
    )
    edge_box = add_box(
        ax,
        24,
        27,
        18,
        17,
        "Edge Encoder",
        [r"Linear$(11 \rightarrow 200)$"],
        facecolor=COLORS["encoder"],
        text_size=11,
    )

    draw_stack(ax, 49, 21, 27, 58)

    readout_box = add_box(
        ax,
        82,
        31,
        19,
        37,
        "Attentive Readout",
        [
            r"$g^{(0)} = \sum_v h_v^{(16)}$",
            "2 refinement steps",
            "GATConv(nodes -> graph)",
            "ELU + Dropout(0.1)",
            "GRUCell update",
            r"graph embedding $g \in \mathbb{R}^{200}$",
        ],
        facecolor=COLORS["readout"],
        title_size=13,
        text_size=10.1,
    )

    head_box = add_box(
        ax,
        105,
        36,
        11,
        27,
        "MLP Head",
        [
            "Linear",
            "200 -> 1024",
            "ReLU",
            "1024 -> 1",
        ],
        facecolor=COLORS["head"],
        title_size=12.5,
        text_size=10.2,
    )

    output_box = add_box(
        ax,
        104,
        10,
        12,
        14,
        "Output",
        [
            "predicted RT",
            "in seconds",
        ],
        facecolor=COLORS["output"],
        title_size=12.5,
        text_size=10.4,
    )

    add_arrow(ax, (19.5, 53.5), (24, 64.0))
    add_arrow(ax, (19.5, 45.5), (24, 35.5))
    add_arrow(ax, (42, 64.0), (49, 64.0))
    add_arrow(ax, (42, 35.5), (49, 35.5))
    add_arrow(ax, (76, 50), (82, 50))
    add_arrow(ax, (101, 50), (105, 50))
    add_arrow(ax, (110.5, 36), (110.5, 24), style="-|>", rad=0.0)

    ax.text(
        79.0,
        54.6,
        r"node states $H^{(16)}$",
        ha="right",
        va="center",
        fontsize=10.3,
        color=COLORS["muted"],
    )

    add_badge(ax, 7, 6, 16, "hidden dim = 200")
    add_badge(ax, 26, 6, 17, "GCN layers = 16")
    add_badge(ax, 46, 6, 18, "readout steps = 2")
    add_badge(ax, 67, 6, 17, "dropout = 0.1")
    add_badge(ax, 87, 6, 23, "update_func = no_relu")

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
