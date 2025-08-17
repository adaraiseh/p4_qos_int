#!/usr/bin/env python3
# pretty_topology.py
import json
import argparse
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt


# ---- Colors & Shapes ---------------------------------------------------------
PALETTE = {
    "host":  "#86efac",   # light green
    "tor":   "#93c5fd",   # light blue
    "agg":   "#fbbf24",   # amber
    "core":  "#f87171",   # light red
    "other": "#9ca3af",   # gray
}
SHAPES = {
    "host": "o",
    "tor":  "s",
    "agg":  "^",
    "core": "D",
    "other": "o",
}


# ---- Helpers -----------------------------------------------------------------
def role_of(node_id: str) -> str:
    if node_id.startswith('h'): return "host"
    if node_id.startswith('t'): return "tor"
    if node_id.startswith('a'): return "agg"
    if node_id.startswith('c'): return "core"
    return "other"


def load_topology(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def build_graph(data):
    G = nx.Graph()
    roles = defaultdict(list)
    for node in data['nodes']:
        node_id = node['id']
        G.add_node(node_id)
        roles[role_of(node_id)].append(node_id)

    # Edges with per-end ports
    for link in data['links']:
        n1, n2 = link['node1'], link['node2']
        p1, p2 = link.get('port1', 'N/A'), link.get('port2', 'N/A')
        G.add_edge(n1, n2, ports={n1: p1, n2: p2})
    return G, roles


def _sort_numeric_suffix(nodes):
    def key(n):
        digits = ''.join([c for c in n if c.isdigit()])
        return int(digits) if digits else 0
    return sorted(nodes, key=key)


def assign_positions(roles, layer_gap=2.2, node_gap=1.8):
    """
    Layered positions: hosts (0), ToR (1), Agg (2), Core (3).
    Evenly spreads each layer horizontally based on its count.
    """
    layers = [
        ("host", 0),
        ("tor", 1),
        ("agg", 2),
        ("core", 3),
    ]
    pos = {}

    # Find max width to center all layers
    counts = []
    for name, _ in layers:
        counts.append(len(roles.get(name, [])))
    max_count = max(counts) if counts else 1
    total_width = max(1, max_count - 1) * node_gap

    for name, ly in layers:
        nodes = _sort_numeric_suffix(roles.get(name, []))
        n = len(nodes)
        if n == 0:
            continue
        if n == 1:
            xs = [total_width / 2.0]
        else:
            xs = [i * (total_width / (n - 1)) for i in range(n)]
        y = ly * layer_gap
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)
    return pos


def draw_nodes_by_role(G, pos, ax):
    # group nodes by role for shapes & colors
    buckets = defaultdict(list)
    for n in G.nodes:
        buckets[role_of(n)].append(n)

    for r, nodes in buckets.items():
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes,
            node_size=800,
            node_color=PALETTE.get(r, PALETTE["other"]),
            edgecolors='black',
            linewidths=1.0,
            node_shape=SHAPES.get(r, "o"),
            ax=ax
        )


def draw_edges_and_labels(G, pos, ax, port_label=True, curved=True):
    # Slight curvature to reduce overlaps
    connectionstyle = 'arc3,rad=0.08' if curved else None
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=1.6,
        edge_color=(0.7, 0.7, 0.7, 0.9),
        arrows=False,
        connectionstyle=connectionstyle,
    )

    if port_label:
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            ports = data.get('ports', {})
            pu, pv = ports.get(u, 'N/A'), ports.get(v, 'N/A')
            edge_labels[(u, v)] = f"{pu}  ↔  {pv}"

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=8,
            font_color="#ef4444",
            bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.55), ec="none"),
            rotate=False
        )


def draw_node_labels(G, pos, ax):
    # Slightly smaller, bold labels
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in G.nodes},
        font_size=9, font_weight='bold', ax=ax
    )


def add_legend(ax):
    # Build a compact legend
    from matplotlib.lines import Line2D
    handles = []
    order = ["host", "tor", "agg", "core"]
    for r in order:
        handles.append(Line2D(
            [0], [0],
            marker=SHAPES[r], color='w',
            markerfacecolor=PALETTE[r],
            markeredgecolor='black',
            markersize=12, linewidth=0,
            label=r.upper()
        ))
    ax.legend(
        handles=handles, title="Node Types",
        loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=4, frameon=False, fontsize=9, title_fontsize=9
    )


def highlight_path(ax, pos, path, color="#10b981", lw=3.0):
    """
    Optionally emphasize a path (list of node ids).
    """
    if not path or len(path) < 2: return
    for a, b in zip(path[:-1], path[1:]):
        xs = [pos[a][0], pos[b][0]]
        ys = [pos[a][1], pos[b][1]]
        ax.plot(xs, ys, linewidth=lw, color=color, alpha=0.9, zorder=5)


def plot_graph(G, pos, title="Fat-Tree Network Topology",
               highlight=None, save=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    draw_nodes_by_role(G, pos, ax)
    draw_edges_and_labels(G, pos, ax, port_label=True, curved=True)
    draw_node_labels(G, pos, ax)

    if highlight:
        highlight_path(ax, pos, highlight)

    add_legend(ax)
    ax.set_title(title, fontsize=16, pad=18)
    ax.set_axis_off()
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ---- CLI ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Pretty plot a fat-tree topology.json")
    p.add_argument("-f", "--file", default="topology.json", help="Path to topology.json")
    p.add_argument("-o", "--out", default=None, help="Save figure to file (png/pdf/svg)")
    p.add_argument("--no-show", action="store_true", help="Do not open a window")
    p.add_argument("--title", default="Fat-Tree Network Topology", help="Plot title")
    p.add_argument("--layer-gap", type=float, default=2.2, help="Vertical spacing between layers")
    p.add_argument("--node-gap", type=float, default=1.8, help="Horizontal spacing within a layer")
    p.add_argument("--highlight", nargs="*", help="Optional path to highlight, e.g. t1 a1 c1")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_topology(args.file)
    G, roles = build_graph(data)
    pos = assign_positions(roles, layer_gap=args.layer_gap, node_gap=args.node_gap)
    plot_graph(G, pos, title=args.title, highlight=args.highlight, save=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
