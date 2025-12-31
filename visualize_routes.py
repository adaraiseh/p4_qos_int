#!/usr/bin/env python3
"""
Network Traffic Visualizer

Real-time visualization of network traffic flows across different topology types.
Supports Fat-Tree, Leaf-Spine, and Three-Tier topologies with automatic layout.
"""
import json
import time
import os
import signal
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import math
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, Rectangle
from typing import Dict, List, Optional, Tuple

# === Configuration ===
DEFAULT_TOPOLOGY_FILE = "topology.json"
PATHS_FILE = "/tmp/p4_paths.json"
REFRESH_INTERVAL_MS = 1000


def get_topology_info(config_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Get topology type and switch roles from configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (topology_type, switch_roles_dict)
        topology_type: 'fat-tree', 'leaf-spine', or 'three-tier'
        switch_roles_dict: {switch_name: role} e.g., {'leaf1': 'leaf', 'spine1': 'spine'}
    """
    try:
        from topology.factory import create_topology

        builder = create_topology(config_path)
        topo_type = builder.config.topology.type.value

        # Build switch roles from builder
        switch_roles = {}
        for sw_name, sw_info in builder.switches.items():
            switch_roles[sw_name] = sw_info.role

        return topo_type, switch_roles

    except Exception as e:
        print(f"Warning: Could not load topology config: {e}")
        return 'fat-tree', {}  # Default fallback

# Styles - White Theme
BG_COLOR = "#ffffff"
LINK_COLOR = "#e5e5e5" # Very light grey
TEXT_COLOR = "#000000"

# Node Visuals
NODE_SIZE = 1200
# Layout is width=24. Node size 1200 ~= radius 20pts.
# In data coords (approx): 
NODE_RADIUS = 0.35 

# Matplotlib Colors (Tab10 / Standard) for White BG
QUEUE_COLORS = {
    0: "#d62728",    # Voice (Red)
    1: "#2ca02c",    # Video (Green)
    7: "#1f77b4"     # Best Effort (Blue)
}
QUEUE_NAMES = {
    0: "Voice (Q0)",
    1: "Video (Q1)",
    7: "Best Effort (Q7)"
}

# Hide default toolbar
plt.rcParams['toolbar'] = 'None'

class NetworkVisualizer:
    # Role normalization for visualization layers
    EDGE_ROLES = {'leaf', 'tor', 'access'}  # Y=2
    AGG_ROLES = {'spine', 'agg', 'distribution'}  # Y=4
    CORE_ROLES = {'core'}  # Y=6

    def __init__(self, topo_file: str, config_path: str = None):
        """
        Initialize the network visualizer.

        Args:
            topo_file: Path to topology.json (e.g., /tmp/topology.json)
            config_path: Optional path to YAML topology configuration
        """
        self.topo_file = topo_file
        self.config_path = config_path
        self.graph = nx.Graph()
        self.pos = {}
        self.node_roles = {}

        # Load topology type and switch roles from config if provided
        self.topology_type = 'fat-tree'
        self.switch_roles_from_config = {}
        if config_path:
            self.topology_type, self.switch_roles_from_config = get_topology_info(config_path)
            print(f"Visualization: Loaded topology type '{self.topology_type}' from config")

        # UI State
        self.visibility = {
            'phy': True,
            0: True,
            1: True,
            7: True
        }

        self.load_topology()
        self.compute_layout()

        # Plot setup
        self.fig, self.ax = plt.subplots(figsize=(16, 12))  # Larger size
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)

        # Button State (for hit testing)
        self.ui_buttons = []  # List of (x, y, w, h, key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # State
        self.paths_data = {}
        
    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        # Check clicks against buttons
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        for (bx, by, bw, bh, key) in self.ui_buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.visibility[key] = not self.visibility[key]
                return # Handled
        
    def _get_switch_role(self, switch_name: str) -> str:
        """
        Get the visualization role for a switch.

        Uses roles from config if available, otherwise falls back to prefix-based detection.

        Returns normalized role: 'edge', 'agg', 'core', or 'switch'
        """
        # Try to get role from config
        if switch_name in self.switch_roles_from_config:
            role = self.switch_roles_from_config[switch_name]
            # Normalize role to visualization category
            if role in self.EDGE_ROLES:
                return 'edge'
            elif role in self.AGG_ROLES:
                return 'agg'
            elif role in self.CORE_ROLES:
                return 'core'
            return 'switch'

        # Fallback: prefix-based detection (legacy)
        if switch_name.startswith("c") or switch_name.startswith("core"):
            return "core"
        elif switch_name.startswith("a") or switch_name.startswith("spine") or switch_name.startswith("dist"):
            return "agg"
        elif switch_name.startswith("t") or switch_name.startswith("leaf") or switch_name.startswith("access"):
            return "edge"
        return "switch"

    def load_topology(self):
        """Parse topology.json to build the graph and identify roles."""
        if not os.path.exists(self.topo_file):
            print(f"Error: {self.topo_file} not found.")
            sys.exit(1)

        with open(self.topo_file, 'r') as f:
            data = json.load(f)

        # Add nodes
        nodes_to_add = []
        for node in data.get("nodes", []):
            nid = node["id"]

            # Filter INT collectors (h100+)
            if node.get("isHost"):
                try:
                    # Assumes id format 'h<number>'
                    hid = int(nid[1:])
                    if hid >= 100:
                        continue  # Skip this node
                except ValueError:
                    pass  # Not a host with numeric ID, or malformed, process normally
                self.node_roles[nid] = "host"
            elif node.get("isSwitch"):
                # Use config-based or fallback role detection
                self.node_roles[nid] = self._get_switch_role(nid)
            else:
                # If not host or switch, assign a default role or skip
                self.node_roles[nid] = "unknown"

            # Only add to graph if not filtered
            nodes_to_add.append(nid)
            self.graph.add_node(nid)

        # Add links
        for link in data.get("links", []):
            u, v = link["node1"], link["node2"]
            # Only add if both nodes exist (were not filtered)
            if self.graph.has_node(u) and self.graph.has_node(v):
                self.graph.add_edge(u, v)

    def compute_layout(self):
        """
        Compute automatic layout based on topology type.

        Supports Fat-Tree, Leaf-Spine, and Three-Tier topologies.
        Layout structure:
          - Y=0: hosts
          - Y=2: edge switches (leaf/tor/access)
          - Y=4: aggregation switches (spine/agg/distribution)
          - Y=6: core switches
        """
        # Y-coordinates (Layers) - unified for all topology types
        layer_y = {
            "host": 0,
            "edge": 2,      # leaf, tor, access
            "agg": 4,       # spine, agg, distribution
            "core": 6,
            "switch": 3,    # unknown role
            "unknown": 1
        }

        # Group nodes by their visualization role
        nodes_by_role = {role: [] for role in layer_y.keys()}
        for n, r in self.node_roles.items():
            if self.graph.has_node(n):
                nodes_by_role[r].append(n)

        # Sort nodes for consistent ordering
        for r in nodes_by_role:
            nodes_by_role[r].sort(key=self._node_sort_key)

        width = 24.0

        # Dispatch to topology-specific layout
        if self.topology_type == 'leaf-spine':
            self._layout_leaf_spine(nodes_by_role, layer_y, width)
        elif self.topology_type == 'three-tier':
            self._layout_three_tier(nodes_by_role, layer_y, width)
        else:
            # Default: Fat-Tree or generic hierarchical
            self._layout_fat_tree(nodes_by_role, layer_y, width)

    def _node_sort_key(self, node_name: str):
        """Sort key for node ordering - extract numeric suffix."""
        import re
        match = re.search(r'(\d+)$', node_name)
        if match:
            return (node_name[0], int(match.group(1)))
        return (node_name, 0)

    def _layout_leaf_spine(self, nodes_by_role: dict, layer_y: dict, width: float):
        """
        Leaf-Spine topology layout.

        Structure: Hosts -> Leaves -> Spines (evenly distributed)
        """
        # Spines at top (Y=4)
        spines = nodes_by_role["agg"]
        if spines:
            dx = width / (len(spines) + 1)
            for i, n in enumerate(spines):
                self.pos[n] = ((i + 1) * dx, layer_y["agg"])

        # Leaves at Y=2 (evenly distributed)
        leaves = nodes_by_role["edge"]
        if leaves:
            dx = width / (len(leaves) + 1)
            for i, n in enumerate(leaves):
                self.pos[n] = ((i + 1) * dx, layer_y["edge"])

        # Hosts below their connected leaf
        self._layout_hosts_under_switches(nodes_by_role, layer_y)

    def _layout_three_tier(self, nodes_by_role: dict, layer_y: dict, width: float):
        """
        Three-Tier topology layout.

        Structure: Hosts -> Access -> Distribution -> Core
        """
        # Core at top (Y=6)
        cores = nodes_by_role["core"]
        if cores:
            dx = width / (len(cores) + 1)
            for i, n in enumerate(cores):
                self.pos[n] = ((i + 1) * dx, layer_y["core"])

        # Distribution at Y=4
        dist = nodes_by_role["agg"]
        if dist:
            dx = width / (len(dist) + 1)
            for i, n in enumerate(dist):
                self.pos[n] = ((i + 1) * dx, layer_y["agg"])

        # Access at Y=2
        access = nodes_by_role["edge"]
        if access:
            dx = width / (len(access) + 1)
            for i, n in enumerate(access):
                self.pos[n] = ((i + 1) * dx, layer_y["edge"])

        # Hosts below their connected access switch
        self._layout_hosts_under_switches(nodes_by_role, layer_y)

    def _layout_fat_tree(self, nodes_by_role: dict, layer_y: dict, width: float):
        """
        Fat-Tree topology layout.

        Structure: Hosts -> ToR -> Aggregation -> Core (with pod grouping)
        """
        # Cores at top (Y=6) - evenly distributed
        cores = nodes_by_role["core"]
        if cores:
            dx = width / (len(cores) + 1)
            for i, n in enumerate(cores):
                self.pos[n] = ((i + 1) * dx, layer_y["core"])

        # For Fat-Tree, group edge and agg switches by pods
        edge_switches = nodes_by_role["edge"]
        agg_switches = nodes_by_role["agg"]

        # Determine number of pods based on edge switches
        num_edge = len(edge_switches)
        num_agg = len(agg_switches)

        if num_edge == 0:
            return

        # Estimate pods: k/2 ToRs per pod in k-ary fat-tree
        # Common case: k=4 has 2 ToRs per pod, k=8 has 4 per pod
        # Heuristic: assume pods if agg count matches edge count
        if num_agg > 0 and num_edge > 0:
            # Try to detect pod structure
            pods = self._detect_pods(edge_switches, agg_switches)
        else:
            pods = [[s] for s in edge_switches]

        num_pods = len(pods)
        pod_width = width / num_pods

        # Layout each pod
        for pod_idx, pod_edges in enumerate(pods):
            pod_center = (pod_idx + 0.5) * pod_width

            # Position edge switches in this pod
            edge_spacing = pod_width * 0.6 / max(1, len(pod_edges))
            edge_start = pod_center - (len(pod_edges) - 1) * edge_spacing / 2
            for i, n in enumerate(pod_edges):
                self.pos[n] = (edge_start + i * edge_spacing, layer_y["edge"])

        # Position agg switches - map to pods or distribute evenly
        if num_agg > 0:
            if num_agg == num_edge:
                # Same structure as edge (common in fat-tree)
                for pod_idx, pod_edges in enumerate(pods):
                    pod_center = (pod_idx + 0.5) * pod_width
                    # Find agg switches for this pod
                    pod_aggs = self._find_pod_aggs(pod_edges, agg_switches)
                    if pod_aggs:
                        agg_spacing = pod_width * 0.5 / max(1, len(pod_aggs))
                        agg_start = pod_center - (len(pod_aggs) - 1) * agg_spacing / 2
                        for i, n in enumerate(pod_aggs):
                            self.pos[n] = (agg_start + i * agg_spacing, layer_y["agg"])
            else:
                # Distribute agg switches evenly
                dx = width / (num_agg + 1)
                for i, n in enumerate(agg_switches):
                    if n not in self.pos:
                        self.pos[n] = ((i + 1) * dx, layer_y["agg"])

        # Hosts below their connected edge switch
        self._layout_hosts_under_switches(nodes_by_role, layer_y)

    def _detect_pods(self, edge_switches: list, agg_switches: list) -> list:
        """
        Detect pod structure by analyzing connectivity.

        Returns list of lists, where each inner list contains edge switches in a pod.
        """
        # Simple heuristic: group by connectivity to agg switches
        # If edge switches share the same agg neighbors, they're in the same pod

        pod_map = {}  # frozenset(agg_neighbors) -> [edge_switches]

        for edge in edge_switches:
            agg_neighbors = frozenset(
                n for n in self.graph.neighbors(edge)
                if n in agg_switches
            )
            if agg_neighbors:
                if agg_neighbors not in pod_map:
                    pod_map[agg_neighbors] = []
                pod_map[agg_neighbors].append(edge)
            else:
                # No agg neighbors - standalone pod
                pod_map[frozenset([edge])] = [edge]

        # Convert to list of lists, sorted by first edge switch
        pods = list(pod_map.values())
        pods.sort(key=lambda p: self._node_sort_key(p[0]) if p else ('z', 999))
        return pods

    def _find_pod_aggs(self, pod_edges: list, agg_switches: list) -> list:
        """Find agg switches connected to the given pod edge switches."""
        pod_aggs = set()
        for edge in pod_edges:
            for neighbor in self.graph.neighbors(edge):
                if neighbor in agg_switches:
                    pod_aggs.add(neighbor)
        return sorted(pod_aggs, key=self._node_sort_key)

    def _layout_hosts_under_switches(self, nodes_by_role: dict, layer_y: dict):
        """Position hosts below their connected edge switches."""
        edge_switches = nodes_by_role["edge"]
        host_spread = 2.0

        for sw in edge_switches:
            if sw not in self.pos:
                continue
            sw_x, sw_y = self.pos[sw]

            # Find hosts connected to this switch
            sw_hosts = sorted(
                [n for n in self.graph.neighbors(sw) if self.node_roles.get(n) == "host"],
                key=self._node_sort_key
            )
            if not sw_hosts:
                continue

            # Spread hosts below the switch
            start_x = sw_x - (len(sw_hosts) - 1) * host_spread / 2
            for i, h in enumerate(sw_hosts):
                self.pos[h] = (start_x + i * host_spread, layer_y["host"])

    def update_data(self):
        if not os.path.exists(PATHS_FILE):
            return
        try:
            with open(PATHS_FILE, 'r') as f:
                self.paths_data = json.load(f)
        except Exception as e:
            # print(f"Error reading paths: {e}")
            pass

    def draw_custom_ui(self):
        """Draw custom toggle buttons on the left side."""
        # Define Buttons (x, y, w, h, label, key, color)
        # Using data coordinates. The graph is roughly x=[0, 24], y=[0, 6].
        # We will extend xlim to include negative space for UI.
        
        bg_active = "#dddddd"
        bg_inactive = "#ffffff"
        
        buttons = [
            (-7, 5.0, 5, 0.8, "Physical Links", 'phy', "#333333"),
            (-7, 4.0, 5, 0.8, "Voice (Q0)", 0, QUEUE_COLORS[0]),
            (-7, 3.0, 5, 0.8, "Video (Q1)", 1, QUEUE_COLORS[1]),
            (-7, 2.0, 5, 0.8, "Best Effort (Q7)", 7, QUEUE_COLORS[7])
        ]
        
        self.ui_buttons = []
        
        for x, y, w, h, label, key, color in buttons:
            is_active = self.visibility[key]
            
            # Store for hit testing
            self.ui_buttons.append((x, y, w, h, key))
            
            # Draw Box
            # Active = Filled with Color (light alpha), Border = Color
            # Inactive = White, Border = Grey
            
            facecolor = color if is_active else "white"
            edgecolor = color if is_active else "#aaaaaa"
            alpha = 0.2 if is_active else 1.0
            linewidth = 3 if is_active else 1
            
            rect = Rectangle((x, y), w, h, 
                           facecolor=facecolor, 
                           edgecolor=edgecolor,
                           linewidth=linewidth,
                           alpha=alpha if is_active else 1.0, 
                           zorder=100) # Ensure on top
            self.ax.add_patch(rect)
            
            # Checkmark or Status Indicator
            status_text = "Are Active" if is_active else "Off"
            if is_active:
                # tick mark or just filled box? 
                # Let's simple bold text
                fontweight = 'bold'
                textcolor = color
            else:
                fontweight = 'normal'
                textcolor = "#888888"
                
            self.ax.text(x + w/2, y + h/2, label, 
                       ha='center', va='center', 
                       fontsize=12, fontweight=fontweight, color=textcolor,
                       zorder=101)

    def draw(self, frame):
        self.update_data()
        self.ax.clear()
        self.ax.axis('off')
        
        # Extend limits to show UI
        self.ax.set_xlim(-8, 26)
        self.ax.set_ylim(-1, 8)
        
        # 0. Draw Custom UI
        self.draw_custom_ui()
        
        # 1. Draw Static Links
        if self.visibility['phy']:
            edges = nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax, edge_color=LINK_COLOR, width=1.0)
            if edges: edges.set_zorder(5)
        
        # 2. Draw Nodes (BIG ICONS)
        role_colors = {
            "host": "#cccccc",
            "edge": "#17becf",    # leaf/tor/access
            "agg": "#1f77b4",     # spine/agg/distribution
            "core": "#9467bd",
            "switch": "gray",
            "unknown": "lightgray"
        }
        
        node_colors = [role_colors.get(self.node_roles.get(n, "switch"), "white") for n in self.graph.nodes()]
        
        # Draw nodes with black outline
        nodes = nx.draw_networkx_nodes(self.graph, self.pos, ax=self.ax, 
                             node_color=node_colors, 
                             node_size=NODE_SIZE, 
                             linewidths=2,
                             edgecolors='black')
        if nodes: nodes.set_zorder(20)
                             
        # Black text labels
        labels = nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax, 
                              font_size=11, 
                              font_color=TEXT_COLOR, 
                              font_family='sans-serif',
                              font_weight='bold')
        for label in labels.values():
            label.set_zorder(25)
        
        # 3. Traffic Overlay
        link_counts = {}
        for qid_s, flows in self.paths_data.items():
            qid = int(qid_s)
            
            # Check visibility for this queue
            if not self.visibility[qid]:
                continue
                
            for flow in flows:
                path = flow.get("path", [])
                if not path or len(path) < 2: continue
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    # Filter out links involving hidden nodes (e.g. if path has filtered host)
                    if u not in self.graph or v not in self.graph: 
                        continue
                        
                    # Canonical key for counting
                    key = tuple(sorted((u, v)))
                    link_counts.setdefault(key, {})
                    link_counts[key][qid] = link_counts[key].get(qid, 0) + 1

        # Max counts for normalization
        max_counts = {qid: 1 for qid in QUEUE_COLORS}
        for q_counts in link_counts.values():
            for qid, count in q_counts.items():
                max_counts[qid] = max(max_counts[qid], count)

        # Draw Lines
        for (u, v), q_counts in link_counts.items():
            if u not in self.pos or v not in self.pos: continue
            
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            
            # ORIENTATION NORMALIZATION
            # We want a consistent "Up" direction so that "Left" is always "West".
            # Fat Tree is hierarchical: Layers 0 -> 2 -> 4 -> 6.
            # If we enforce y1 < y2, we are drawing from Lower Layer to Higher Layer.
            # Normal (-dy, dx) will point Left (West).
            
            # Check for horizontal links (same layer) - rare in Fat Tree but possible (Core ring?)
            # If horizontal, sort by X to ensure consistent Left->Right flow implies consistent "Side" offset.
            if abs(y1 - y2) < 0.01: # Effectively horizontal
                if x1 > x2: # Ensure x1 is always left of x2 for horizontal links
                    x1, y1, x2, y2 = x2, y2, x1, y1
            elif y1 > y2: # Ensure y1 is always below y2 for vertical/diagonal links
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            dx, dy = x2 - x1, y2 - y1
            length = (dx*dx + dy*dy)**0.5
            if length == 0: continue
            
            # Normal Vector (Right Hand Rule relative to vector u->v)
            # vector: (dx, dy)
            # left normal in 2D: (-dy, dx)
            nx_ = -dy / length
            ny_ = dx / length
            
            # Offsets: Q0 (Left/West), Q1 (Center), Q7 (Right/East)
            queue_offsets = {0: -0.25, 1: 0.0, 7: 0.25} # Wider spacing
            
            for qid, count in q_counts.items():
                # Check visibility for this specific queue again, in case it was toggled mid-loop
                if not self.visibility[qid] or count == 0: continue
                
                # Dynamic Color
                ratio = count / max_counts[qid]
                base_color = QUEUE_COLORS.get(qid, "black")
                
                # Reduced width scaling formula
                width = 1.0 + min(count, 10) * 0.4
                
                # Alpha based on intensity
                alpha = 0.5 + 0.5 * ratio
                
                off = queue_offsets.get(qid, 0)
                
                # Apply offset to center line
                lx1 = x1 + nx_ * off
                ly1 = y1 + ny_ * off
                lx2 = x2 + nx_ * off
                ly2 = y2 + ny_ * off
                
                # Shorten the line to terminate at node CIRCLE EDGE
                # The line is parallel to center connection, offset by 'off'.
                # Triangle: R = NODE_RADIUS, d = off. 
                # Intersection distance from center perpendicular = sqrt(R^2 - off^2)
                # But we act along the vector (dx, dy).
                # The "Circle" is centered at (x1, y1). Line passes distance 'off'.
                # So the intersection chord length is 2 * sqrt(R^2 - off^2).
                # We need to move start point "forward" by sqrt(R^2 - off^2)
                # And end point "backward" by sqrt(R^2 - off^2).
                
                if abs(off) < NODE_RADIUS:
                    shorten = math.sqrt(NODE_RADIUS**2 - off**2)
                    
                    # Normalized direction vector
                    ux, uy = dx/length, dy/length
                    
                    lx1 += ux * shorten
                    ly1 += uy * shorten
                    lx2 -= ux * shorten
                    ly2 -= uy * shorten
                
                # Draw subtle curve using FancyArrowPatch
                # "Introduce small curves at the end" -> Gentle arc
                arrow = FancyArrowPatch((lx1, ly1), (lx2, ly2),
                                      arrowstyle='-',
                                      connectionstyle="arc3,rad=0.03", 
                                      color=base_color,
                                      linewidth=width,
                                      alpha=alpha,
                                      zorder=10)
                self.ax.add_patch(arrow)

        # Title (Centered Top)
        self.ax.text(12.0, 7.5, "Real-Time Network Traffic", fontsize=20, fontweight='bold', ha='center')

    def main(self):  # unused directly, compat
        pass


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time network traffic visualization"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML topology configuration for dynamic layout and role detection'
    )
    parser.add_argument(
        '--topology-file',
        type=str,
        default='/tmp/topology.json',
        help='Path to topology.json (default: /tmp/topology.json)'
    )
    parser.add_argument(
        '--paths-file',
        type=str,
        default=PATHS_FILE,
        help=f'Path to paths JSON file (default: {PATHS_FILE})'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=REFRESH_INTERVAL_MS,
        help=f'Refresh interval in milliseconds (default: {REFRESH_INTERVAL_MS})'
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Update global paths file if overridden
    global PATHS_FILE
    PATHS_FILE = args.paths_file

    vis = NetworkVisualizer(
        topo_file=args.topology_file,
        config_path=args.config
    )

    # Fix Warning: UserWarning: frames=None... passed cache_frame_data=True
    ani = FuncAnimation(vis.fig, vis.draw, interval=args.refresh, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
