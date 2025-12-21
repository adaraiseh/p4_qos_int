#!/usr/bin/env python3
import json
import time
import os
import signal
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import math
from matplotlib.patches import FancyArrowPatch, Rectangle

# === Configuration ===
TOPOLOGY_FILE = "topology.json"
PATHS_FILE = "/tmp/p4_paths.json"
REFRESH_INTERVAL_MS = 1000

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
    def __init__(self, topo_file):
        self.topo_file = topo_file
        self.graph = nx.Graph()
        self.pos = {}
        self.node_roles = {}
        
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
        self.fig, self.ax = plt.subplots(figsize=(16, 12)) # Larger size
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)
        
        # Button State (for hit testing)
        self.ui_buttons = [] # List of (x, y, w, h, key)
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
                        continue # Skip this node
                except ValueError:
                    pass # Not a host with numeric ID, or malformed, process normally
                self.node_roles[nid] = "host"
            elif node.get("isSwitch"):
                if nid.startswith("c"):
                    self.node_roles[nid] = "core"
                elif nid.startswith("a"):
                    self.node_roles[nid] = "agg"
                elif nid.startswith("t"):
                    self.node_roles[nid] = "tor"
                else:
                    self.node_roles[nid] = "switch"
            else:
                # If not host or switch, assign a default role or skip
                self.node_roles[nid] = "unknown" # Or handle as needed
            
            # Only add to graph if not filtered
            nodes_to_add.append(nid)
            self.graph.add_node(nid) # Add node first to check for links

        # Add links
        for link in data.get("links", []):
            u, v = link["node1"], link["node2"]
            # Only add if both nodes exist (were not filtered)
            if self.graph.has_node(u) and self.graph.has_node(v):
                self.graph.add_edge(u, v)

    def compute_layout(self):
        """Compute a fixed Fat-Tree layout with better spacing."""
        # Y-coordinates (Layers)
        layer_y = {
            "host": 0,
            "tor": 2,
            "agg": 4,
            "core": 6,
            "switch": 3
        }
        
        nodes_by_role = {"host": [], "tor": [], "agg": [], "core": [], "switch": [], "unknown": []}
        for n, r in self.node_roles.items():
            # Only consider nodes that are actually in the graph (not filtered)
            if self.graph.has_node(n):
                nodes_by_role[r].append(n)
        for r in nodes_by_role:
            nodes_by_role[r].sort()
            
        width = 24.0 # Wider
        
        # Cores (Top)
        c_nodes = nodes_by_role["core"]
        if c_nodes:
            dx = width / (len(c_nodes) + 1)
            for i, n in enumerate(c_nodes):
                self.pos[n] = ((i + 1) * dx, layer_y["core"])
                
        # Pod Centers
        pod1_center = width * 0.25
        pod2_center = width * 0.75
        
        # Aggs (a1, a2 in Pod1; a3, a4 in Pod2)
        agg_spacing = 3.0
        for n in nodes_by_role["agg"]:
            if n in ["a1", "a2"]:
                offset = -agg_spacing/2 if n == "a1" else agg_spacing/2
                self.pos[n] = (pod1_center + offset, layer_y["agg"])
            elif n in ["a3", "a4"]:
                offset = -agg_spacing/2 if n == "a3" else agg_spacing/2
                self.pos[n] = (pod2_center + offset, layer_y["agg"])
            else:
                self.pos[n] = (width/2, layer_y["agg"])

        # ToRs (t1, t2 in Pod1; t3, t4 in Pod2)
        tor_spacing = 5.0
        for n in nodes_by_role["tor"]:
            if n in ["t1", "t2"]:
                offset = -tor_spacing/2 if n == "t1" else tor_spacing/2
                self.pos[n] = (pod1_center + offset, layer_y["tor"])
            elif n in ["t3", "t4"]:
                offset = -tor_spacing/2 if n == "t3" else tor_spacing/2
                self.pos[n] = (pod2_center + offset, layer_y["tor"])
            else:
                 self.pos[n] = (width/2, layer_y["tor"])

        # Hosts
        for sw in nodes_by_role["tor"]:
            if sw not in self.pos: continue
            sw_x, sw_y = self.pos[sw]
            sw_hosts = sorted([n for n in self.graph.neighbors(sw) if self.node_roles.get(n) == "host"])
            if not sw_hosts: continue
            
            # Spread hosts significantly
            host_spread = 2.0
            start_x = sw_x - (len(sw_hosts)-1) * host_spread / 2
            
            for i, h in enumerate(sw_hosts):
                # Droop hosts slightly below Y=0 purely for visual separation if needed, or keep at 0
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
            "host": "#cccccc",   "tor":  "#17becf",
            "agg":  "#1f77b4",   "core": "#9467bd",
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

    def main(self): # unused directly, compat
        pass

def main():
    vis = NetworkVisualizer(TOPOLOGY_FILE)
    # Fix Warning: UserWarning: frames=None... passed cache_frame_data=True
    ani = FuncAnimation(vis.fig, vis.draw, interval=REFRESH_INTERVAL_MS, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
