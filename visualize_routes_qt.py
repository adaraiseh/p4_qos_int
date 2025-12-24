#!/usr/bin/env python3
"""
Real-Time Network Route Visualization using PyQtGraph.

This is a high-performance alternative to visualize_routes.py (Matplotlib).
PyQtGraph is optimized for real-time data visualization with significantly
lower CPU usage.

Usage:
    python visualize_routes_qt.py

Dependencies:
    pip install pyqtgraph PyQt5
"""

import json
import os
import sys
from collections import defaultdict

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

# Enable Antialiasing
pg.setConfigOptions(antialias=True)

# === Configuration ===
TOPOLOGY_FILE = "topology.json"
PATHS_FILE = "/tmp/p4_paths.json"
REFRESH_INTERVAL_MS = 1000

# Styles - White Theme (Matched to visualize_routes.py)
BG_COLOR = "#ffffff" 
TEXT_COLOR = "#000000"

# Queue Colors (Matched)
QUEUE_COLORS = {
    0: "#d62728",    # Voice (Red)
    1: "#2ca02c",    # Video (Green)
    7: "#1f77b4"     # Best Effort (Blue)
}
QUEUE_NAMES = {
    0: "Voice (Q0)",
    1: "Video (Q1)",
    7: "Best Effort (Q7)",
}

# Node Colors by Role (Matched)
ROLE_COLORS = {
    "host": "#cccccc",
    "tor":  "#17becf",
    "agg":  "#1f77b4",
    "core": "#9467bd",
    "switch": "gray",
    "unknown": "lightgray"
}

# Layout Configuration
NODE_SIZE = 60 # Increased size (approx matching 1200 area in matplotlib)
LAYER_Y = {
    "host": 0,
    "tor": 100,
    "agg": 200,
    "core": 300,
    "switch": 150,
    "unknown": 50
}


class NetworkGraph(pg.GraphItem):
    """Custom GraphItem for network topology with traffic overlay."""
    
    def __init__(self):
        super().__init__()
        self.scatter = pg.ScatterPlotItem()
        self.node_positions = {}
        self.node_roles = {}
        self.edges = []
        self.node_list = []
        
    def set_data(self, pos, adj, node_colors, node_sizes, pen=None, **kwargs):
        """Override to store positions for edge drawing."""
        super().setData(pos=pos, adj=adj, pen=pen, **kwargs)
        

class NetworkVisualizer(QtWidgets.QMainWindow):
    """Main window for network visualization."""
    
    def __init__(self, topo_file):
        super().__init__()
        self.topo_file = topo_file
        
        # Data structures
        self.nodes = []
        self.node_positions = {}
        self.node_roles = {}
        self.edges = []
        self.paths_data = {}
        
        # Visibility state for toggle buttons
        self.visibility = {
            'phy': True,
            0: True,
            1: True,
            7: True
        }
        
        # Load topology
        self.load_topology()
        self.compute_layout()
        
        # Setup UI
        self.setup_ui()
        
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(REFRESH_INTERVAL_MS)
        
        # Initial draw
        self.draw_static_elements()
        
    def load_topology(self):
        """Parse topology.json to build the graph."""
        if not os.path.exists(self.topo_file):
            print(f"Error: {self.topo_file} not found.")
            sys.exit(1)
            
        with open(self.topo_file, 'r') as f:
            data = json.load(f)
            
        # Process nodes
        for node in data.get("nodes", []):
            nid = node["id"]
            
            # Filter INT collectors (h100+)
            if node.get("isHost"):
                try:
                    hid = int(nid[1:])
                    if hid >= 100:
                        continue
                except ValueError:
                    pass
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
                self.node_roles[nid] = "unknown"
                
            self.nodes.append(nid)
            
        # Process edges
        node_set = set(self.nodes)
        for link in data.get("links", []):
            u, v = link["node1"], link["node2"]
            if u in node_set and v in node_set:
                self.edges.append((u, v))
                
    def compute_layout(self):
        """Compute Fat-Tree layout positions."""
        nodes_by_role = defaultdict(list)
        for n in self.nodes:
            nodes_by_role[self.node_roles[n]].append(n)
        for role in nodes_by_role:
            nodes_by_role[role].sort()
            
        width = 600  # Pixel width for layout
        
        # Core nodes (top)
        c_nodes = nodes_by_role["core"]
        if c_nodes:
            dx = width / (len(c_nodes) + 1)
            for i, n in enumerate(c_nodes):
                self.node_positions[n] = ((i + 1) * dx, LAYER_Y["core"])
                
        # Pod centers
        pod1_center = width * 0.25
        pod2_center = width * 0.75
        
        # Aggregation nodes
        agg_spacing = 60
        for n in nodes_by_role["agg"]:
            if n in ["a1", "a2"]:
                offset = -agg_spacing/2 if n == "a1" else agg_spacing/2
                self.node_positions[n] = (pod1_center + offset, LAYER_Y["agg"])
            elif n in ["a3", "a4"]:
                offset = -agg_spacing/2 if n == "a3" else agg_spacing/2
                self.node_positions[n] = (pod2_center + offset, LAYER_Y["agg"])
            else:
                self.node_positions[n] = (width/2, LAYER_Y["agg"])
                
        # ToR nodes
        tor_spacing = 100
        for n in nodes_by_role["tor"]:
            if n in ["t1", "t2"]:
                offset = -tor_spacing/2 if n == "t1" else tor_spacing/2
                self.node_positions[n] = (pod1_center + offset, LAYER_Y["tor"])
            elif n in ["t3", "t4"]:
                offset = -tor_spacing/2 if n == "t3" else tor_spacing/2
                self.node_positions[n] = (pod2_center + offset, LAYER_Y["tor"])
            else:
                self.node_positions[n] = (width/2, LAYER_Y["tor"])
                
        # Host nodes (connected to ToRs)
        for tor in nodes_by_role["tor"]:
            if tor not in self.node_positions:
                continue
            tor_x, tor_y = self.node_positions[tor]
            
            # Find hosts connected to this ToR
            hosts = []
            for u, v in self.edges:
                if u == tor and self.node_roles.get(v) == "host":
                    hosts.append(v)
                elif v == tor and self.node_roles.get(u) == "host":
                    hosts.append(u)
            hosts.sort()
            
            if hosts:
                host_spread = 40
                start_x = tor_x - (len(hosts) - 1) * host_spread / 2
                for i, h in enumerate(hosts):
                    self.node_positions[h] = (start_x + i * host_spread, LAYER_Y["host"])
                    
    def setup_ui(self):
        """Setup the PyQt5 UI."""
        self.setWindowTitle("Real-Time Network Traffic Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QtWidgets.QHBoxLayout(central)
        
        # Control panel (left side)
        control_panel = QtWidgets.QVBoxLayout()
        control_panel.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("Toggle Layers")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        control_panel.addWidget(title)
        
        # Toggle buttons
        self.toggle_buttons = {}
        
        # Physical links toggle
        btn_phy = QtWidgets.QPushButton("Physical Links")
        btn_phy.setCheckable(True)
        btn_phy.setChecked(True)
        btn_phy.setStyleSheet(self._get_button_style("#333333", True))
        btn_phy.clicked.connect(lambda: self._toggle_visibility('phy', btn_phy, "#333333"))
        self.toggle_buttons['phy'] = btn_phy
        control_panel.addWidget(btn_phy)
        
        # Queue toggles
        for qid in [0, 1, 7]:
            color = QUEUE_COLORS[qid]
            btn = QtWidgets.QPushButton(QUEUE_NAMES[qid])
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(self._get_button_style(color, True))
            btn.clicked.connect(lambda checked, q=qid, b=btn, c=color: self._toggle_visibility(q, b, c))
            self.toggle_buttons[qid] = btn
            control_panel.addWidget(btn)
            
        control_panel.addStretch()
        
        # Stats display
        self.stats_label = QtWidgets.QLabel("Flows: 0")
        self.stats_label.setStyleSheet("font-size: 12px; color: #666;")
        control_panel.addWidget(self.stats_label)
        
        # Add control panel to layout
        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(180)
        layout.addWidget(control_widget)
        
        # Graphics view (right side)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')  # White background
        layout.addWidget(self.graphics_widget, stretch=1)
        
        # Create plot
        self.plot = self.graphics_widget.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setTitle("Network Topology", color='k', size='14pt')
        
        # Store plot items for updates
        self.static_edge_items = []
        self.traffic_edge_items = []
        self.node_scatter = None
        self.node_labels = []
        
    def _get_button_style(self, color, active):
        """Generate button stylesheet."""
        if active:
            return f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: 2px solid {color};
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: white;
                    color: {color};
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    padding: 8px;
                }}
                QPushButton:hover {{
                    border-color: {color};
                }}
            """
            
    def _toggle_visibility(self, key, button, color):
        """Handle visibility toggle."""
        self.visibility[key] = button.isChecked()
        button.setStyleSheet(self._get_button_style(color, button.isChecked()))
        self.update_visualization()
        
    def draw_static_elements(self):
        """Draw static topology elements (nodes and physical links)."""
        # Draw physical links
        self._draw_physical_links()
        
        # Draw nodes
        self._draw_nodes()
        
    def _draw_physical_links(self):
        """Draw physical network links."""
        # Clear existing
        for item in self.static_edge_items:
            self.plot.removeItem(item)
        self.static_edge_items.clear()
        
        if not self.visibility['phy']:
            return
            
        for u, v in self.edges:
            if u in self.node_positions and v in self.node_positions:
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                
                line = pg.PlotDataItem(
                    [x1, x2], [y1, y2],
                    pen=pg.mkPen(color='#e5e5e5', width=1)
                )
                self.plot.addItem(line)
                self.static_edge_items.append(line)
                
    def _draw_nodes(self):
        """Draw network nodes."""
        # Remove existing scatter and labels
        if self.node_scatter:
            self.plot.removeItem(self.node_scatter)
        for label in self.node_labels:
            self.plot.removeItem(label)
        self.node_labels.clear()
        
        # Prepare node data as spots list (PyQtGraph format)
        spots = []
        
        for node in self.nodes:
            if node in self.node_positions:
                x, y = self.node_positions[node]
                role = self.node_roles.get(node, "unknown")
                color = ROLE_COLORS.get(role, "#888888")
                
                spots.append({
                    'pos': (x, y),
                    'size': NODE_SIZE,
                    'pen': pg.mkPen('k', width=2),
                    'brush': pg.mkBrush(color),
                    'data': node
                })
                
        # Create scatter plot for nodes
        self.node_scatter = pg.ScatterPlotItem()
        self.node_scatter.addPoints(spots)
        self.plot.addItem(self.node_scatter)
        
        # Add labels
        for i, node in enumerate(self.nodes):
            if node in self.node_positions:
                x, y = self.node_positions[node]
                label = pg.TextItem(node, color='k', anchor=(0.5, 0.5))
                # Font size doubled (approx) from 9 to 16
                label.setFont(QtGui.QFont('Arial', 16, QtGui.QFont.Bold))
                label.setPos(x, y)
                self.plot.addItem(label)
                self.node_labels.append(label)
                
    def update_paths_data(self):
        """Read latest paths from JSON file."""
        if not os.path.exists(PATHS_FILE):
            self.paths_data = {}
            return
            
        try:
            with open(PATHS_FILE, 'r') as f:
                self.paths_data = json.load(f)
        except Exception:
            pass
            
    def update_visualization(self):
        """Update the visualization with latest data."""
        self.update_paths_data()
        
        # Redraw physical links (in case visibility changed)
        self._draw_physical_links()
        
        # Clear existing traffic overlays
        for item in self.traffic_edge_items:
            self.plot.removeItem(item)
        self.traffic_edge_items.clear()
        
        # Count traffic per link per queue
        link_counts = defaultdict(lambda: defaultdict(int))
        total_flows = 0
        
        for qid_str, flows in self.paths_data.items():
            qid = int(qid_str)
            if not self.visibility.get(qid, True):
                continue
                
            for flow in flows:
                path = flow.get("path", [])
                if len(path) < 2:
                    continue
                total_flows += 1
                    
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if u not in self.node_positions or v not in self.node_positions:
                        continue
                    key = tuple(sorted((u, v)))
                    link_counts[key][qid] += 1
                    
        # Update stats
        self.stats_label.setText(f"Active Flows: {total_flows}")
        
        # Calculate max counts for normalization
        max_counts = {qid: 1 for qid in QUEUE_COLORS}
        for q_counts in link_counts.values():
            for qid, count in q_counts.items():
                max_counts[qid] = max(max_counts[qid], count)
                
        # Draw traffic overlays
        queue_offsets = {0: -8, 1: 0, 7: 8}  # Pixel offsets
        
        for (u, v), q_counts in link_counts.items():
            x1, y1 = self.node_positions[u]
            x2, y2 = self.node_positions[v]
            
            # Normalize direction for consistent offsets
            if y1 > y2 or (abs(y1 - y2) < 0.01 and x1 > x2):
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            dx, dy = x2 - x1, y2 - y1
            length = (dx*dx + dy*dy)**0.5
            if length == 0:
                continue
                
            # Normal vector for offset direction
            nx = -dy / length
            ny = dx / length
            
            for qid, count in q_counts.items():
                if not self.visibility.get(qid, True) or count == 0:
                    continue
                    
                color = QUEUE_COLORS.get(qid, "#888888")
                offset = queue_offsets.get(qid, 0)
                
                # Apply offset
                ox1 = x1 + nx * offset
                oy1 = y1 + ny * offset
                ox2 = x2 + nx * offset
                oy2 = y2 + ny * offset
                
                # Line width based on count (Matched to visualize_routes.py logic, scaled for screen)
                # Original: 1.0 + min(count, 10) * 0.4
                # We scale by 2.0 to make it look similar in pixel thickness
                base_width = 1.0 + min(count, 10) * 0.4
                width = base_width * 2.0 
                
                # Alpha based on intensity (Matched to visualize_routes.py logic)
                # Original: 0.5 + 0.5 * ratio
                ratio = count / max_counts[qid]
                alpha_val = 0.5 + 0.5 * ratio
                alpha = int(255 * alpha_val)
                
                # Create colored line with curve
                # pen = pg.mkPen(color=color + "{:02x}".format(alpha), width=width)
                # line = pg.PlotDataItem([ox1, ox2], [oy1, oy2], pen=pen)
                
                # Use Bezier curve
                xs, ys = self.get_curve_points(ox1, oy1, ox2, oy2)
                pen = pg.mkPen(color=color + "{:02x}".format(alpha), width=width)
                line = pg.PlotDataItem(xs, ys, pen=pen)
                
                self.plot.addItem(line)
                self.traffic_edge_items.append(line)
                
        # Ensure nodes are on top by re-adding them
        if self.node_scatter:
            self.plot.removeItem(self.node_scatter)
            self.plot.addItem(self.node_scatter)
        for label in self.node_labels:
            self.plot.removeItem(label)
            self.plot.addItem(label)
            
    def get_curve_points(self, x1, y1, x2, y2, curvature=0.08):
        """Generate smooth curve points using quadratic Bezier."""
        # Midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Vector
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2)**0.5
        
        if dist == 0:
            return np.array([x1]), np.array([y1])
            
        # Normal vector (orthogonal)
        nx, ny = -dy / dist, dx / dist
        
        # Control point (midpoint + offset)
        cx = mx + nx * (dist * curvature)
        cy = my + ny * (dist * curvature)
        
        # Parameter t from 0 to 1
        t = np.linspace(0, 1, 50)
        
        # Quadratic Bezier formula
        # B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        # optimized: P0 + t * (2 * (1-t) * (P1 - P0) + t * (P2 - P0)) 
        # But explicit is clearer:
        mt = 1 - t
        sx = (mt**2 * x1) + (2 * mt * t * cx) + (t**2 * x2)
        sy = (mt**2 * y1) + (2 * mt * t * cy) + (t**2 * y2)
        
        return sx, sy


def main():
    """Main entry point."""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show visualizer
    vis = NetworkVisualizer(TOPOLOGY_FILE)
    vis.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
