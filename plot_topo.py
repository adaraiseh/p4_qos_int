import json
import networkx as nx
import matplotlib.pyplot as plt

def load_topology(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def build_graph(data):
    G = nx.Graph()
    node_labels = {}

    # Classify nodes based on their IDs
    hosts = []
    tor_switches = []
    agg_switches = []
    core_switches = []

    for node in data['nodes']:
        node_id = node['id']
        G.add_node(node_id)
        node_labels[node_id] = node_id

        if node_id.startswith('h'):
            hosts.append(node_id)
        elif node_id.startswith('t'):
            tor_switches.append(node_id)
        elif node_id.startswith('a'):
            agg_switches.append(node_id)
        elif node_id.startswith('c'):
            core_switches.append(node_id)

    # Add edges
    for link in data['links']:
        node1 = link['node1']
        node2 = link['node2']
        G.add_edge(node1, node2)

    return G, hosts, tor_switches, agg_switches, core_switches, node_labels

def assign_positions(hosts, tor_switches, agg_switches, core_switches):
    pos = {}
    layer_spacing = 2
    node_spacing = 2

    def sort_nodes(nodes):
        return sorted(nodes, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Position hosts at the bottom layer (Layer 0)
    hosts = sort_nodes(hosts)
    num_hosts = len(hosts)
    for i, host in enumerate(hosts):
        pos[host] = (i * node_spacing, 0)

    # Position ToR switches at Layer 1
    tor_switches = sort_nodes(tor_switches)
    num_tors = len(tor_switches)
    tor_spacing = node_spacing * (num_hosts - 1) / max(num_tors - 1, 1) if num_tors > 1 else 0
    for i, tor in enumerate(tor_switches):
        pos[tor] = (i * tor_spacing, layer_spacing)

    # Position Aggregation switches at Layer 2
    agg_switches = sort_nodes(agg_switches)
    num_aggs = len(agg_switches)
    agg_spacing = node_spacing * (num_hosts - 1) / max(num_aggs - 1, 1) if num_aggs > 1 else 0
    for i, agg in enumerate(agg_switches):
        pos[agg] = (i * agg_spacing, layer_spacing * 2)

    # Position Core switches at Layer 3
    core_switches = sort_nodes(core_switches)
    num_cores = len(core_switches)
    core_spacing = node_spacing * (num_hosts - 1) / max(num_cores - 1, 1) if num_cores > 1 else 0
    for i, core in enumerate(core_switches):
        pos[core] = (i * core_spacing, layer_spacing * 3)

    return pos

def plot_graph(G, pos, labels, data):
    plt.figure(figsize=(12, 8))

    # Assign colors based on node types
    node_colors = []
    for node in G.nodes():
        if node.startswith('h'):
            node_colors.append('lightgreen')
        elif node.startswith('t'):
            node_colors.append('lightblue')
        elif node.startswith('a'):
            node_colors.append('orange')
        elif node.startswith('c'):
            node_colors.append('red')
        else:
            node_colors.append('grey')

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    # Add port numbers to edges
    edge_labels = {}
    for link in data['links']:
        node1 = link['node1']
        node2 = link['node2']
        port1 = link.get('port1', 'N/A')
        port2 = link.get('port2', 'N/A')
        edge_labels[(node1, node2)] = f"{port1} - {port2}"

    # Draw edge labels without background fill
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, bbox=dict(alpha=0), font_color="blue", font_weight='bold')
    plt.title('Fat-Tree Network Topology', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = 'topology.json'
    data = load_topology(filename)
    G, hosts, tor_switches, agg_switches, core_switches, labels = build_graph(data)
    pos = assign_positions(hosts, tor_switches, agg_switches, core_switches)
    plot_graph(G, pos, labels, data)
