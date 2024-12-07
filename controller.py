import networkx as nx
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI


class Controller:

    def __init__(self):
        self.topo = load_topo("topology.json")
        self.controllers = {}
        self.forwarding_entries = {}
        self.net_graph = nx.Graph()
        self.paths = []  # List to store paths between hosts

        self.connect_to_switches()
        self.build_network_graph()
        self.compute_forwarding_entries()
        self.program_switches()

    def connect_to_switches(self):
        """
        Connect to all P4 switches in the topology using Thrift APIs.
        """
        for sw_name in self.topo.get_p4switches().keys():
            thrift_port = self.topo.get_thrift_port(sw_name)
            self.controllers[sw_name] = SimpleSwitchThriftAPI(thrift_port)

    def build_network_graph(self):
        """
        Build a network graph using NetworkX for path computation.
        """
        # Extract node names from the dictionaries
        p4switches = list(self.topo.get_p4switches().keys())
        hosts = list(self.topo.get_hosts().keys())
        nodes = p4switches + hosts
        self.net_graph.add_nodes_from(nodes)

        # Add edges based on links in the topology
        for node in nodes:
            neighbors = self.topo.get_neighbors(node)
            for neighbor in neighbors:
                # Add edge if it doesn't exist
                if not self.net_graph.has_edge(node, neighbor):
                    self.net_graph.add_edge(node, neighbor, weight=1)

    def compute_forwarding_entries(self):
        """
        Compute forwarding entries based on shortest paths between hosts.
        """
        hosts = list(self.topo.get_hosts().keys())
        #           EF(184), CS3(96), AF21(72), 00
        dscp_list = ["0x2E", "0x18", "0x12", "0x00"]

        for src_host in hosts:
            for dst_host in hosts:
                if src_host == dst_host:
                    continue

                # Compute the shortest path between src_host and dst_host
                try:
                    path = nx.shortest_path(self.net_graph, src_host, dst_host, weight='weight')
                except nx.NetworkXNoPath:
                    print(f"No path between {src_host} and {dst_host}")
                    continue

                # Store the path in the list
                path_str = f"Path from {src_host} to {dst_host}: {' -> '.join(path)}"
                self.paths.append(path_str)

                # Iterate over the path and generate forwarding entries
                for i in range(1, len(path) - 1):  # Exclude src_host and dst_host
                    sw_name = path[i]
                    next_hop = path[i + 1]

                    # Ensure the node is a P4 switch
                    if sw_name not in self.topo.get_p4switches().keys():
                        continue

                    # Get destination IP
                    dst_ip = self.topo.get_host_ip(dst_host).split('/')[0]

                    # Get egress port
                    egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
                    port_smac = self.topo.node_to_node_mac(sw_name, next_hop)

                    # Determine next_hop_mac, dst_prefix, and next_hop_ip
                    if next_hop == dst_host:
                        # Last hop to the host; use the host's MAC address
                        next_hop_mac = self.topo.get_host_mac(dst_host)
                        dst_prefix = f"{dst_ip}/32"
                        next_hop_ip = dst_ip
                    else:
                        next_hop_mac = self.topo.node_to_node_mac(next_hop, sw_name)
                        dst_prefix = f"{dst_ip}/24"
                        edge_data = self.net_graph.get_edge_data(sw_name, next_hop)
                        #next_hop_ip = edge_data.get('ip', "0.0.0.0")  # Default to 0.0.0.0 if no IP is found
                        next_hop_ip_with_prefix = self.topo.node_to_node_interface_ip(next_hop, sw_name)
                        next_hop_ip = next_hop_ip_with_prefix.split('/')[0]
                        
                    for dscp in dscp_list:
                        # Create forwarding entry
                        entry = {
                            'dst_prefix': dst_prefix,
                            'dscp': dscp,
                            'next_hop_mac': next_hop_mac,
                            'egress_port': egress_port,
                            'next_hop_ip': next_hop_ip,
                            'port_smac': port_smac
                        }

                        if sw_name not in self.forwarding_entries:
                            self.forwarding_entries[sw_name] = []

                        # Avoid duplicates
                        if entry not in self.forwarding_entries[sw_name]:
                            self.forwarding_entries[sw_name].append(entry)


    def program_switches(self):
        """
        Program each switch with the computed forwarding entries.
        """
        for sw_name, entries in self.forwarding_entries.items():
            controller = self.controllers[sw_name]

            controller.table_set_default("l3_forward.ipv4_lpm", "drop")

            for entry in entries:
                dst_prefix = entry['dst_prefix']
                dscp = entry['dscp']
                next_hop_ip = entry['next_hop_ip']
                next_hop_mac = entry['next_hop_mac']
                egress_port = entry['egress_port']
                egress_port_hex = f"0x{egress_port:x}"
                port_smac = entry['port_smac']
                print(f"Adding entry: dst_prefix={dst_prefix}, dscp={dscp}, next_hop_mac={next_hop_mac}, "
                f"egress_port={egress_port_hex}, next_hop_ip={next_hop_ip}, port_smac={port_smac}")
                print("add LPM")
                controller.table_add(
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_ip, str(egress_port)]
                )

                print("SWITCHING TABLE")
                controller.table_add(
                    "port_forward.switching_table",
                    "set_dmac",
                    [next_hop_ip],
                    [next_hop_mac]
                )

                print("MAC REWRITE")
                controller.table_add(
                    "port_forward.mac_rewriting_table",
                    "set_smac",
                    [egress_port_hex],
                    [port_smac]
                )

    def print_paths(self):
        """
        Print the stored paths between hosts.
        """
        for path_str in self.paths:
            print(path_str)

    def print_forwarding_entries(self):
        """
        Optional: Print the forwarding entries for debugging purposes.
        """
        for sw_name, entries in self.forwarding_entries.items():
            print(f"Switch: {sw_name}")
            for entry in entries:
                print(f"  {entry}")


if __name__ == "__main__":
    controller = Controller()
    print("\n\nSUMMARY:")
    print("\nOSPF Shortest Paths:")
    controller.print_paths()  # Print the stored paths
    print("\nP4 Table Entries:")
    controller.print_forwarding_entries()  # Print forwarding entries
