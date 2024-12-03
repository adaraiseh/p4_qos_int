import networkx as nx
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI


class Controller:

    def __init__(self):
        self.topo = load_topo("topology.json")
        self.controllers = {}
        self.forwarding_entries = {}
        self.net_graph = nx.Graph()

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

                # Iterate over the path and generate forwarding entries
                for i in range(1, len(path) - 1):  # Exclude src_host and dst_host
                    sw_name = path[i]
                    next_hop = path[i + 1]

                    # Ensure the node is a P4 switch
                    if sw_name not in self.topo.get_p4switches().keys():
                        continue

                    # Get destination IP
                    dst_ip = self.topo.get_host_ip(dst_host).split('/')[0]
                    dscp = "0x00"

                    # Get egress port
                    egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)

                    if next_hop == dst_host:
                        # Last hop to the host; use the host's MAC address
                        next_hop_mac = self.topo.get_host_mac(dst_host)
                        dst_prefix = f"{dst_ip}/32"
                    else:
                        # Transit node; set MAC address to zero
                        next_hop_mac = "00:00:00:00:00:00"
                        dst_prefix = f"{dst_ip}/24"

                    # Create forwarding entry
                    entry = {
                        'dst_prefix': dst_prefix,
                        'dscp': dscp,
                        'next_hop_mac': next_hop_mac,
                        'egress_port': egress_port
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
                next_hop_mac = entry['next_hop_mac']
                egress_port = entry['egress_port']

                controller.table_add(
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_mac, str(egress_port)]
                )

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
    controller.print_forwarding_entries()