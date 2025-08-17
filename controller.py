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
        hosts = list(self.topo.get_hosts().keys())
        dscp_list = ["0x2E", "0x18", "0x12", "0x00"]

        # Per-switch, per-table structures
        # lpm: {(dst_prefix, dscp): (next_hop_ip, egress_port)}
        # switching: {next_hop_ip: next_hop_mac}
        # mac: {egress_port: port_smac}
        self.forwarding_entries = {}  # reset to structured maps

        for src_host in hosts:
            for dst_host in hosts:
                if src_host == dst_host:
                    continue

                try:
                    path = nx.shortest_path(self.net_graph, src_host, dst_host, weight='weight')
                except nx.NetworkXNoPath:
                    print(f"No path between {src_host} and {dst_host}")
                    continue

                self.paths.append(f"Path from {src_host} to {dst_host}: {' -> '.join(path)}")

                dst_ip = self.topo.get_host_ip(dst_host).split('/')[0]

                for i in range(1, len(path) - 1):  # switches only
                    sw_name = path[i]
                    next_hop = path[i + 1]
                    if sw_name not in self.topo.get_p4switches().keys():
                        continue

                    egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
                    port_smac   = self.topo.node_to_node_mac(sw_name, next_hop)

                    if next_hop == dst_host:
                        next_hop_mac = self.topo.get_host_mac(dst_host)
                        dst_prefix   = f"{dst_ip}/32"
                        next_hop_ip  = dst_ip
                    else:
                        next_hop_mac = self.topo.node_to_node_mac(next_hop, sw_name)
                        dst_prefix   = f"{dst_ip}/24"
                        next_hop_ip  = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]

                    # init per-switch maps
                    if sw_name not in self.forwarding_entries:
                        self.forwarding_entries[sw_name] = {
                            'lpm': {},
                            'switching': {},
                            'mac': {}
                        }

                    # Record unique switching and mac keys once
                    self.forwarding_entries[sw_name]['switching'][next_hop_ip] = next_hop_mac
                    self.forwarding_entries[sw_name]['mac'][egress_port] = port_smac

                    # LPM entries per DSCP
                    for dscp in dscp_list:
                        self.forwarding_entries[sw_name]['lpm'][(dst_prefix, dscp)] = (next_hop_ip, egress_port)


    def update_path(self, sw_name, dst_prefix, dscp, next_hop_ip, egress_port):
        controller = self.controllers[sw_name]
        # Update LPM entry
        controller.table_modify_match(
            "l3_forward.ipv4_lpm",
            "ipv4_forward",
            [dst_prefix, dscp],
            [next_hop_ip, str(egress_port)]
        )
        # (Optionally) ensure aux tables have the needed keys only once:
        # controller.table_add(...) guarded by a read/exists check if your API supports it,
        # or keep a small in-memory set per switch to avoid re-adding.
        
        ### helper functions:
        #def table_modify(self, table_name, action_name, entry_handle, action_params=[]):
        #def table_modify_match(self, table_name, action_name, match_keys, action_params=[]):
        #def table_delete_match(self, table_name, match_keys):
        #def table_add(self, table_name, action_name, match_keys, action_params=[], prio=0):

    def program_switches(self):
        for sw_name, tables in self.forwarding_entries.items():
            controller = self.controllers[sw_name]

            # If your P4 has a real drop action, set it here; otherwise, remove this line.
            # controller.table_set_default("l3_forward.ipv4_lpm", "drop_pkt")

            # 1) Switching table: unique next_hop_ip
            for next_hop_ip, next_hop_mac in tables['switching'].items():
                controller.table_add(
                    "port_forward.switching_table",
                    "set_dmac",
                    [next_hop_ip],
                    [next_hop_mac]
                )

            # 2) MAC rewriting: unique egress_port
            for egress_port, port_smac in tables['mac'].items():
                egress_port_hex = f"0x{egress_port:x}"
                controller.table_add(
                    "port_forward.mac_rewriting_table",
                    "set_smac",
                    [egress_port_hex],
                    [port_smac]
                )

            # 3) LPM forwarding: unique (dst_prefix, dscp)
            for (dst_prefix, dscp), (next_hop_ip, egress_port) in tables['lpm'].items():
                controller.table_add(
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_ip, str(egress_port)]
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


'''
if __name__ == "__main__":
    controller = Controller()
    print("\n\nSUMMARY:")
    print("\nOSPF Shortest Paths:")
    controller.print_paths()  # Print the stored paths
    print("\nP4 Table Entries:")
    controller.print_forwarding_entries()  # Print forwarding entries
'''
