# controller.py

import re
import glob
from pathlib import Path
from ipaddress import ip_network

import networkx as nx
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI


class Controller:

    def __init__(self):
        self.topo = load_topo("topology.json")
        self.controllers = {}
        self.forwarding_entries = {}
        self.net_graph = nx.Graph()
        self.paths = []  # Human-readable list to store paths between hosts
        # Fast lookups for paths: (src_host, dst_host) -> [nodes...]
        self.path_map = {}

        # === Parse switch id/name/role from rules/test/*-commands.txt (used for role-aware alt choices) ===
        self.rules_dir = Path("rules/test")
        self.switch_id_to_name, self.switch_name_to_id, self.switch_id_role = self._parse_switch_ids_and_roles(self.rules_dir)
        self.tor_ids = {sid for sid, role in self.switch_id_role.items() if role == "tor"}

        # === Map host IPs to host names for graph/path lookups ===
        self.ip_to_host = {}
        for hname in self.topo.get_hosts().keys():
            hip = self.topo.get_host_ip(hname).split('/')[0]
            self.ip_to_host[hip] = hname

        # === Minimal change history for revert ===
        self.change_history = []  # list of dicts describing modifications we can revert

        self.connect_to_switches()
        self.build_network_graph()
        self.compute_forwarding_entries()  # also fills self.path_map
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
        self.paths.clear()
        self.path_map.clear()

        for src_host in hosts:
            for dst_host in hosts:
                if src_host == dst_host:
                    continue

                try:
                    path = nx.shortest_path(self.net_graph, src_host, dst_host, weight='weight')
                except nx.NetworkXNoPath:
                    print(f"No path between {src_host} and {dst_host}")
                    continue

                # Track for debugging and fast lookup
                self.paths.append(f"Path from {src_host} to {dst_host}: {' -> '.join(path)}")
                self.path_map[(src_host, dst_host)] = list(path)

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
                        # >>> FIX: use network address for /24
                        net = ip_network(f"{dst_ip}/24", strict=False).network_address
                        dst_prefix   = f"{net}/24"
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

    # -----------------------
    # Table/aux helpers
    # -----------------------

    def update_path(self, sw_name, dst_prefix, dscp, next_hop_ip, egress_port):
        controller = self.controllers[sw_name]
        # MODIFY-OR-ADD policy:
        exists = (sw_name in self.forwarding_entries
                  and (dst_prefix, dscp) in self.forwarding_entries[sw_name].get('lpm', {}))
        try:
            if exists:
                controller.table_modify_match(
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_ip, str(egress_port)]
                )
            else:
                controller.table_add(
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_ip, str(egress_port)]
                )
        finally:
            # Keep shadow copy up to date for reversions
            if sw_name in self.forwarding_entries:
                self.forwarding_entries[sw_name].setdefault('lpm', {})
                self.forwarding_entries[sw_name]['lpm'][(dst_prefix, dscp)] = (next_hop_ip, egress_port)

        # (Optionally) ensure aux tables have the needed keys only once:
        # controller.table_add(...) guarded by a read/exists check if your API supports it,
        # or keep a small in-memory set per switch to avoid re-adding.
        
        ### helper functions:
        #def table_modify(self, table_name, action_name, entry_handle, action_params=[]):
        #def table_modify_match(self, table_name, action_name, match_keys, action_params=[]):
        #def table_delete_match(self, table_name, match_keys):
        #def table_add(self, table_name, action_name, match_keys, action_params=[], prio=0):

    def ensure_switching_and_mac(self, sw_name: str, next_hop: str):
        """
        Lazily ensure switching_table and mac_rewriting_table entries exist for (sw_name -> next_hop).
        Safe to call repeatedly; it updates our shadow maps so we don't re-add unnecessarily.
        """
        controller = self.controllers[sw_name]

        egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
        port_smac   = self.topo.node_to_node_mac(sw_name, next_hop)

        if sw_name not in self.forwarding_entries:
            self.forwarding_entries[sw_name] = {'lpm': {}, 'switching': {}, 'mac': {}}

        # switching_table key: next_hop_ip, value: next_hop_mac
        if next_hop in self.topo.get_hosts().keys():
            next_hop_ip  = self.topo.get_host_ip(next_hop).split('/')[0]
            next_hop_mac = self.topo.get_host_mac(next_hop)
        else:
            next_hop_ip  = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]
            next_hop_mac = self.topo.node_to_node_mac(next_hop, sw_name)

        if next_hop_ip not in self.forwarding_entries[sw_name]['switching']:
            controller.table_add(
                "port_forward.switching_table",
                "set_dmac",
                [next_hop_ip],
                [next_hop_mac]
            )
            self.forwarding_entries[sw_name]['switching'][next_hop_ip] = next_hop_mac

        # mac_rewriting_table key: egress_port
        if egress_port not in self.forwarding_entries[sw_name]['mac']:
            egress_port_hex = f"0x{egress_port:x}"
            controller.table_add(
                "port_forward.mac_rewriting_table",
                "set_smac",
                [egress_port_hex],
                [port_smac]
            )
            self.forwarding_entries[sw_name]['mac'][egress_port] = port_smac

    # -----------------------
    # Role/alt helpers
    # -----------------------

    def _parse_switch_ids_and_roles(self, rules_dir: Path):
        """
        Parse rules_dir/*-commands.txt to derive:
         - switch_id -> switch_name (by filename stem e.g., a1-commands.txt => 'a1')
         - switch_name -> switch_id
         - switch_id -> role ('tor'|'agg'|'core'|'other') by name prefix
        Assumes each file contains:
          table_set_default process_int_transit.tb_int_insert init_metadata <ID>
        """
        sid_to_name = {}
        name_to_sid = {}
        sid_role = {}
        if not rules_dir.exists():
            return sid_to_name, name_to_sid, sid_role

        pat = re.compile(
            r"table_set_default\s+process_int_transit\.tb_int_insert\s+init_metadata\s+(\d+)",
            re.IGNORECASE
        )
        for path in glob.glob(str(rules_dir / "*-commands.txt")):
            fname = Path(path).name
            sw_name = fname.split("-")[0]  # t1, a2, c3...
            role = "other"
            if sw_name.startswith("t"):
                role = "tor"
            elif sw_name.startswith("a"):
                role = "agg"
            elif sw_name.startswith("c"):
                role = "core"
            try:
                with open(path, "r") as f:
                    text = f.read()
                m = pat.search(text)
                if m:
                    sid = int(m.group(1))
                    sid_to_name[sid] = sw_name
                    name_to_sid[sw_name] = sid
                    sid_role[sid] = role
            except Exception as e:
                print(f"Failed to parse {path}: {e}")

        return sid_to_name, name_to_sid, sid_role

    def _role_of_sid(self, sid: int) -> str:
        return self.switch_id_role.get(int(sid), "other")

    def _neighbors_in_graph(self, node_name: str):
        return list(self.net_graph.neighbors(node_name))

    def _tors_connected_to_agg(self, agg_name: str):
        # ToRs are nodes starting with 't'
        return [n for n in self._neighbors_in_graph(agg_name) if n.startswith("t")]

    def _is_same_pod_agg(self, candidate_agg: str, current_agg: str) -> bool:
        # Share at least one ToR => same pod (simple heuristic)
        cur_tors = set(self._tors_connected_to_agg(current_agg))
        cand_tors = set(self._tors_connected_to_agg(candidate_agg))
        return len(cur_tors & cand_tors) > 0

    def find_alternate_for_worst(self, worst_switch_id: int):
        """
        Pick an alternate switch name for the given worst switch id:
          - ToR: return None
          - Agg: another 'a*' in same pod (shares at least one ToR)
          - Core: any other 'c*'
        """
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return None

        role = self._role_of_sid(worst_switch_id)
        if role == "tor":
            return None

        # all switch nodes present in topology graph
        sw_nodes = [n for n in self.topo.get_p4switches().keys()]

        if role == "agg":
            candidates = [n for n in sw_nodes if n.startswith("a") and n != worst_name]
            same_pod = [n for n in candidates if self._is_same_pod_agg(n, worst_name)]
            same_pod.sort()
            return same_pod[0] if same_pod else None

        if role == "core":
            candidates = [n for n in sw_nodes if n.startswith("c") and n != worst_name]
            candidates.sort()
            return candidates[0] if candidates else None

        # other roles
        return None

    def has_alternate_for_worst(self, worst_switch_id: int) -> bool:
        """
        Compatibility wrapper used by rl_agent.py.
        Returns True if we can propose an alternate switch for the given worst switch id,
        subject to role constraints (no ToR, same pod for aggs, any core).
        """
        return self.find_alternate_for_worst(int(worst_switch_id)) is not None

    def has_pending_change(self) -> bool:
        """
        Compatibility wrapper used by rl_agent.py.
        Returns True if there is at least one recorded path/routing change
        that can be reverted via revert_last_change().
        """
        return hasattr(self, "change_history") and bool(self.change_history)

    # -----------------------
    # Misc helpers
    # -----------------------

    def _host_name_from_ip(self, ip: str):
        """Return topology host name for an IP (e.g., 'h3'), or None if unknown."""
        return self.ip_to_host.get(ip)

    def _dscp_for_qid(self, qid: int) -> str:
        """
        Adjust these mappings to your reality.
        You said: voice queue is 0, video is 1, best-effort is 7.
        Below DSCP hex values align with the pre-installed set used by compute_forwarding_entries().
        """
        mapping = {
            0: "0x00",  # voice -> default (adjust if needed)
            1: "0x18",  # video -> CS3
            7: "0x2E",  # best-effort -> EF
        }
        return mapping.get(int(qid), "0x00")

    def _neighbor_iface_ip(self, neighbor: str, myself: str) -> str:
        """IP address of 'neighbor' interface that faces 'myself' (no mask)."""
        return self.topo.node_to_node_interface_ip(neighbor, myself).split('/')[0]

    # -----------------------
    # Reroute using stored paths (path-tracking approach)
    # -----------------------

    def reroute_one_demand_symmetric(self, src_ip: str, dst_ip: str, qid: int,
                                     worst_switch_id: int, alt_switch_name: str):
        """
        Change route for one demand (src_ip,dst_ip) within the specific queue (DSCP),
        by splicing 'alt_switch_name' between the original upstream (prev) and downstream (next)
        around 'worst_switch_id', using the STORED PATHS for the demand.
        Do it symmetrically (src->dst and dst->src).
        Returns (ok: bool, details: str)
        """
        dscp = self._dscp_for_qid(qid)
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return False, "worst switch id has no name"

        if alt_switch_name not in self.topo.get_p4switches().keys():
            return False, f"alt switch {alt_switch_name} not found in topology"

        src_host = self._host_name_from_ip(src_ip)
        dst_host = self._host_name_from_ip(dst_ip)
        if not src_host or not dst_host:
            return False, f"host name not found for src={src_ip} dst={dst_ip}"

        # helper to rewire one direction given hosts
        def _rewire_direction(src_h, dst_h, dst_ip_):
            path = self.path_map.get((src_h, dst_h))
            if not path or worst_name not in path:
                return None  # couldn't prove worst is on this stored path

            idx = path.index(worst_name)
            if idx == 0 or idx == len(path)-1:
                return None  # worst at endpoint, ignore

            prev_node = path[idx-1]
            next_node = path[idx+1]
            if prev_node not in self.topo.get_p4switches().keys():
                return None  # previous hop isn't a switch (shouldn't happen)

            # Determine the exact dst_prefix style we should use for this hop:
            # - If next is host -> /32
            # - Else -> /24 network normalized
            if next_node == dst_h:
                dst_prefix_used = f"{dst_ip_}/32"
            else:
                net = ip_network(f"{dst_ip_}/24", strict=False).network_address
                dst_prefix_used = f"{net}/24"

            # Prepare prev -> ALT
            self.ensure_switching_and_mac(prev_node, alt_switch_name)
            nh_ip_prev_to_alt = self._neighbor_iface_ip(alt_switch_name, prev_node)
            eport_prev_to_alt = self.topo.node_to_node_port_num(prev_node, alt_switch_name)

            # Prepare ALT -> next
            self.ensure_switching_and_mac(alt_switch_name, next_node)
            if next_node in self.topo.get_p4switches().keys():
                nh_ip_alt_to_next = self._neighbor_iface_ip(next_node, alt_switch_name)
            else:
                nh_ip_alt_to_next = dst_ip_
            eport_alt_to_next = self.topo.node_to_node_port_num(alt_switch_name, next_node)

            # Save originals for revert
            orig_prev  = self.forwarding_entries.get(prev_node,        {}).get('lpm', {}).get((dst_prefix_used, dscp))
            orig_alt   = self.forwarding_entries.get(alt_switch_name,  {}).get('lpm', {}).get((dst_prefix_used, dscp))
            orig_worst = self.forwarding_entries.get(worst_name,       {}).get('lpm', {}).get((dst_prefix_used, dscp))

            # Apply changes
            self.update_path(prev_node, dst_prefix_used, dscp, nh_ip_prev_to_alt, eport_prev_to_alt)
            self.update_path(alt_switch_name, dst_prefix_used, dscp, nh_ip_alt_to_next, eport_alt_to_next)

            # Delete worst's entry if present
            if orig_worst is not None:
                try:
                    self.controllers[worst_name].table_delete_match("l3_forward.ipv4_lpm", [dst_prefix_used, dscp])
                except Exception:
                    pass
                try:
                    del self.forwarding_entries[worst_name]['lpm'][(dst_prefix_used, dscp)]
                except Exception:
                    pass

            # Update the stored path: replace worst_name with alt_switch_name
            new_path = list(path)
            new_path[idx] = alt_switch_name
            self.path_map[(src_h, dst_h)] = new_path

            return {
                "prev_node": prev_node,
                "dst_prefix": dst_prefix_used,
                "dscp": dscp,
                "prev_before": orig_prev,
                "prev_after": (nh_ip_prev_to_alt, eport_prev_to_alt),
                "alt_node": alt_switch_name,
                "alt_before": orig_alt,
                "alt_after": (nh_ip_alt_to_next, eport_alt_to_next),
                "worst_node": worst_name,
                "worst_deleted": (dst_prefix_used, dscp, orig_worst),
                "old_path": path,
                "new_path": new_path,
            }

        # Forward (src -> dst) and reverse (dst -> src)
        fwd_change = _rewire_direction(src_host, dst_host, dst_ip)
        rev_change = _rewire_direction(dst_host, src_host, src_ip)

        if fwd_change is None and rev_change is None:
            return False, "worst not present on stored path for either direction"

        # Record for revert()
        self.change_history.append({
            "fwd": fwd_change,
            "rev": rev_change
        })
        return True, "ok"

    def revert_last_change(self):
        if not self.change_history:
            return False
        change = self.change_history.pop()

        def _revert_one(side):
            if not side:
                return
            prev_node = side["prev_node"]
            dst_prefix = side["dst_prefix"]
            dscp = side["dscp"]
            alt_node = side["alt_node"]
            worst_node = side["worst_node"]

            # Revert prev node
            if side["prev_before"] is not None:
                nh_ip, eport = side["prev_before"]
                self.update_path(prev_node, dst_prefix, dscp, nh_ip, eport)

            # Revert alt node
            if side["alt_before"] is not None:
                nh_ip, eport = side["alt_before"]
                self.update_path(alt_node, dst_prefix, dscp, nh_ip, eport)
            else:
                # If there was no prior entry, attempt to remove the specific match
                try:
                    self.controllers[alt_node].table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
                except Exception:
                    pass
                try:
                    del self.forwarding_entries[alt_node]['lpm'][(dst_prefix, dscp)]
                except Exception:
                    pass

            # Restore worst node entry if we had deleted it
            _, _, worst_before = side["worst_deleted"]
            if worst_before is not None:
                nh_ip, eport = worst_before
                self.update_path(worst_node, dst_prefix, dscp, nh_ip, eport)

            # Restore path snapshot, if present
            old_path = side.get("old_path")
            new_path = side.get("new_path")
            if old_path and new_path:
                # Determine src/dst hosts from either path ends
                src_h = old_path[0]
                dst_h = old_path[-1]
                # Only restore if our current record matches the new_path we had written
                cur = self.path_map.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.path_map[(src_h, dst_h)] = old_path

        _revert_one(change.get("fwd"))
        _revert_one(change.get("rev"))
        return True

    # -----------------------
    # Debug
    # -----------------------

    def print_paths(self):
        """
        Print the stored paths between hosts.
        """
        for (src, dst), path in sorted(self.path_map.items()):
            print(f"Path from {src} to {dst}: {' -> '.join(path)}")

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
