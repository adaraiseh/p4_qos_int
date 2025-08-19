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
        self.paths = []  # List to store paths between hosts

        # === NEW: id<->name maps and roles (parsed from rules) ===
        self.rules_dir = Path("rules/test")
        self.switch_id_to_name, self.switch_name_to_id, self.switch_id_role = self._parse_switch_ids_and_roles(self.rules_dir)

        # === Map host IPs to host names for graph lookups ===
        self.ip_to_host = {}
        for hname in self.topo.get_hosts().keys():
            hip = self.topo.get_host_ip(hname).split('/')[0]
            self.ip_to_host[hip] = hname

        # === NEW: minimal change history for revert ===
        self.change_history = []  # list of dicts describing modifications we can revert

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


    def update_path(self, sw_name, dst_prefix, dscp, next_hop_ip, egress_port):
        controller = self.controllers[sw_name]
        # >>> MODIFY-OR-ADD policy:
        # If we already have this (dst_prefix,dscp) key, modify; otherwise add a new match.
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

    # =========================
    # NEW: Helpers for action-1
    # =========================

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
        return self.find_alternate_for_worst(int(worst_switch_id)) is not None

    def has_pending_change(self) -> bool:
        return len(self.change_history) > 0

    def _dscp_for_qid(self, qid: int) -> str:
        """
        Adjust these mappings to your reality.
        You said: voice queue is 0, video is 1, best-effort is 7.
        Below DSCP hex values align with the pre-installed set used by compute_forwarding_entries().
        """
        mapping = {
            0: "0x00",  # voice -> default (adjust if needed)
            1: "0x18",  # video -> CS3
            7: "0x2E",  # best-effort -> EF (adjust if needed)
        }
        return mapping.get(int(qid), "0x00")

    def _dst_prefix_for_pair(self, dst_ip: str, next_hop_is_host: bool) -> str:
        if next_hop_is_host:
            return f"{dst_ip}/32"
        # >>> FIX: use network address for /24
        net = ip_network(f"{dst_ip}/24", strict=False).network_address
        return f"{net}/24"

    def _next_hop_params(self, sw_name: str, next_hop: str, dst_ip: str):
        """
        Returns (dst_prefix, next_hop_ip, egress_port)
        next_hop can be a switch or a host.
        """
        egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
        if next_hop in self.topo.get_hosts().keys():
            # toward host
            next_hop_ip = dst_ip
            dst_prefix = f"{dst_ip}/32"
        else:
            # toward switch
            next_hop_ip = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]
            # >>> FIX: /24 normalized to network
            net = ip_network(f"{dst_ip}/24", strict=False).network_address
            dst_prefix = f"{net}/24"
        return dst_prefix, next_hop_ip, egress_port

    def _host_name_from_ip(self, ip: str):
        """Return topology host name for an IP (e.g., 'h3'), or None if unknown."""
        return self.ip_to_host.get(ip)

    def _neighbor_iface_ip(self, neighbor: str, myself: str) -> str:
        """IP address of 'neighbor' interface that faces 'myself' (no mask)."""
        return self.topo.node_to_node_interface_ip(neighbor, myself).split('/')[0]

    def _find_prev_and_prefix(self, worst_name: str, dst_ip: str, dscp: str):
        """
        Find a neighbor switch that currently forwards (dst_prefix,dscp) *to* worst_name.
        Returns: (prev_node, dst_prefix) or (None, None) if not found.
        Strategy:
          - Check all switch neighbors N of worst.
          - For each N, test both candidate keys (dst_ip/32 and net(dst_ip)/24).
          - If N's LPM[(key,dscp)] -> next_hop_ip == ifaceIP(worst->N), that's our prev.
        """
        sw_neighbors = [n for n in self.net_graph.neighbors(worst_name)
                        if n in self.topo.get_p4switches().keys()]
        # Candidate keys we may encounter in the tables
        # (we prefer exact /32 first, then /24 network)
        from ipaddress import ip_network
        net24 = f"{ip_network(f'{dst_ip}/24', strict=False).network_address}/24"
        candidates = [f"{dst_ip}/32", net24]

        for prev in sw_neighbors:
            worst_ip_from_prev = self._neighbor_iface_ip(worst_name, prev)
            lpm = self.forwarding_entries.get(prev, {}).get('lpm', {})
            for key in candidates:
                val = lpm.get((key, dscp))
                if val is None:
                    continue
                nh_ip, _eport = val
                if nh_ip == worst_ip_from_prev:
                    return prev, key
        return None, None

    def _find_next_from_worst(self, worst_name: str, dst_ip: str, dscp: str, dst_prefix_hint: str | None):
        """
        Read worst's LPM to discover its current next hop for the demand.
        Returns: (next_node_name_or_host, dst_prefix_used, nh_ip, eport) or (None, None, None, None)
        - Tries the hint key first, then falls back to /32 and /24.
        - Maps nh_ip back to either a neighbor switch (by ifaceIP match) or to the dst host (nh_ip == dst_ip).
        """
        from ipaddress import ip_network
        lpm = self.forwarding_entries.get(worst_name, {}).get('lpm', {})
        prefer = []
        if dst_prefix_hint:
            prefer.append(dst_prefix_hint)
        prefer.append(f"{dst_ip}/32")
        prefer.append(f"{ip_network(f'{dst_ip}/24', strict=False).network_address}/24")

        for key in prefer:
            val = lpm.get((key, dscp))
            if val is None:
                continue
            nh_ip, eport = val
            if nh_ip == dst_ip:
                # next hop is the destination host
                return self._host_name_from_ip(dst_ip), key, nh_ip, eport
            # else try to find which neighbor switch owns this IP towards worst
            for neigh in self.net_graph.neighbors(worst_name):
                if neigh in self.topo.get_p4switches().keys():
                    neigh_ip_towards_worst = self._neighbor_iface_ip(neigh, worst_name)
                    if nh_ip == neigh_ip_towards_worst:
                        return neigh, key, nh_ip, eport
            # nh_ip didn't map to a known neighbor; still return raw info
            return None, key, nh_ip, eport

        return None, None, None, None

    def reroute_one_demand_symmetric(self, src_ip: str, dst_ip: str, qid: int,
                                     worst_switch_id: int, alt_switch_name: str):
        """
        Change route for one demand (src_ip,dst_ip) within the specific queue (DSCP),
        by splicing 'alt_switch_name' between the original upstream (prev) and downstream (next)
        around 'worst_switch_id'. Do it symmetrically (src->dst and dst->src).
        Implementation uses actual table state instead of recomputing a path.
        Returns (ok: bool, details: str)
        """
        dscp = self._dscp_for_qid(qid)
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return False, "worst switch id has no name"

        if alt_switch_name not in self.topo.get_p4switches().keys():
            return False, f"alt switch {alt_switch_name} not found in topology"

        # helper to rewire one direction given (src,dst)
        def _rewire_direction(src_ip_, dst_ip_):
            # 1) find upstream prev and the exact dst_prefix key it uses
            prev_node, dst_prefix = self._find_prev_and_prefix(worst_name, dst_ip_, dscp)
            if not prev_node or not dst_prefix:
                return None  # couldn't prove worst is on path for this direction

            # 2) find worst's current downstream next hop for this same key
            next_node, key_used, nh_ip_from_worst, eport_from_worst = self._find_next_from_worst(
                worst_name, dst_ip_, dscp, dst_prefix
            )
            if key_used is None:
                # The worst switch doesn't have an entry for this (dst,dscp) right now
                return None

            # 3) compute params to point prev -> ALT for (key_used,dscp)
            dst_prefix_used = key_used  # stick to the key we actually found
            nh_ip_prev_to_alt = self._neighbor_iface_ip(alt_switch_name, prev_node)
            eport_prev_to_alt = self.topo.node_to_node_port_num(prev_node, alt_switch_name)

            # 4) compute params to point ALT -> next for (key_used,dscp)
            if next_node is None:
                # Fallback: we know nh_ip_from_worst and eport_from_worst but couldn't map to a neighbor name.
                # In many P4 targets you still need the neighbor name to compute egress port; bail safely.
                return None

            if next_node in self.topo.get_p4switches().keys():
                nh_ip_alt_to_next = self._neighbor_iface_ip(next_node, alt_switch_name)
                eport_alt_to_next = self.topo.node_to_node_port_num(alt_switch_name, next_node)
            else:
                # next is the host
                nh_ip_alt_to_next = dst_ip_
                eport_alt_to_next = self.topo.node_to_node_port_num(alt_switch_name, next_node)

            # Save originals for revert
            orig_prev = self.forwarding_entries.get(prev_node, {}).get('lpm', {}).get((dst_prefix_used, dscp))
            orig_alt  = self.forwarding_entries.get(alt_switch_name, {}).get('lpm', {}).get((dst_prefix_used, dscp))
            orig_worst = self.forwarding_entries.get(worst_name, {}).get('lpm', {}).get((dst_prefix_used, dscp))

            # Apply changes
            self.update_path(prev_node, dst_prefix_used, dscp, nh_ip_prev_to_alt, eport_prev_to_alt)
            self.update_path(alt_switch_name, dst_prefix_used, dscp, nh_ip_alt_to_next, eport_alt_to_next)

            # Delete worst's entry if present
            if orig_worst is not None:
                try:
                    self.controllers[worst_name].table_delete_match("l3_forward.ipv4_lpm",
                                                                    [dst_prefix_used, dscp])
                except Exception:
                    pass
                try:
                    del self.forwarding_entries[worst_name]['lpm'][(dst_prefix_used, dscp)]
                except Exception:
                    pass

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
                "next_node": next_node,
            }

        # Forward (src -> dst) and reverse (dst -> src)
        fwd_change = _rewire_direction(src_ip, dst_ip)
        rev_change = _rewire_direction(dst_ip, src_ip)

        if fwd_change is None and rev_change is None:
            return False, "could not locate upstream/downstream around worst for either direction"

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

        _revert_one(change.get("fwd"))
        _revert_one(change.get("rev"))
        return True


'''
if __name__ == "__main__":
    controller = Controller()
    print("\n\nSUMMARY:")
    print("\nOSPF Shortest Paths:")
    controller.print_paths()  # Print the stored paths
    print("\nP4 Table Entries:")
    controller.print_forwarding_entries()  # Print forwarding entries
'''
