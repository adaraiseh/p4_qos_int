# controller.py

import re
import json
import glob
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from ipaddress import ip_network
from typing import Optional, Dict, List, Set

import networkx as nx
import warnings
# Suppress NetworkX 3.6 future warning coming from p4utils
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI


class Controller:

    # Role normalization: edge switches (can't be rerouted around)
    EDGE_ROLES = {'leaf', 'access', 'tor'}
    # Role normalization: aggregation/spine switches
    AGG_ROLES = {'spine', 'distribution', 'agg'}
    # Role normalization: core switches
    CORE_ROLES = {'core'}

    def __init__(self, verbose: bool = False, topology_builder=None, rules_dir: str = None):
        """
        Initialize the controller.

        Args:
            verbose: Enable verbose logging
            topology_builder: Optional TopologyBuilder instance for role metadata.
                            If not provided, roles are parsed from rules files.
            rules_dir: Directory containing P4 rules files (default: rules/test)
        """
        self.verbose = verbose
        self.topo = load_topo("topology.json")
        self.controllers = {}
        self.forwarding_entries = {}
        self.net_graph = nx.Graph()
        self.paths = []
        self.path_map = {}
        self.alt_rr_pos = {}

        # Store topology builder reference for role lookups
        self._topology_builder = topology_builder

        # Determine rules directory
        if rules_dir is None:
            if topology_builder is not None:
                topo_name = topology_builder.config.topology.name.replace('-', '_')
                rules_dir = str(Path(__file__).parent / 'rules' / topo_name)
            else:
                rules_dir = "rules/test"
        self.rules_dir = Path(rules_dir)

        # Parse switch mappings
        if topology_builder is not None:
            self.switch_id_to_name, self.switch_name_to_id, self.switch_id_role = \
                self._build_mappings_from_builder(topology_builder)
        else:
            self.switch_id_to_name, self.switch_name_to_id, self.switch_id_role = \
                self._parse_switch_ids_and_roles(self.rules_dir)

        # Edge switch IDs (leaf/tor/access - can't be bypassed)
        self.edge_switch_ids = {
            sid for sid, role in self.switch_id_role.items()
            if self._normalize_role(role) == 'edge'
        }
        # Legacy alias for backward compatibility
        self.tor_ids = self.edge_switch_ids

        # Map host IPs to host names
        self.ip_to_host = {}
        for hname in self.topo.get_hosts().keys():
            hip = self.topo.get_host_ip(hname).split('/')[0]
            self.ip_to_host[hip] = hname

        # Per-queue change history stacks
        self.change_history_by_qid = {}

        # Usage tracking for alternatives
        self.switch_usage = {}
        self.queue_changes = {0: 0, 1: 0, 7: 0}
        self.queue_last_change_step = {0: 0, 1: 0, 7: 0}

        # Per-queue path tracking
        self.paths_per_queue = {0: {}, 1: {}, 7: {}}

        self.connect_to_switches()
        self.build_network_graph()
        self.compute_forwarding_entries()
        self.program_switches()
        self.dump_paths_json()

    def _normalize_role(self, role: str) -> str:
        """
        Normalize role to one of: 'edge', 'agg', 'core', 'other'.

        Edge switches connect to hosts and cannot be bypassed.
        """
        role_lower = role.lower() if role else 'other'
        if role_lower in self.EDGE_ROLES:
            return 'edge'
        elif role_lower in self.AGG_ROLES:
            return 'agg'
        elif role_lower in self.CORE_ROLES:
            return 'core'
        return 'other'

    def _build_mappings_from_builder(self, builder):
        """
        Build switch mappings from TopologyBuilder instance.

        Returns:
            Tuple of (sid_to_name, name_to_sid, sid_role)
        """
        sid_to_name = {}
        name_to_sid = {}
        sid_role = {}

        for switch_name, sw_info in builder.switches.items():
            sid = sw_info.switch_id
            sid_to_name[sid] = switch_name
            name_to_sid[switch_name] = sid
            sid_role[sid] = sw_info.role

        return sid_to_name, name_to_sid, sid_role

    # -----------------------
    # Quiet helpers
    # -----------------------
    def _silent_call(self, fn, *args, **kwargs):
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            return fn(*args, **kwargs)

    def _call(self, fn, *args, **kwargs):
        if self.verbose:
            return fn(*args, **kwargs)
        return self._silent_call(fn, *args, **kwargs)

    def connect_to_switches(self):
        for sw_name in self.topo.get_p4switches().keys():
            thrift_port = self.topo.get_thrift_port(sw_name)
            self.controllers[sw_name] = SimpleSwitchThriftAPI(thrift_port)

    def build_network_graph(self):
        p4switches = list(self.topo.get_p4switches().keys())
        hosts = list(self.topo.get_hosts().keys())

        # Exclude collector hosts from the graph (they're for INT reports only, not routing)
        if self._topology_builder is not None:
            collector_names = set(self._topology_builder.collectors.keys())
            hosts = [h for h in hosts if h not in collector_names]
        else:
            # Fallback: exclude hosts with id >= 100 (h100, h101, etc.)
            hosts = [h for h in hosts if not (h.startswith('h') and h[1:].isdigit() and int(h[1:]) >= 100)]

        nodes = p4switches + hosts
        self.net_graph.add_nodes_from(nodes)

        for node in nodes:
            neighbors = self.topo.get_neighbors(node)
            for neighbor in neighbors:
                # Skip adding edges to collectors (not in our filtered nodes)
                if neighbor not in nodes:
                    continue
                if not self.net_graph.has_edge(node, neighbor):
                    self.net_graph.add_edge(node, neighbor, weight=1)

    def compute_forwarding_entries(self):
        hosts = list(self.topo.get_hosts().keys())

        # Exclude collector hosts (they're for INT reports only, not traffic)
        if self._topology_builder is not None:
            collector_names = set(self._topology_builder.collectors.keys())
            hosts = [h for h in hosts if h not in collector_names]
        else:
            # Fallback: exclude hosts with id >= 100 (h100, h101, etc.)
            hosts = [h for h in hosts if not (h.startswith('h') and h[1:].isdigit() and int(h[1:]) >= 100)]

        dscp_list = ["0x2E", "0x18", "0x00"]

        self.forwarding_entries = {}
        self.paths.clear()
        self.path_map.clear()
        for qid in self.paths_per_queue:
            self.paths_per_queue[qid].clear()

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
                self.path_map[(src_host, dst_host)] = list(path)
                for qid in self.paths_per_queue:
                    self.paths_per_queue[qid][(src_host, dst_host)] = list(path)

                dst_ip = self.topo.get_host_ip(dst_host).split('/')[0]

                for i in range(1, len(path) - 1):
                    sw_name = path[i]
                    next_hop = path[i + 1]
                    if sw_name not in self.topo.get_p4switches().keys():
                        continue

                    egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
                    port_smac = self.topo.node_to_node_mac(sw_name, next_hop)

                    if next_hop == dst_host:
                        next_hop_mac = self.topo.get_host_mac(dst_host)
                        dst_prefix = f"{dst_ip}/32"
                        next_hop_ip = dst_ip
                    else:
                        next_hop_mac = self.topo.node_to_node_mac(next_hop, sw_name)
                        net = ip_network(f"{dst_ip}/24", strict=False).network_address
                        dst_prefix = f"{net}/24"
                        next_hop_ip = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]

                    if sw_name not in self.forwarding_entries:
                        self.forwarding_entries[sw_name] = {
                            'lpm': {},
                            'switching': {},
                            'mac': {}
                        }

                    self.forwarding_entries[sw_name]['switching'][next_hop_ip] = next_hop_mac
                    self.forwarding_entries[sw_name]['mac'][egress_port] = port_smac

                    for dscp in dscp_list:
                        self.forwarding_entries[sw_name]['lpm'][(dst_prefix, dscp)] = (next_hop_ip, egress_port)

    def program_switches(self):
        for sw_name, tables in self.forwarding_entries.items():
            controller = self.controllers[sw_name]
            for next_hop_ip, next_hop_mac in tables['switching'].items():
                self._call(
                    controller.table_add,
                    "port_forward.switching_table",
                    "set_dmac",
                    [next_hop_ip],
                    [next_hop_mac]
                )
            for egress_port, port_smac in tables['mac'].items():
                egress_port_hex = f"0x{egress_port:x}"
                self._call(
                    controller.table_add,
                    "port_forward.mac_rewriting_table",
                    "set_smac",
                    [egress_port_hex],
                    [port_smac]
                )
            for (dst_prefix, dscp), (next_hop_ip, egress_port) in tables['lpm'].items():
                self._call(
                    controller.table_add,
                    "l3_forward.ipv4_lpm",
                    "ipv4_forward",
                    [dst_prefix, dscp],
                    [next_hop_ip, str(egress_port)]
                )

    def clear_all_tables(self):
        """Clear all P4 tables on all switches and reset internal state."""
        for sw_name, controller in self.controllers.items():
            try:
                self._call(controller.table_clear, "l3_forward.ipv4_lpm")
                self._call(controller.table_clear, "port_forward.switching_table")
                self._call(controller.table_clear, "port_forward.mac_rewriting_table")
            except Exception as e:
                print(f"[WARN] Failed to clear tables on {sw_name}: {e}")

        self.forwarding_entries.clear()
        self.change_history_by_qid.clear()
        self.switch_usage.clear()
        self.queue_changes = {0: 0, 1: 0, 7: 0}
        self.queue_last_change_step = {0: 0, 1: 0, 7: 0}

        for qid in self.paths_per_queue:
            self.paths_per_queue[qid].clear()

    # -----------------------
    # Table/aux helpers
    # -----------------------

    def _upsert_lpm(self, sw_name: str, dst_prefix: str, dscp: str, next_hop_ip: str, egress_port: int) -> bool:
        controller = self.controllers[sw_name]
        try:
            try:
                self._call(controller.table_delete_match, "l3_forward.ipv4_lpm", [dst_prefix, dscp])
            except Exception:
                pass
            self._call(
                controller.table_add,
                "l3_forward.ipv4_lpm",
                "ipv4_forward",
                [dst_prefix, dscp],
                [next_hop_ip, str(egress_port)]
            )
            self.forwarding_entries.setdefault(sw_name, {}).setdefault('lpm', {})
            self.forwarding_entries[sw_name]['lpm'][(dst_prefix, dscp)] = (next_hop_ip, egress_port)
            return True
        except Exception as e:
            print(f"[LPM upsert FAILED] {sw_name} {dst_prefix} dscp={dscp} -> {next_hop_ip}/{egress_port}: {e}")
            return False

    def update_path(self, sw_name, dst_prefix, dscp, next_hop_ip, egress_port):
        return self._upsert_lpm(sw_name, dst_prefix, dscp, next_hop_ip, egress_port)

    def ensure_switching_and_mac(self, sw_name: str, next_hop: str):
        controller = self.controllers[sw_name]
        egress_port = self.topo.node_to_node_port_num(sw_name, next_hop)
        port_smac = self.topo.node_to_node_mac(sw_name, next_hop)

        if sw_name not in self.forwarding_entries:
            self.forwarding_entries[sw_name] = {'lpm': {}, 'switching': {}, 'mac': {}}

        if next_hop in self.topo.get_hosts().keys():
            next_hop_ip = self.topo.get_host_ip(next_hop).split('/')[0]
            next_hop_mac = self.topo.get_host_mac(next_hop)
        else:
            next_hop_ip = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]
            next_hop_mac = self.topo.node_to_node_mac(next_hop, sw_name)

        if next_hop_ip not in self.forwarding_entries[sw_name]['switching']:
            self._call(
                controller.table_add,
                "port_forward.switching_table",
                "set_dmac",
                [next_hop_ip],
                [next_hop_mac]
            )
            self.forwarding_entries[sw_name]['switching'][next_hop_ip] = next_hop_mac

        if egress_port not in self.forwarding_entries[sw_name]['mac']:
            egress_port_hex = f"0x{egress_port:x}"
            self._call(
                controller.table_add,
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
        Parse switch IDs and roles from rules files.

        This is the fallback method when no TopologyBuilder is provided.
        Role detection uses name prefixes as a heuristic.
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
        files = sorted(glob.glob(str(rules_dir / "*-commands.txt")))

        for path in files:
            fname = Path(path).name
            sw_name = fname.split("-")[0]

            # Heuristic role detection from name prefix
            # This is less reliable than using TopologyBuilder
            role = self._infer_role_from_name(sw_name)

            try:
                with open(path, "r") as f:
                    text = f.read()
                m = pat.search(text)
                if not m:
                    print(f"[WARN] No init_metadata ID found in {fname}; skipping mapping for {sw_name}")
                    continue

                sid = int(m.group(1))

                if sid in sid_to_name and sid_to_name[sid] != sw_name:
                    print(f"[WARN] Duplicate switch_id {sid}: already mapped to {sid_to_name[sid]}, "
                          f"ignoring later mapping from {sw_name} ({fname})")
                    name_to_sid[sw_name] = sid
                    continue

                sid_to_name.setdefault(sid, sw_name)
                name_to_sid[sw_name] = sid
                sid_role[sid] = role

            except Exception as e:
                print(f"Failed to parse {path}: {e}")

        return sid_to_name, name_to_sid, sid_role

    def _infer_role_from_name(self, sw_name: str) -> str:
        """
        Infer switch role from name using common naming conventions.

        This is a fallback heuristic when no topology metadata is available.
        """
        name_lower = sw_name.lower()

        # Edge switches (leaf/tor/access)
        if any(name_lower.startswith(p) for p in ('t', 'tor', 'leaf', 'l', 'access', 'acc', 'a')):
            # Check if it looks like aggregation (starts with 'a' but has 'gg' or numbers suggest agg range)
            if name_lower.startswith('a') and not name_lower.startswith('acc'):
                # Could be 'a1' (agg) or 'access1'
                # Check ID range: typically agg switches have higher IDs
                return 'agg'
            if name_lower.startswith('t') or name_lower.startswith('tor') or name_lower.startswith('leaf') or name_lower.startswith('l'):
                return 'leaf'
            return 'access'

        # Spine/aggregation switches
        if any(name_lower.startswith(p) for p in ('s', 'spine', 'agg', 'dist')):
            if name_lower.startswith('s') and not name_lower.startswith('spine'):
                # Could be 's1' which might be spine or just switch
                return 'spine'
            return 'spine'

        # Core switches
        if any(name_lower.startswith(p) for p in ('c', 'core')):
            return 'core'

        return 'other'

    def _role_of_sid(self, sid: int) -> str:
        """Get the role of a switch by its ID."""
        return self.switch_id_role.get(int(sid), "other")

    def _is_edge_switch(self, sid: int) -> bool:
        """Check if a switch is an edge switch (cannot be bypassed)."""
        role = self._role_of_sid(sid)
        return self._normalize_role(role) == 'edge'

    def _neighbors_in_graph(self, node_name: str):
        return list(self.net_graph.neighbors(node_name))

    def _edge_switches_connected_to(self, switch_name: str) -> List[str]:
        """
        Get all edge switches (leaf/tor/access) connected to a switch.

        This replaces the hardcoded _tors_connected_to_agg method.
        """
        neighbors = self._neighbors_in_graph(switch_name)
        edge_switches = []

        for neighbor in neighbors:
            if neighbor not in self.switch_name_to_id:
                continue
            sid = self.switch_name_to_id[neighbor]
            if self._is_edge_switch(sid):
                edge_switches.append(neighbor)

        return edge_switches

    # Legacy method for backward compatibility
    def _tors_connected_to_agg(self, agg_name: str):
        return self._edge_switches_connected_to(agg_name)

    def _is_same_pod(self, candidate_switch: str, current_switch: str) -> bool:
        """
        Check if two switches are in the same pod.

        Pod membership is determined by shared edge switch connections.
        """
        cur_edges = set(self._edge_switches_connected_to(current_switch))
        cand_edges = set(self._edge_switches_connected_to(candidate_switch))
        return len(cur_edges & cand_edges) > 0

    # Legacy alias
    def _is_same_pod_agg(self, candidate_agg: str, current_agg: str) -> bool:
        return self._is_same_pod(candidate_agg, current_agg)

    def find_alternate_for_worst(self, worst_switch_id: int, path: list[str]):
        """Legacy method for backward compatibility."""
        alts = self.find_all_alternates(worst_switch_id, path)
        if not alts:
            return None
        last_idx = self.alt_rr_pos.get(int(worst_switch_id), -1)
        next_idx = (last_idx + 1) % len(alts)
        self.alt_rr_pos[int(worst_switch_id)] = next_idx
        return alts[next_idx]

    def find_all_alternates(self, worst_switch_id: int, path: list[str]) -> list[str]:
        """
        Return all valid alternative switches for the worst node in the path.

        Generic Logic:
        Finds any switch (other than 'edge' or 'bottleneck') that allows a valid path
        from source to destination in a graph view EXCLUDING the bottleneck.
        This supports both local stitching (c1 -> c2) and global rerouting (c1 -> a2 -> c3 -> a4).
        """
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return []

        # Edge switches cannot be bypassed
        if self._is_edge_switch(worst_switch_id):
            return []

        if not path or worst_name not in path or len(path) < 2:
            return []

        src_node = path[0]
        dst_node = path[-1]

        # Candidates are all switches EXCEPT:
        # 1. The bottleneck (worst) switch itself
        # 2. Edge switches (generally we don't route through other edge switches as transit)
        candidates = []
        all_sids = self.get_all_switch_ids()
        
        # Create a graph view WITHOUT the bottleneck to verify independent reachability
        # We must copy because we'll check connectivity
        G_view = self.net_graph.copy()
        if worst_name in G_view:
            G_view.remove_node(worst_name)
        else:
            # If bottleneck not in graph, something is wrong, but proceed safely
            return []

        # Optimization: Pre-check connectivity from src/dst in the restricted graph
        if not (G_view.has_node(src_node) and G_view.has_node(dst_node)):
            return []

        # Verify if src can reach dst at all without the bottleneck
        if not nx.has_path(G_view, src_node, dst_node):
            return []

        worst_role = self._normalize_role(self._role_of_sid(worst_switch_id))

        for sid in all_sids:
            if sid == int(worst_switch_id):
                continue
            
            if self._is_edge_switch(sid):
                continue
            
            # Enforce Matching Role: Candidate must have same role as bottleneck
            cand_role = self._normalize_role(self._role_of_sid(sid))
            if cand_role != worst_role:
                continue

            cand_name = self.switch_id_to_name.get(sid)
            if not cand_name or cand_name not in G_view:
                continue

            # Check 1: Reachability from Source -> Candidate (without bottleneck)
            if nx.has_path(G_view, src_node, cand_name):
                # Check 2: Reachability from Candidate -> Destination (without bottleneck)
                if nx.has_path(G_view, cand_name, dst_node):
                    candidates.append(cand_name)

        return sorted(candidates)

    def has_alternate_for_worst(self, worst_switch_id: int, path: list[str]) -> bool:
        return bool(self.find_all_alternates(int(worst_switch_id), path))

    def find_worst_with_alternative(self, switch_metrics: list, qid: int):
        """
        Find the worst bottleneck that has at least one alternative path.

        Args:
            switch_metrics: List of switch metric objects with switch_id and congestion_score
            qid: Queue ID to check for flows

        Returns:
            Switch metric object if found, None otherwise
        """
        # Sort by congestion (worst first)
        sorted_switches = sorted(switch_metrics, key=lambda s: getattr(s, 'congestion_score', 0), reverse=True)

        for switch in sorted_switches:
            sid = getattr(switch, 'switch_id', None)
            if sid is None:
                continue

            # Edge switches can't be bypassed
            if self._is_edge_switch(sid):
                continue

            # Get a representative path for this switch
            path = self._get_representative_path(sid, qid)
            if path is None:
                continue

            # Check if alternatives exist
            alts = self.find_all_alternates(sid, path)
            if alts:
                return switch

        return None

    def _get_representative_path(self, switch_id: int, qid: int):
        """Get a path that includes this switch for the given queue."""
        switch_name = self.switch_id_to_name.get(int(switch_id))
        if not switch_name:
            return None

        # Look through paths for this queue
        queue_paths = self.paths_per_queue.get(int(qid), {})
        for (src, dst), path in queue_paths.items():
            if switch_name in path:
                return path

        # Fall back to global path map
        for (src, dst), path in self.path_map.items():
            if switch_name in path:
                return path

        return None

    def get_all_switch_ids(self) -> list[int]:
        """Return sorted list of all known switch IDs."""
        return sorted(list(self.switch_id_to_name.keys()))

    def track_usage(self, alt_name: str):
        """Track usage count for an alternative switch."""
        self.switch_usage[alt_name] = self.switch_usage.get(alt_name, 0) + 1

    def get_usage_count(self, alt_name: str) -> int:
        """Get usage count for a switch."""
        return self.switch_usage.get(alt_name, 0)

    def record_queue_change(self, qid: int, global_step: int):
        """Track that a queue was changed at this global step."""
        qid = int(qid)
        self.queue_changes[qid] = self.queue_changes.get(qid, 0) + 1
        self.queue_last_change_step[qid] = global_step

    def get_queue_history(self, qid: int, current_step: int) -> tuple:
        """Returns (total_changes, steps_since_change) for a queue."""
        qid = int(qid)
        total = self.queue_changes.get(qid, 0)
        last_change = self.queue_last_change_step.get(qid, 0)
        return total, current_step - last_change

    def dump_paths_json(self):
        """Export current paths per queue to JSON for visualization."""
        export_data = {}
        for qid, pmap in self.paths_per_queue.items():
            q_list = []
            for (src, dst), path in pmap.items():
                q_list.append({
                    "src": src,
                    "dst": dst,
                    "path": path
                })
            export_data[qid] = q_list

        final_path = "/tmp/p4_paths.json"
        try:
            with open(final_path, "w") as f:
                json.dump(export_data, f)
                f.flush()
                os.fsync(f.fileno())

            try:
                os.chmod(final_path, 0o666)
            except OSError:
                pass

        except Exception:
            pass

    # -----------------------
    # Per-queue change tracking / revert
    # -----------------------

    def has_pending_change_for_qid(self, qid: int) -> bool:
        qid = int(qid)
        stack = self.change_history_by_qid.get(qid, [])
        return bool(stack)

    def has_pending_change(self) -> bool:
        return any(self.change_history_by_qid.get(q, []) for q in self.change_history_by_qid)

    def _revert_change_object(self, change: dict) -> bool:
        if not change:
            return False

        def _revert_overlay(side):
            if not side:
                return
            for ent in reversed(side.get("overlays", [])):
                sw = ent["sw"]
                dst_prefix = ent["dst_prefix"]
                dscp = ent["dscp"]
                before = ent["before"]
                if before is not None:
                    self._upsert_lpm(sw, dst_prefix, dscp, before[0], before[1])
                else:
                    try:
                        self._call(self.controllers[sw].table_delete_match, "l3_forward.ipv4_lpm", [dst_prefix, dscp])
                        try:
                            del self.forwarding_entries[sw]['lpm'][(dst_prefix, dscp)]
                        except Exception:
                            pass
                    except Exception:
                        pass
            old_path = side.get("old_path")
            new_path = side.get("new_path")
            if old_path and new_path:
                src_h = old_path[0]
                dst_h = old_path[-1]
                cur = self.path_map.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.path_map[(src_h, dst_h)] = old_path

        def _revert_legacy(side):
            if not side:
                return
            prev_node = side["prev_node"]
            dst_prefix = side["dst_prefix"]
            dscp = side["dscp"]
            alt_node = side["alt_node"]
            worst_node = side["worst_node"]

            if side["prev_before"] is not None:
                nh_ip, eport = side["prev_before"]
                self.update_path(prev_node, dst_prefix, dscp, nh_ip, eport)

            if side["alt_before"] is not None:
                nh_ip, eport = side["alt_before"]
                self.update_path(alt_node, dst_prefix, dscp, nh_ip, eport)
            else:
                try:
                    self._call(self.controllers[alt_node].table_delete_match, "l3_forward.ipv4_lpm", [dst_prefix, dscp])
                except Exception:
                    pass
                try:
                    del self.forwarding_entries[alt_node]['lpm'][(dst_prefix, dscp)]
                except Exception:
                    pass

            _, _, worst_before = side["worst_deleted"]
            if worst_before is not None:
                nh_ip, eport = worst_before
                self.update_path(worst_node, dst_prefix, dscp, nh_ip, eport)

            old_path = side.get("old_path")
            new_path = side.get("new_path")
            if old_path and new_path:
                src_h = old_path[0]
                dst_h = old_path[-1]
                cur = self.path_map.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.path_map[(src_h, dst_h)] = old_path

        def _revert_side(side):
            if side and "overlays" in side:
                _revert_overlay(side)
            else:
                _revert_legacy(side)

        _revert_side(change.get("fwd"))
        _revert_side(change.get("rev"))
        return True

    def revert_last_change_for_qid(self, qid: int) -> bool:
        qid = int(qid)
        stack = self.change_history_by_qid.get(qid, [])
        if not stack:
            return False
        change = stack.pop()
        res = self._revert_change_object(change)

        if res and change:
            fwd = change.get("fwd", {})
            old_p = fwd.get("old_path")
            if old_p:
                src = old_p[0]
                dst = old_p[-1]
                self.paths_per_queue[qid][(src, dst)] = old_p

            rev = change.get("rev", {})
            old_p_rev = rev.get("old_path")
            if old_p_rev:
                src = old_p_rev[0]
                dst = old_p_rev[-1]
                self.paths_per_queue[qid][(dst, src)] = old_p_rev

            self.dump_paths_json()

        return res

    def revert_last_change(self) -> bool:
        latest_q = None
        latest_idx = -1
        for q, stack in self.change_history_by_qid.items():
            if stack:
                if latest_q is None or id(stack[-1]) > id(self.change_history_by_qid[latest_q][-1]):
                    latest_q = q
                    latest_idx = len(stack) - 1
        if latest_q is None:
            return False
        change = self.change_history_by_qid[latest_q].pop(latest_idx)
        return self._revert_change_object(change)

    # -----------------------
    # Reroute using stored paths
    # -----------------------

    def _dscp_for_qid(self, qid: int) -> str:
        mapping = {
            0: "0x2E",
            1: "0x18",
            7: "0x00",
        }
        return mapping.get(int(qid), "0x00")

    def _host_name_from_ip(self, ip: str):
        return self.ip_to_host.get(ip)

    def _neighbor_iface_ip(self, neighbor: str, myself: str) -> str:
        return self.topo.node_to_node_interface_ip(neighbor, myself).split('/')[0]

    def reroute_one_demand_symmetric(self, src_ip: str, dst_ip: str, qid: int,
                                 worst_switch_id: int, alt_switch_name: str):
        """
        Install per-demand (/32) overlays for (src_ip,dst_ip) within DSCP of 'qid'
        along the entire new path where 'worst' is replaced by 'alt'.
        
        Strategy: A Hybrid Rerouting Approach
        1. Try simple local swap (stitching) first to preserve max path structure.
        2. If invalid, fall back to "Sticky Routing": shortest path calculation
           on a graph where original path edges are hyper-preferred (weight=0.01).
        3. Enforce strict symmetry for the return path.
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

        path_fwd_orig = self.path_map.get((src_host, dst_host))
        if not path_fwd_orig or worst_name not in path_fwd_orig:
            return False, "no stored forward path or worst not in path"
            
        # --- PATH CALCULATION STRATEGY ---
        
        # 1. Try Local Swap (Stitching)
        def _try_local_swap(original_path, w_name, a_name):
            try:
                newp = list(original_path)
                idx = newp.index(w_name)
                newp[idx] = a_name
                # Validate edges
                for i in range(len(newp) - 1):
                    if not self.net_graph.has_edge(newp[i], newp[i+1]):
                        return None
                return newp
            except ValueError:
                return None
                
        fwd_new_path = _try_local_swap(path_fwd_orig, worst_name, alt_switch_name)
        
        # 2. Fallback to Sticky Routing (if swap failed)
        if not fwd_new_path:
            # Create restricted graph (no bottleneck)
            G_temp = self.net_graph.copy()
            if worst_name in G_temp:
                G_temp.remove_node(worst_name)
            else:
                return False, "bottleneck node not in graph"
                
            # Bias towards existing path edges (Sticky Routing)
            # Default weight is 1.0. Set existing path edges to 0.01
            for i in range(len(path_fwd_orig) - 1):
                u, v = path_fwd_orig[i], path_fwd_orig[i+1]
                if G_temp.has_edge(u, v):
                    G_temp[u][v]['weight'] = 0.01
            
            try:
                # Compute path passing through alt
                # src -> ... -> alt
                p1 = nx.shortest_path(G_temp, src_host, alt_switch_name, weight='weight')
                # alt -> ... -> dst
                p2 = nx.shortest_path(G_temp, alt_switch_name, dst_host, weight='weight')
                
                # Merge (slice p2 to avoid duplicating alt node)
                fwd_new_path = p1 + p2[1:]
                
            except nx.NetworkXNoPath:
                return False, f"no physical path found via {alt_switch_name} (sticky fallback failed)"

        # 3. Enforce Symmetry for Reverse Path
        # The return path should be the exact reverse of the new forward path
        rev_new_path = list(reversed(fwd_new_path))
        path_rev_orig = self.path_map.get((dst_host, src_host))
        if not path_rev_orig:
             # If we don't have a stored rev path, we can't revert well, but we can try to proceed
             path_rev_orig = [] 

        # --- INSTALL OVERLAYS ---

        def _install_overlay_along_path(path: list[str], dst_h: str, dst_ip_: str):
            if not path or len(path) < 3:
                return None
            dst_prefix = f"{dst_ip_}/32"
            changes = []
            for i in range(1, len(path) - 1):
                sw = path[i]
                nxt = path[i + 1]
                if sw not in self.topo.get_p4switches().keys():
                    continue
                self.ensure_switching_and_mac(sw, nxt)

                if nxt in self.topo.get_p4switches().keys():
                    nh_ip = self._neighbor_iface_ip(nxt, sw)
                else:
                    nh_ip = dst_ip_

                eport = self.topo.node_to_node_port_num(sw, nxt)
                before = self.forwarding_entries.get(sw, {}).get('lpm', {}).get((dst_prefix, dscp))
                
                # Only upsert if different from current
                if before != (nh_ip, eport):
                    ok = self._upsert_lpm(sw, dst_prefix, dscp, nh_ip, eport)
                    if not ok:
                        # Rollback this path's changes on failure
                        for ent in reversed(changes):
                            b = ent["before"]
                            if b is not None:
                                self._upsert_lpm(ent["sw"], ent["dst_prefix"], ent["dscp"], b[0], b[1])
                            else:
                                try:
                                    self._call(
                                        self.controllers[ent["sw"]].table_delete_match,
                                        "l3_forward.ipv4_lpm", [ent["dst_prefix"], ent["dscp"]]
                                    )
                                    # Cleanup local cache
                                    try:
                                        del self.forwarding_entries[ent["sw"]]['lpm'][(ent["dst_prefix"], ent["dscp"])]
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                        return None

                changes.append({
                    "sw": sw,
                    "dst_prefix": dst_prefix,
                    "dscp": dscp,
                    "before": before,
                    "after": (nh_ip, eport),
                })
            return changes

        fwd_changes = _install_overlay_along_path(fwd_new_path, dst_host, dst_ip)
        rev_changes = _install_overlay_along_path(rev_new_path, src_host, src_ip)

        if fwd_changes is None and rev_changes is None:
             # Note: It's possible for one direction to fail installation if no changes were needed, 
             # but here None usually means installation error.
             # If both failed or one critical failed, we should conceptually rollback.
             # For now, simplistic check.
             if not fwd_changes and not rev_changes:
                return False, "failed to install overlays"

        # Update Path Maps
        self.path_map[(src_host, dst_host)] = fwd_new_path
        self.path_map[(dst_host, src_host)] = rev_new_path
        
        self.paths_per_queue[int(qid)][(src_host, dst_host)] = fwd_new_path
        self.paths_per_queue[int(qid)][(dst_host, src_host)] = rev_new_path

        # Record History for Revert
        rec = {
            "qid": int(qid),
            "fwd": {"old_path": path_fwd_orig, "new_path": fwd_new_path, "overlays": fwd_changes or []},
            "rev": {"old_path": path_rev_orig, "new_path": rev_new_path, "overlays": rev_changes or []},
        }
        self.change_history_by_qid.setdefault(int(qid), []).append(rec)
        self.dump_paths_json()

        return True, "ok"

    # ---------- Public helpers for RL agent logging ----------

    def host_from_ip(self, ip: str):
        return self.ip_to_host.get(ip)

    def get_path_by_hosts(self, src_host: str, dst_host: str):
        return self.path_map.get((src_host, dst_host))

    def get_path_by_ips(self, src_ip: str, dst_ip: str):
        sh = self.host_from_ip(src_ip)
        dh = self.host_from_ip(dst_ip)
        if not sh or not dh:
            return None
        return self.get_path_by_hosts(sh, dh)

    # -----------------------
    # Debug
    # -----------------------

    def print_paths(self):
        for (src, dst), path in sorted(self.path_map.items()):
            print(f"Path from {src} to {dst}: {' -> '.join(path)}")

    def print_forwarding_entries(self):
        for sw_name, entries in self.forwarding_entries.items():
            print(f"Switch: {sw_name}")
            for entry in entries:
                print(f"  {entry}")
