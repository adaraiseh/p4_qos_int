# controller.py

import re
import json
import glob
import os
import threading
import logging
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache
from pathlib import Path
from ipaddress import ip_network
from typing import Optional, Dict, List, Set, Tuple

import networkx as nx
import warnings
# Suppress NetworkX 3.6 future warning coming from p4utils
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

# Import unified logging
from logging_config import setup_unified_logging

log = logging.getLogger(__name__)


class Controller:

    # Role normalization: edge switches (can't be rerouted around)
    EDGE_ROLES = {'leaf', 'access', 'tor'}
    # Role normalization: aggregation/spine switches
    AGG_ROLES = {'spine', 'distribution', 'agg'}
    # Role normalization: core switches
    CORE_ROLES = {'core'}

    # Change history depth limit to prevent unbounded memory growth
    MAX_HISTORY_DEPTH = 10  # Max changes per queue to store for rollback

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
        # Thread safety: RLock allows same thread to acquire multiple times (for nested calls)
        self._lock = threading.RLock()

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

        # Per-queue change history stacks - bounded to prevent memory growth
        # Each queue gets its own deque with max depth
        self.change_history_by_qid = {}

        # Usage tracking for alternatives
        self.switch_usage = {}
        self.queue_changes = {0: 0, 1: 0, 7: 0}
        self.queue_last_change_step = {0: 0, 1: 0, 7: 0}
        self.loop_detection_events = 0

        # Per-queue path tracking
        self.paths_per_queue = {0: {}, 1: {}, 7: {}}

        self.connect_to_switches()
        self.build_network_graph()
        self.compute_forwarding_entries()
        self.program_switches()
        self.dump_paths_json()

    def cleanup(self) -> None:
        """
        Clean up controller resources.

        Closes Thrift connections to all P4 switches.
        Critical for multi-topology training to prevent connection leaks.
        """
        with self._lock:
            for sw_name, controller in self.controllers.items():
                try:
                    # SimpleSwitchThriftAPI uses Thrift client internally
                    # Close the transport if available
                    if hasattr(controller, 'client') and hasattr(controller.client, '_iprot'):
                        if hasattr(controller.client._iprot.trans, 'close'):
                            controller.client._iprot.trans.close()
                except Exception as e:
                    log.warning(f"Error closing connection to {sw_name}: {e}")

    def __enter__(self):
        """Context manager support for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.cleanup()
        return False

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

        # Use one common OSPF-style dimensionless cost model for the RL reset
        # baseline and for alternate-path calculations. Cost is inversely
        # proportional to configured bandwidth and normalized so the fastest
        # link has cost 100. Equal-bandwidth topologies remain unchanged.
        configured_costs = {}
        if self._topology_builder is not None and self._topology_builder.links:
            max_bw = max(float(link.bw) for link in self._topology_builder.links)
            for link in self._topology_builder.links:
                bandwidth = max(float(link.bw), 1e-9)
                cost = max(1, int(round(100.0 * max_bw / bandwidth)))
                configured_costs[frozenset((link.node1, link.node2))] = cost

        for node in nodes:
            neighbors = self.topo.get_neighbors(node)
            for neighbor in neighbors:
                # Skip adding edges to collectors (not in our filtered nodes)
                if neighbor not in nodes:
                    continue
                if not self.net_graph.has_edge(node, neighbor):
                    weight = configured_costs.get(
                        frozenset((node, neighbor)),
                        1,
                    )
                    self.net_graph.add_edge(node, neighbor, weight=weight)

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
                    log.warning(f"No path between {src_host} and {dst_host}")
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
        with self._lock:
            self._program_switches_unlocked()

    def verify_forwarding_tables(self, raise_on_error: bool = True) -> Dict:
        """Read back baseline forwarding-table counts from every switch.

        P4Utils' Thrift helpers print several table-operation errors and return
        ``None`` instead of raising them.  Counting entries after a clean
        program operation gives benchmark runners an explicit treatment-
        integrity check instead of trusting attempted writes.
        """
        table_map = {
            'lpm': "l3_forward.ipv4_lpm",
            'switching': "port_forward.switching_table",
            'mac': "port_forward.mac_rewriting_table",
        }
        per_switch = {}
        errors = []

        with self._lock:
            for sw_name, thrift in self.controllers.items():
                expected_tables = self.forwarding_entries.get(
                    sw_name,
                    {'lpm': {}, 'switching': {}, 'mac': {}},
                )
                switch_report = {}
                for cache_name, table_name in table_map.items():
                    expected = len(expected_tables.get(cache_name, {}))
                    observed = self._call(thrift.table_num_entries, table_name)
                    if observed is None:
                        observed_value = None
                        errors.append(
                            f"{sw_name}:{table_name} count unavailable"
                        )
                    else:
                        observed_value = int(observed)
                        if observed_value != expected:
                            errors.append(
                                f"{sw_name}:{table_name} expected {expected}, "
                                f"observed {observed_value}"
                            )
                    switch_report[table_name] = {
                        'expected': expected,
                        'observed': observed_value,
                    }
                per_switch[sw_name] = switch_report

        report = {
            'verified': not errors,
            'errors': errors,
            'per_switch': per_switch,
        }
        if errors and raise_on_error:
            raise RuntimeError(
                "Baseline forwarding-table verification failed: "
                + "; ".join(errors[:5])
            )
        return report

    def _program_switches_unlocked(self):
        """Internal implementation without locking (for use within locked contexts)."""
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

    def clear_all_tables(self, verify: bool = False):
        """Clear all P4 tables on all switches and reset internal state.

        When ``verify`` is true, read every table back and fail if any entries
        remain. This is required for benchmark isolation because the Thrift
        wrapper can otherwise print and swallow clear failures.
        """
        failures = []
        with self._lock:
            for sw_name, controller in self.controllers.items():
                table_names = (
                    "l3_forward.ipv4_lpm",
                    "port_forward.switching_table",
                    "port_forward.mac_rewriting_table",
                    # ECMP must also be empty before RL/OSPF starts because it
                    # takes precedence over ipv4_lpm in the P4 pipeline.
                    "l3_forward.ecmp_group",
                    "l3_forward.ecmp_nhop",
                )
                for table_name in table_names:
                    try:
                        self._call(controller.table_clear, table_name)
                        if verify:
                            observed = self._call(
                                controller.table_num_entries,
                                table_name,
                            )
                            if observed is None:
                                failures.append(
                                    f"{sw_name}:{table_name} count unavailable "
                                    "after clear"
                                )
                            elif int(observed) != 0:
                                failures.append(
                                    f"{sw_name}:{table_name} still has "
                                    f"{int(observed)} entries after clear"
                                )
                    except Exception as exc:
                        failures.append(f"{sw_name}:{table_name}: {exc}")

            self.forwarding_entries.clear()
            self.change_history_by_qid.clear()
            self.switch_usage.clear()
            self.alt_rr_pos.clear()
            self.queue_changes = {0: 0, 1: 0, 7: 0}
            self.queue_last_change_step = {0: 0, 1: 0, 7: 0}

            for qid in self.paths_per_queue:
                self.paths_per_queue[qid].clear()

        if failures:
            message = "P4 table clear verification failed: " + "; ".join(
                failures[:5]
            )
            if verify:
                raise RuntimeError(message)
            log.warning(message)

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
            log.error(f"[LPM upsert FAILED] {sw_name} {dst_prefix} dscp={dscp} -> {next_hop_ip}/{egress_port}: {e}")
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
                    log.warning(f"No init_metadata ID found in {fname}; skipping mapping for {sw_name}")
                    continue

                sid = int(m.group(1))

                if sid in sid_to_name and sid_to_name[sid] != sw_name:
                    log.warning(f"Duplicate switch_id {sid}: already mapped to {sid_to_name[sid]}, "
                                f"ignoring later mapping from {sw_name} ({fname})")
                    name_to_sid[sw_name] = sid
                    continue

                sid_to_name.setdefault(sid, sw_name)
                name_to_sid[sw_name] = sid
                sid_role[sid] = role

            except Exception as e:
                log.error(f"Failed to parse {path}: {e}")

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

        An alternative is valid only if the controller can produce a safe
        simple path through that switch while avoiding the bottleneck.
        """
        with self._lock:
            # Use cached version with hashable arguments (CPU optimization)
            return list(self._cached_find_all_alternates(worst_switch_id, tuple(path)))

    @lru_cache(maxsize=256)
    def _cached_find_all_alternates(self, worst_switch_id: int, path_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Cached version of find_all_alternates.
        Returns tuple for hashability. Cache invalidated by _invalidate_alternates_cache().
        """
        return tuple(self._find_all_alternates_unlocked(worst_switch_id, list(path_tuple)))

    def _invalidate_alternates_cache(self):
        """Clear the alternates cache after topology changes or reroutes."""
        self._cached_find_all_alternates.cache_clear()

    def _find_all_alternates_unlocked(self, worst_switch_id: int, path: list[str]) -> list[str]:
        """Internal implementation without locking."""
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return []

        # Edge switches cannot be bypassed
        if self._is_edge_switch(worst_switch_id):
            return []

        if not path or worst_name not in path or len(path) < 2:
            return []

        has_loop, _ = self._detect_routing_loop(path)
        if has_loop:
            return []

        worst_role = self._normalize_role(self._role_of_sid(worst_switch_id))

        # Candidates are all switches EXCEPT:
        # 1. The bottleneck (worst) switch itself
        # 2. Edge switches (generally we don't route through other edge switches as transit)
        candidates = []
        all_sids = self.get_all_switch_ids()

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
            if not cand_name or cand_name not in self.net_graph:
                continue

            planned, _ = self._plan_safe_reroute_path(path, worst_name, cand_name)
            if planned is not None:
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

        # Get qid from change record for queue-specific path restoration
        qid = change.get("qid")

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
            if old_path and new_path and qid is not None:
                src_h = old_path[0]
                dst_h = old_path[-1]
                # Restore to queue-specific paths (not global path_map)
                queue_paths = self.paths_per_queue.get(int(qid), {})
                cur = queue_paths.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.paths_per_queue[int(qid)][(src_h, dst_h)] = old_path
                    log.debug(f"[Revert] Restored paths_per_queue[{qid}][({src_h}, {dst_h})] to original")
                elif cur:
                    log.warning(f"[Revert] paths_per_queue[{qid}][({src_h}, {dst_h})] changed since reroute, skipping restoration")
            elif not old_path:
                log.debug(f"[Revert] No old_path stored, skipping path restoration")

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
            if old_path and new_path and qid is not None:
                src_h = old_path[0]
                dst_h = old_path[-1]
                # Restore to queue-specific paths (not global path_map)
                queue_paths = self.paths_per_queue.get(int(qid), {})
                cur = queue_paths.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.paths_per_queue[int(qid)][(src_h, dst_h)] = old_path

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
        for q, stack in self.change_history_by_qid.items():
            if stack:
                if latest_q is None or id(stack[-1]) > id(self.change_history_by_qid[latest_q][-1]):
                    latest_q = q
        if latest_q is None:
            return False
        # Pop from deque (always pops last item)
        change = self.change_history_by_qid[latest_q].pop()
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

    def _detect_routing_loop(self, path: list[str]) -> tuple[bool, str]:
        """
        Detect if a path is unsafe for destination-only LPM forwarding.

        The installed overlay is keyed by destination prefix and DSCP only; it
        does not include ingress port or path position. Therefore any repeated
        switch in a computed path is unsafe even when the directed edges differ:
        the second visit would overwrite the next hop needed by the first
        visit and can blackhole or loop the queue.

        Args:
            path: List of node names forming the path

        Returns:
            (has_loop: bool, diagnostic_msg: str)
        """
        if not path or len(path) < 2:
            return False, ""

        first_seen = {}
        for idx, node in enumerate(path):
            if node in first_seen:
                segment = path[first_seen[node]:idx + 1]
                return (
                    True,
                    "LPM-unsafe repeated node "
                    f"{node}: {' -> '.join(segment)}",
                )
            first_seen[node] = idx

        edges_seen = {}  # edge -> first occurrence position
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            # Check for duplicate edge
            edge = (u, v)
            if edge in edges_seen:
                # Same directed edge appears twice - this is a loop!
                first_pos = edges_seen[edge]
                loop_path = path[first_pos:i+2]
                return True, f"Loop detected: {' -> '.join(loop_path)} (duplicate edge)"
            edges_seen[edge] = i

        return False, ""

    def _erase_repeated_nodes(self, path: list[str]) -> list[str]:
        """
        Remove looped path segments while preserving traversal order.

        Example:
            h1 -> t1 -> a1 -> c1 -> a1 -> t2 -> h4
        becomes:
            h1 -> t1 -> a1 -> t2 -> h4
        """
        stack = []
        positions = {}
        for node in path:
            if node in positions:
                keep_len = positions[node] + 1
                for removed in stack[keep_len:]:
                    positions.pop(removed, None)
                stack = stack[:keep_len]
                continue
            positions[node] = len(stack)
            stack.append(node)
        return stack

    def _plan_safe_reroute_path(
        self,
        original_path: list[str],
        worst_name: str,
        alt_switch_name: str,
    ) -> tuple[Optional[list[str]], str]:
        """
        Plan a safe reroute for a queue path.

        The P4 overlay only stores one next hop per (dst_prefix, dscp), so the
        installed path must be simple. The planner first tries a local
        replacement with loop erasure. If that is physically impossible, it
        computes a detour through the selected alternative with the bottleneck
        removed, and accepts it only if the final path is simple and every
        adjacent link physically exists.
        """
        def _validate_simple_path(candidate_path: list[str], label: str):
            if not candidate_path or len(candidate_path) < 3:
                return None, f"{label} path too short"

            p4_switches = self.topo.get_p4switches().keys()
            for node in candidate_path[1:-1]:
                if node not in p4_switches:
                    return None, f"{label} uses non-switch transit node {node}"

            for i in range(len(candidate_path) - 1):
                if not self.net_graph.has_edge(candidate_path[i], candidate_path[i + 1]):
                    return (
                        None,
                        f"{label} lacks link "
                        f"{candidate_path[i]} -> {candidate_path[i + 1]}",
                    )

            has_loop, loop_msg = self._detect_routing_loop(candidate_path)
            if has_loop:
                return None, f"{label} is unsafe: {loop_msg}"

            return candidate_path, "ok"

        if not original_path or len(original_path) < 3:
            return None, "path too short for local replacement"

        has_loop, loop_msg = self._detect_routing_loop(original_path)
        if has_loop:
            return None, f"current path is unsafe: {loop_msg}"

        try:
            idx = original_path.index(worst_name)
        except ValueError:
            return None, f"bottleneck {worst_name} not in path"

        if idx == 0 or idx == len(original_path) - 1:
            return None, "cannot replace path endpoint"

        new_path = list(original_path)
        new_path[idx] = alt_switch_name
        new_path = self._erase_repeated_nodes(new_path)

        planned, msg = _validate_simple_path(new_path, "local replacement")
        if planned is not None:
            return planned, msg

        # If a local replacement is physically impossible, allow a controlled
        # multi-hop detour through the chosen alternative. The bottleneck is
        # removed from the graph, and the candidate is accepted only when the
        # final installed path is still a simple physical path.
        src_node = original_path[0]
        dst_node = original_path[-1]
        if worst_name not in self.net_graph:
            return None, f"{msg}; bottleneck node not in graph"
        if alt_switch_name not in self.net_graph:
            return None, f"{msg}; alternative node not in graph"

        G_view = nx.restricted_view(self.net_graph, nodes=[worst_name], edges=[])
        if not (
            G_view.has_node(src_node)
            and G_view.has_node(dst_node)
            and G_view.has_node(alt_switch_name)
        ):
            return None, f"{msg}; reroute endpoint missing after bottleneck removal"

        try:
            p1 = nx.shortest_path(G_view, src_node, alt_switch_name, weight='weight')
            p2 = nx.shortest_path(G_view, alt_switch_name, dst_node, weight='weight')
        except nx.NetworkXNoPath:
            return None, f"{msg}; no safe detour via {alt_switch_name}"

        detour_path = p1 + p2[1:]
        planned, detour_msg = _validate_simple_path(detour_path, "multi-hop detour")
        if planned is None:
            return None, f"{msg}; {detour_msg}"

        return planned, "ok"

    def reroute_one_demand_symmetric(self, src_ip: str, dst_ip: str, qid: int,
                                 worst_switch_id: int, alt_switch_name: str):
        """
        Install per-demand (/32) overlays for (src_ip,dst_ip) within DSCP of 'qid'
        along the entire new path where 'worst' is replaced by 'alt'.

        Thread-safe wrapper that acquires lock before calling internal implementation.

        Strategy:
        1. Perform a local replacement.
        2. Erase repeated-node segments into a simple path when possible.
        3. Allow multi-hop detours through the selected alternative only when
           the final path is simple and the bottleneck is avoided.
        4. Enforce strict symmetry for the return path.
        """
        with self._lock:
            return self._reroute_one_demand_symmetric_unlocked(src_ip, dst_ip, qid, worst_switch_id, alt_switch_name)

    def _reroute_one_demand_symmetric_unlocked(self, src_ip: str, dst_ip: str, qid: int,
                                 worst_switch_id: int, alt_switch_name: str):
        """Internal implementation without locking."""
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

        # Use queue-specific path first (matches snapshot's path lookup)
        queue_paths = self.paths_per_queue.get(int(qid), {})
        path_fwd_orig = queue_paths.get((src_host, dst_host))
        if not path_fwd_orig:
            # Fallback to global path_map
            path_fwd_orig = self.path_map.get((src_host, dst_host))
        if not path_fwd_orig or worst_name not in path_fwd_orig:
            return False, "no stored forward path or worst not in path"
            
        fwd_new_path, plan_msg = self._plan_safe_reroute_path(
            path_fwd_orig,
            worst_name,
            alt_switch_name,
        )
        if not fwd_new_path:
            return False, f"unsafe reroute rejected: {plan_msg}"

        # Validate the final simple path before installing overlays.
        has_loop, loop_msg = self._detect_routing_loop(fwd_new_path)
        if has_loop:
            self.loop_detection_events += 1
            log.warning("=" * 60)
            log.warning("UNSAFE REROUTE PATH REJECTED")
            log.warning("=" * 60)
            log.warning(f"Source: {src_host}, Destination: {dst_host}")
            log.warning(f"Bottleneck: {worst_name}, Alternative: {alt_switch_name}")
            log.warning(f"Candidate path: {' -> '.join(fwd_new_path)}")
            log.warning(f"{loop_msg}")
            log.warning("=" * 60)
            return False, f"candidate path is unsafe for LPM overlay: {loop_msg}"

        # 3. Enforce Symmetry for Reverse Path
        # The return path should be the exact reverse of the new forward path
        rev_new_path = list(reversed(fwd_new_path))

        # Get reverse path from queue-specific storage first (consistent with forward path lookup)
        queue_paths_rev = self.paths_per_queue.get(int(qid), {})
        path_rev_orig = queue_paths_rev.get((dst_host, src_host))
        if not path_rev_orig:
            # Fallback to global path_map (baseline OSPF path)
            path_rev_orig = self.path_map.get((dst_host, src_host))
        if not path_rev_orig:
             # If we don't have a stored rev path, we can't revert properly
             # Use None (not []) so revert logic knows there's nothing to restore
             log.warning(f"[Reroute] No stored reverse path for ({dst_host}, {src_host}) - revert will not restore paths_per_queue")
             path_rev_orig = None 

        # --- INSTALL OVERLAYS ---

        def _install_overlay_along_path(path: list[str], dst_h: str, dst_ip_: str):
            if not path or len(path) < 3:
                log.warning(f"[Reroute] Path too short for overlay: path={path}, len={len(path) if path else 0}, dst={dst_ip_}")
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
                        log.warning(f"[Reroute] LPM upsert failed at sw={sw}, rolling back {len(changes)} changes")
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

                    # Only record change if upsert was actually performed
                    changes.append({
                        "sw": sw,
                        "dst_prefix": dst_prefix,
                        "dscp": dscp,
                        "before": before,
                        "after": (nh_ip, eport),
                    })
            return changes

        log.debug(f"[Reroute] Installing overlays: fwd_path={' -> '.join(fwd_new_path)}, rev_path={' -> '.join(rev_new_path)}")
        fwd_changes = _install_overlay_along_path(fwd_new_path, dst_host, dst_ip)
        rev_changes = _install_overlay_along_path(rev_new_path, src_host, src_ip)

        # CRITICAL: If EITHER direction fails, we must rollback and fail the entire operation
        # to prevent asymmetric routing (packets going one way but not returning correctly)
        if fwd_changes is None or rev_changes is None:
            fwd_status = "ok" if fwd_changes is not None else "FAILED"
            rev_status = "ok" if rev_changes is not None else "FAILED"
            log.warning(f"[Reroute] Asymmetric failure detected: fwd={fwd_status}, rev={rev_status}")

            # Rollback successful direction to maintain consistency
            if fwd_changes is not None:
                log.warning(f"[Reroute] Rolling back {len(fwd_changes)} forward path changes")
                for ent in reversed(fwd_changes):
                    b = ent["before"]
                    if b is not None:
                        self._upsert_lpm(ent["sw"], ent["dst_prefix"], ent["dscp"], b[0], b[1])
                    else:
                        try:
                            self._call(
                                self.controllers[ent["sw"]].table_delete_match,
                                "l3_forward.ipv4_lpm", [ent["dst_prefix"], ent["dscp"]]
                            )
                            try:
                                del self.forwarding_entries[ent["sw"]]['lpm'][(ent["dst_prefix"], ent["dscp"])]
                            except Exception:
                                pass
                        except Exception:
                            pass

            if rev_changes is not None:
                log.warning(f"[Reroute] Rolling back {len(rev_changes)} reverse path changes")
                for ent in reversed(rev_changes):
                    b = ent["before"]
                    if b is not None:
                        self._upsert_lpm(ent["sw"], ent["dst_prefix"], ent["dscp"], b[0], b[1])
                    else:
                        try:
                            self._call(
                                self.controllers[ent["sw"]].table_delete_match,
                                "l3_forward.ipv4_lpm", [ent["dst_prefix"], ent["dscp"]]
                            )
                            try:
                                del self.forwarding_entries[ent["sw"]]['lpm'][(ent["dst_prefix"], ent["dscp"])]
                            except Exception:
                                pass
                        except Exception:
                            pass

            return False, f"failed to install overlays (fwd={fwd_status}, rev={rev_status})"

        # Both directions succeeded - also check for empty changes (no-op case)
        if not fwd_changes and not rev_changes:
            # No changes were needed in either direction - this is OK, not a failure
            pass

        # Update Queue-Specific Path Maps (atomic transaction with rollback on failure)
        # NOTE: We only update paths_per_queue[qid], NOT the global path_map.
        # path_map stores the immutable baseline OSPF paths and is used as fallback.
        # This ensures queue independence - Q0's reroute doesn't affect Q1's path lookup.
        old_queue_path_fwd = self.paths_per_queue[int(qid)].get((src_host, dst_host))
        old_queue_path_rev = self.paths_per_queue[int(qid)].get((dst_host, src_host))

        try:
            # Apply queue-specific updates only (path_map remains immutable)
            self.paths_per_queue[int(qid)][(src_host, dst_host)] = fwd_new_path
            self.paths_per_queue[int(qid)][(dst_host, src_host)] = rev_new_path
        except Exception as e:
            # Rollback queue-specific paths on any exception
            if old_queue_path_fwd is not None:
                self.paths_per_queue[int(qid)][(src_host, dst_host)] = old_queue_path_fwd
            if old_queue_path_rev is not None:
                self.paths_per_queue[int(qid)][(dst_host, src_host)] = old_queue_path_rev
            return False, f"Path update failed: {e}"

        # Record History for Revert (bounded deque to prevent memory growth)
        rec = {
            "qid": int(qid),
            "fwd": {"old_path": path_fwd_orig, "new_path": fwd_new_path, "overlays": fwd_changes or []},
            "rev": {"old_path": path_rev_orig, "new_path": rev_new_path, "overlays": rev_changes or []},
        }
        # Initialize deque with maxlen if not exists, then append
        if int(qid) not in self.change_history_by_qid:
            self.change_history_by_qid[int(qid)] = deque(maxlen=self.MAX_HISTORY_DEPTH)
        self.change_history_by_qid[int(qid)].append(rec)
        self.dump_paths_json()

        # Invalidate alternates cache since path changed (CPU optimization cache management)
        self._invalidate_alternates_cache()

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

    def get_path_by_ips_for_queue(self, src_ip: str, dst_ip: str, qid: int):
        """Get the path for a specific queue between two IPs.

        Each queue can have its own rerouted path stored in paths_per_queue.
        This method looks up the queue-specific path first, falling back to
        the global path_map if no queue-specific path exists.

        Args:
            src_ip: Source IP address
            dst_ip: Destination IP address
            qid: Queue ID (0=voice, 1=video, 7=BE)

        Returns:
            List of node names representing the path, or None if not found.
        """
        sh = self.host_from_ip(src_ip)
        dh = self.host_from_ip(dst_ip)
        if not sh or not dh:
            return None
        # Try queue-specific path first (has rerouted paths per queue)
        queue_paths = self.paths_per_queue.get(int(qid), {})
        path = queue_paths.get((sh, dh))
        if path:
            return path
        # Fallback to global path_map
        return self.path_map.get((sh, dh))

    # -----------------------
    # Debug
    # -----------------------

    def print_paths(self):
        for (src, dst), path in sorted(self.path_map.items()):
            log.debug(f"Path from {src} to {dst}: {' -> '.join(path)}")

    def print_forwarding_entries(self):
        for sw_name, entries in self.forwarding_entries.items():
            log.debug(f"Switch: {sw_name}")
            for entry in entries:
                log.debug(f"  {entry}")
