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

        # === Per-queue change history stacks (strictly per-qid) ===
        # Dict[int, list[change_record]]
        self.change_history_by_qid = {}

        self.connect_to_switches()
        self.build_network_graph()
        self.compute_forwarding_entries()  # also fills self.path_map
        self.program_switches()

    def connect_to_switches(self):
        for sw_name in self.topo.get_p4switches().keys():
            thrift_port = self.topo.get_thrift_port(sw_name)
            self.controllers[sw_name] = SimpleSwitchThriftAPI(thrift_port)

    def build_network_graph(self):
        p4switches = list(self.topo.get_p4switches().keys())
        hosts = list(self.topo.get_hosts().keys())
        nodes = p4switches + hosts
        self.net_graph.add_nodes_from(nodes)

        for node in nodes:
            neighbors = self.topo.get_neighbors(node)
            for neighbor in neighbors:
                if not self.net_graph.has_edge(node, neighbor):
                    self.net_graph.add_edge(node, neighbor, weight=1)

    def compute_forwarding_entries(self):
        hosts = list(self.topo.get_hosts().keys())
        dscp_list = ["0x2E", "0x18", "0x00"]

        self.forwarding_entries = {}
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

                self.paths.append(f"Path from {src_host} to {dst_host}: {' -> '.join(path)}")
                self.path_map[(src_host, dst_host)] = list(path)

                dst_ip = self.topo.get_host_ip(dst_host).split('/')[0]

                for i in range(1, len(path) - 1):
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
                        net = ip_network(f"{dst_ip}/24", strict=False).network_address
                        dst_prefix   = f"{net}/24"
                        next_hop_ip  = self.topo.node_to_node_interface_ip(next_hop, sw_name).split('/')[0]

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
                controller.table_add(
                    "port_forward.switching_table",
                    "set_dmac",
                    [next_hop_ip],
                    [next_hop_mac]
                )
            for egress_port, port_smac in tables['mac'].items():
                egress_port_hex = f"0x{egress_port:x}"
                controller.table_add(
                    "port_forward.mac_rewriting_table",
                    "set_smac",
                    [egress_port_hex],
                    [port_smac]
                )
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

    def _upsert_lpm(self, sw_name: str, dst_prefix: str, dscp: str, next_hop_ip: str, egress_port: int) -> bool:
        controller = self.controllers[sw_name]
        try:
            try:
                controller.table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
            except Exception:
                pass
            controller.table_add(
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
        port_smac   = self.topo.node_to_node_mac(sw_name, next_hop)

        if sw_name not in self.forwarding_entries:
            self.forwarding_entries[sw_name] = {'lpm': {}, 'switching': {}, 'mac': {}}

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

    def _role_of_sid(self, sid: int) -> str:
        return self.switch_id_role.get(int(sid), "other")

    def _neighbors_in_graph(self, node_name: str):
        return list(self.net_graph.neighbors(node_name))

    def _tors_connected_to_agg(self, agg_name: str):
        return [n for n in self._neighbors_in_graph(agg_name) if n.startswith("t")]

    def _is_same_pod_agg(self, candidate_agg: str, current_agg: str) -> bool:
        cur_tors = set(self._tors_connected_to_agg(current_agg))
        cand_tors = set(self._tors_connected_to_agg(candidate_agg))
        return len(cur_tors & cand_tors) > 0

    def find_alternate_for_worst(self, worst_switch_id: int, path: list[str]):
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return None

        role = self._role_of_sid(worst_switch_id)
        if role == "tor":
            return None

        if not path or worst_name not in path:
            return None

        idx = path.index(worst_name)
        if idx == 0 or idx == len(path) - 1:
            return None

        prev_node = path[idx - 1]
        next_node = path[idx + 1]

        sw_nodes = set(self.topo.get_p4switches().keys())
        sw_nodes.discard(worst_name)
        sw_nodes.discard(prev_node)
        sw_nodes.discard(next_node)

        candidates = sorted(sw_nodes)
        for cand in candidates:
            if self.net_graph.has_edge(prev_node, cand) and self.net_graph.has_edge(cand, next_node):
                return cand
        return None

    def has_alternate_for_worst(self, worst_switch_id: int, path: list[str]) -> bool:
        return self.find_alternate_for_worst(int(worst_switch_id), path) is not None

    # -----------------------
    # Per-queue change tracking / revert
    # -----------------------

    def has_pending_change_for_qid(self, qid: int) -> bool:
        qid = int(qid)
        stack = self.change_history_by_qid.get(qid, [])
        return bool(stack)

    # Back-compat global “any pending” (not used by RL env anymore)
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
                        self.controllers[sw].table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
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
            alt_node  = side["alt_node"]
            worst_node = side["worst_node"]

            if side["prev_before"] is not None:
                nh_ip, eport = side["prev_before"]
                self.update_path(prev_node, dst_prefix, dscp, nh_ip, eport)

            if side["alt_before"] is not None:
                nh_ip, eport = side["alt_before"]
                self.update_path(alt_node, dst_prefix, dscp, nh_ip, eport)
            else:
                try:
                    self.controllers[alt_node].table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
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
        change = stack.pop()  # per-queue LIFO
        return self._revert_change_object(change)

    # Back-compat: global revert (LIFO across all queues) — not used by RL env anymore
    def revert_last_change(self) -> bool:
        # Find newest change among all queues
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
    # Reroute using stored paths (path-tracking approach)
    # -----------------------

    def _dscp_for_qid(self, qid: int) -> str:
        mapping = {
            0: "0x2E",  # voice -> EF
            1: "0x18",  # video -> CS3
            7: "0x00",  # best-effort -> 00
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
        along the entire new path where 'worst' is replaced by 'alt'. No deletions.
        Transactional: on failure, roll back all newly-added entries.
        Records the change on the per-queue stack (qid).
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

        def _with_alt(path: list[str]):
            if not path or worst_name not in path:
                return None
            newp = list(path)
            idx = newp.index(worst_name)
            if idx == 0 or idx == len(newp) - 1:
                return None
            newp[idx] = alt_switch_name
            return newp

        def _install_overlay_along_path(path: list[str], dst_h: str, dst_ip_ : str):
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
                ok = self._upsert_lpm(sw, dst_prefix, dscp, nh_ip, eport)
                if not ok:
                    # rollback local changes made so far
                    for ent in reversed(changes):
                        b = ent["before"]
                        if b is not None:
                            self._upsert_lpm(ent["sw"], ent["dst_prefix"], ent["dscp"], b[0], b[1])
                        else:
                            try:
                                self.controllers[ent["sw"]].table_delete_match(
                                    "l3_forward.ipv4_lpm", [ent["dst_prefix"], ent["dscp"]]
                                )
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
                    "after":  (nh_ip, eport),
                })
            return changes

        path_fwd = self.path_map.get((src_host, dst_host))
        if not path_fwd:
            return False, "no stored forward path"
        fwd_new_path = _with_alt(path_fwd) if worst_name in path_fwd else None
        fwd_changes = _install_overlay_along_path(fwd_new_path, dst_host, dst_ip) if fwd_new_path else None

        path_rev = self.path_map.get((dst_host, src_host))
        if not path_rev:
            return False, "no stored reverse path"
        rev_new_path = _with_alt(path_rev) if worst_name in path_rev else None
        rev_changes = _install_overlay_along_path(rev_new_path, src_host, src_ip) if rev_new_path else None

        if fwd_changes is None and rev_changes is None:
            return False, "worst not present on stored path for either direction"

        if fwd_new_path:
            self.path_map[(src_host, dst_host)] = fwd_new_path
        if rev_new_path:
            self.path_map[(dst_host, src_host)] = rev_new_path

        rec = {
            "qid": int(qid),
            "fwd": {"old_path": path_fwd, "new_path": fwd_new_path if fwd_new_path else path_fwd, "overlays": fwd_changes or []},
            "rev": {"old_path": path_rev, "new_path": rev_new_path if rev_new_path else path_rev, "overlays": rev_changes or []},
        }
        self.change_history_by_qid.setdefault(int(qid), []).append(rec)
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
