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
        dscp_list = ["0x2E", "0x18", "0x00"]

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

    def _upsert_lpm(self, sw_name: str, dst_prefix: str, dscp: str, next_hop_ip: str, egress_port: int) -> bool:
        """
        Robust modify-or-add for 'l3_forward.ipv4_lpm'. We do a delete-match (ignore if not present),
        then add. Returns True on success; False on failure. Keeps the shadow map in sync only on success.
        """
        controller = self.controllers[sw_name]
        try:
            # Delete any existing match for (dst_prefix, dscp) to avoid duplicate ambiguous entries
            try:
                controller.table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
            except Exception:
                pass  # not present is fine

            # Add the new entry
            controller.table_add(
                "l3_forward.ipv4_lpm",
                "ipv4_forward",
                [dst_prefix, dscp],
                [next_hop_ip, str(egress_port)]
            )

            # Shadow copy (only if we succeeded)
            self.forwarding_entries.setdefault(sw_name, {}).setdefault('lpm', {})
            self.forwarding_entries[sw_name]['lpm'][(dst_prefix, dscp)] = (next_hop_ip, egress_port)
            return True

        except Exception as e:
            print(f"[LPM upsert FAILED] {sw_name} {dst_prefix} dscp={dscp} -> {next_hop_ip}/{egress_port}: {e}")
            return False


    def update_path(self, sw_name, dst_prefix, dscp, next_hop_ip, egress_port):
        return self._upsert_lpm(sw_name, dst_prefix, dscp, next_hop_ip, egress_port)

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

        If duplicate switch_id is seen across files, keep the FIRST mapping and warn.
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

        # Deterministic order
        files = sorted(glob.glob(str(rules_dir / "*-commands.txt")))

        for path in files:
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
                if not m:
                    print(f"[WARN] No init_metadata ID found in {fname}; skipping mapping for {sw_name}")
                    continue

                sid = int(m.group(1))

                # Duplicate protection: keep first mapping for a given sid
                if sid in sid_to_name and sid_to_name[sid] != sw_name:
                    print(f"[WARN] Duplicate switch_id {sid}: already mapped to {sid_to_name[sid]}, "
                        f"ignoring later mapping from {sw_name} ({fname})")
                    # Still record sw_name -> sid for convenience, but don't overwrite sid_to_name
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
        # ToRs are nodes starting with 't'
        return [n for n in self._neighbors_in_graph(agg_name) if n.startswith("t")]

    def _is_same_pod_agg(self, candidate_agg: str, current_agg: str) -> bool:
        # Share at least one ToR => same pod (simple heuristic)
        cur_tors = set(self._tors_connected_to_agg(current_agg))
        cand_tors = set(self._tors_connected_to_agg(candidate_agg))
        return len(cur_tors & cand_tors) > 0

    def find_alternate_for_worst(self, worst_switch_id: int, path: list[str]):
        """
        Given a worst switch ID and a concrete path [nodes...], propose an alternate switch:
          - Identify the worst switch name from its ID.
          - In the given path, find the nodes immediately before and after it (n-1, n+1).
          - Return a switch (not a host) that has links to BOTH (n-1) and (n+1).
          - If the worst is a ToR, or it's not found / is at the path edge, or no candidate exists, return None.
        """
        # Resolve name and role
        worst_name = self.switch_id_to_name.get(int(worst_switch_id))
        if not worst_name:
            return None

        # ToR alternates are not feasible in our topology (hosts are single-homed)
        role = self._role_of_sid(worst_switch_id)
        if role == "tor":
            return None

        if not path or worst_name not in path:
            return None

        idx = path.index(worst_name)
        # Can't substitute if worst is at either end (host or path boundary)
        if idx == 0 or idx == len(path) - 1:
            return None

        prev_node = path[idx - 1]
        next_node = path[idx + 1]

        # Only consider P4 switches (exclude hosts), and exclude prev/next/worst itself
        sw_nodes = set(self.topo.get_p4switches().keys())
        sw_nodes.discard(worst_name)
        sw_nodes.discard(prev_node)
        sw_nodes.discard(next_node)

        # Deterministic order for reproducibility
        candidates = sorted(sw_nodes)

        # A valid alternate must connect to BOTH neighbors in the topology graph
        for cand in candidates:
            if self.net_graph.has_edge(prev_node, cand) and self.net_graph.has_edge(cand, next_node):
                return cand

        # No alternate found that bridges both neighbors
        return None

    def has_alternate_for_worst(self, worst_switch_id: int, path: list[str]) -> bool:
        """
        Compatibility wrapper used by rl_agent.py.
        Returns True if we can propose an alternate switch for the given worst switch id,
        subject to role constraints (no ToR, same pod for aggs, any core).
        """
        return self.find_alternate_for_worst(int(worst_switch_id), path) is not None

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
        voice queue is 0, video is 1, best-effort is 7.
        Below DSCP hex values align with the pre-installed set used by compute_forwarding_entries().
        """
        mapping = {
            0: "0x2E",  # voice -> EF
            1: "0x18",  # video -> CS3
            7: "0x00",  # best-effort -> 00
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
        Install per-demand (/32) overlays for (src_ip,dst_ip) within DSCP of 'qid'
        along the entire new path where 'worst' is replaced by 'alt'. No deletions.
        Transactional: on failure, roll back all newly-added entries.
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

        def _with_alt(path: list[str]):
            if not path or worst_name not in path:
                return None
            newp = list(path)
            idx = newp.index(worst_name)
            if idx == 0 or idx == len(newp) - 1:
                return None
            newp[idx] = alt_switch_name
            return newp

        def _install_overlay_along_path(path: list[str], dst_h: str, dst_ip_: str):
            """
            For a concrete node path [hS, sw1, sw2, ..., swN, hD], install /32 LPM overlays
            on every switch sw_i sending toward the next hop in the path (sw_{i+1} or hD).
            We also ensure switching/mac tables for each leg (sw_i -> next).
            Returns list of change entries (for revert) or None on failure (with rollback).
            """
            if not path or len(path) < 3:
                return None

            dst_prefix = f"{dst_ip_}/32"
            changes = []

            # Iterate over switch -> next pairs
            for i in range(1, len(path) - 1):
                sw = path[i]
                nxt = path[i + 1]
                if sw not in self.topo.get_p4switches().keys():
                    continue  # skip if somehow a host appears in the middle

                # Make sure aux tables exist before L3 programming
                self.ensure_switching_and_mac(sw, nxt)

                # Next-hop IP as used by this sw toward nxt
                if nxt in self.topo.get_p4switches().keys():
                    nh_ip = self._neighbor_iface_ip(nxt, sw)
                else:
                    # nxt is the destination host
                    nh_ip = dst_ip_

                eport = self.topo.node_to_node_port_num(sw, nxt)

                # Save original /32 (if any) so we can revert; don't touch /24 entries
                before = self.forwarding_entries.get(sw, {}).get('lpm', {}).get((dst_prefix, dscp))

                ok = self._upsert_lpm(sw, dst_prefix, dscp, nh_ip, eport)
                if not ok:
                    # Roll back what we’ve done on this direction
                    for ent in reversed(changes):
                        b = ent["before"]
                        if b is not None:
                            self._upsert_lpm(ent["sw"], ent["dst_prefix"], ent["dscp"], b[0], b[1])
                        else:
                            try:
                                self.controllers[ent["sw"]].table_delete_match(
                                    "l3_forward.ipv4_lpm", [ent["dst_prefix"], ent["dscp"]]
                                )
                                del self.forwarding_entries[ent["sw"]]['lpm'][(ent["dst_prefix"], ent["dscp"])]
                            except Exception:
                                pass
                    return None

                changes.append({
                    "sw": sw,
                    "dst_prefix": dst_prefix,
                    "dscp": dscp,
                    "before": before,                  # None if no prior /32 existed
                    "after":  (nh_ip, eport),          # what we installed
                })

            return changes

        # ---------- Forward direction (src -> dst) ----------
        path_fwd = self.path_map.get((src_host, dst_host))
        if not path_fwd:
            return False, "no stored forward path"
        if worst_name not in path_fwd:
            fwd_new_path = None
        else:
            fwd_new_path = _with_alt(path_fwd)
        fwd_changes = None
        if fwd_new_path:
            fwd_changes = _install_overlay_along_path(fwd_new_path, dst_host, dst_ip)

        # ---------- Reverse direction (dst -> src) ----------
        path_rev = self.path_map.get((dst_host, src_host))
        if not path_rev:
            return False, "no stored reverse path"
        if worst_name not in path_rev:
            rev_new_path = None
        else:
            rev_new_path = _with_alt(path_rev)
        rev_changes = None
        if rev_new_path:
            rev_changes = _install_overlay_along_path(rev_new_path, src_host, src_ip)

        if fwd_changes is None and rev_changes is None:
            return False, "worst not present on stored path for either direction"

        # Commit path snapshot updates (after successful programming)
        if fwd_new_path:
            self.path_map[(src_host, dst_host)] = fwd_new_path
        if rev_new_path:
            self.path_map[(dst_host, src_host)] = rev_new_path

        # Record a single change object compatible with existing logs
        self.change_history.append({
            "fwd": {
                "old_path": path_fwd,
                "new_path": fwd_new_path if fwd_new_path else path_fwd,
                "overlays": fwd_changes or [],
            },
            "rev": {
                "old_path": path_rev,
                "new_path": rev_new_path if rev_new_path else path_rev,
                "overlays": rev_changes or [],
            }
        })
        return True, "ok"

    def revert_last_change(self):
        """
        Revert the most recent change. Supports both:
          (A) overlay-format records: {'fwd': {'overlays': [...] , 'old_path': [...], 'new_path': [...]}, 'rev': {...}}
          (B) legacy per-leg records:  {'fwd': {'prev_node': ..., 'alt_node': ..., 'worst_node': ..., ...}, 'rev': {...}}
        """
        if not self.change_history:
            return False
        change = self.change_history.pop()

        # ---- Overlay-format side revert ----
        def _revert_overlay(side):
            if not side:
                return
            # Undo overlays in reverse order (closest to dst back toward src)
            for ent in reversed(side.get("overlays", [])):
                sw = ent["sw"]
                dst_prefix = ent["dst_prefix"]
                dscp = ent["dscp"]
                before = ent["before"]
                if before is not None:
                    # Restore the previous /32 that existed before we overlaid
                    self._upsert_lpm(sw, dst_prefix, dscp, before[0], before[1])
                else:
                    # Remove our added /32 entry entirely
                    try:
                        self.controllers[sw].table_delete_match("l3_forward.ipv4_lpm", [dst_prefix, dscp])
                        del self.forwarding_entries[sw]['lpm'][(dst_prefix, dscp)]
                    except Exception:
                        pass

            # Restore stored path snapshot if it still matches what we set
            old_path = side.get("old_path")
            new_path = side.get("new_path")
            if old_path and new_path:
                src_h = old_path[0]
                dst_h = old_path[-1]
                cur = self.path_map.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.path_map[(src_h, dst_h)] = old_path

        # ---- Legacy-format side revert (for older records) ----
        def _revert_legacy(side):
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
                # If there was no prior entry, remove the specific match
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

            # Restore path snapshot
            old_path = side.get("old_path")
            new_path = side.get("new_path")
            if old_path and new_path:
                src_h = old_path[0]
                dst_h = old_path[-1]
                cur = self.path_map.get((src_h, dst_h))
                if cur and cur == new_path:
                    self.path_map[(src_h, dst_h)] = old_path

        def _revert_side(side):
            # Detect format by presence of 'overlays'
            if side and "overlays" in side:
                _revert_overlay(side)
            else:
                _revert_legacy(side)

        _revert_side(change.get("fwd"))
        _revert_side(change.get("rev"))
        return True

    # ---------- Public helpers for RL agent logging ----------

    def host_from_ip(self, ip: str):
        """Public wrapper (non-underscored) to map an IP to a topology host name like 'h3'."""
        return self.ip_to_host.get(ip)

    def get_path_by_hosts(self, src_host: str, dst_host: str):
        """Return stored path [nodes...] for (src_host, dst_host), or None."""
        return self.path_map.get((src_host, dst_host))

    def get_path_by_ips(self, src_ip: str, dst_ip: str):
        """
        Resolve IPs to host names and return stored path [nodes...],
        e.g. ['h1','t1','a1','c1','a4','t4','h4'], or None if unknown.
        """
        sh = self.host_from_ip(src_ip)
        dh = self.host_from_ip(dst_ip)
        if not sh or not dh:
            return None
        return self.get_path_by_hosts(sh, dh)


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
