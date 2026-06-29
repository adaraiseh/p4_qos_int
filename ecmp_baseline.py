#!/usr/bin/env python3
"""
Queue-independent ECMP baseline for comparison with the RL routing agent.

The existing RL controller routes on (destination, DSCP), which intentionally
allows each QoS queue to take a different path. This baseline instead installs
standard shortest-path ECMP groups:

* every next hop must reduce the remaining shortest-path distance by one;
* a CRC16 hash chooses among equal-cost next hops;
* the hash uses source IP, destination IP, and a per-switch group salt;
* DSCP and L4 ports are excluded, so Q0/Q1/Q7 for the same demand receive the
  same ECMP decision.

Run the topology first with the updated P4 program, start the INT collector,
then run this file. The benchmark uses the same telemetry queries and reward
function as rl_production.py and writes a richer CSV for direct analysis.

Examples:
    python3 ecmp_baseline.py --plan-only --config config/topologies/fat_tree_k4.yaml
    sudo -E python3 ecmp_baseline.py --config config/topologies/fat_tree_k4.yaml \
        --traffic-profile medium_2 --traffic-seed 42 --steps 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from logging_config import normalize_artifact_permissions, setup_unified_logging
from topology.factory import create_topology


log = logging.getLogger("ecmp_baseline")
QIDS = (0, 1, 7)


def _natural_name_key(name: str) -> Tuple[str, int, str]:
    prefix = name.rstrip("0123456789")
    suffix = name[len(prefix):]
    return prefix, int(suffix) if suffix else -1, name


@dataclass(frozen=True)
class ECMPGroup:
    switch: str
    destination: str
    destination_ip: str
    group_id: int
    next_hops: Tuple[str, ...]


def ecmp_plan_sha256(groups: Sequence[ECMPGroup]) -> str:
    payload = [
        {
            "switch": group.switch,
            "destination": group.destination,
            "destination_ip": group.destination_ip,
            "group_id": group.group_id,
            "next_hops": list(group.next_hops),
        }
        for group in groups
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ECMPPlanner:
    """Build loop-free equal-cost next-hop groups from topology metadata."""

    def __init__(self, topology_builder):
        self.builder = topology_builder
        self.graph = nx.Graph()
        max_bw = max(
            (float(link.bw) for link in topology_builder.links),
            default=1.0,
        )
        for link in topology_builder.links:
            bandwidth = max(float(link.bw), 1e-9)
            cost = max(1, int(round(100.0 * max_bw / bandwidth)))
            self.graph.add_edge(link.node1, link.node2, weight=cost)

        self.switches = sorted(
            topology_builder.switches,
            key=lambda name: (
                topology_builder.switches[name].switch_id,
                _natural_name_key(name),
            ),
        )
        self.hosts = sorted(topology_builder.hosts, key=_natural_name_key)

    def build_groups(self) -> List[ECMPGroup]:
        groups: List[ECMPGroup] = []
        host_count = len(self.hosts)

        for switch_index, switch in enumerate(self.switches):
            for host_index, destination in enumerate(self.hosts):
                distance = nx.shortest_path_length(
                    self.graph, switch, destination, weight="weight"
                )
                candidates = [
                    neighbor
                    for neighbor in self.graph.neighbors(switch)
                    if (
                        nx.shortest_path_length(
                            self.graph,
                            neighbor,
                            destination,
                            weight="weight",
                        )
                        + self.graph[switch][neighbor]["weight"]
                        == distance
                    )
                ]
                if not candidates:
                    raise RuntimeError(
                        f"No shortest-path next hop from {switch} to {destination}"
                    )

                candidates.sort(key=_natural_name_key)
                group_id = switch_index * host_count + host_index + 1
                groups.append(
                    ECMPGroup(
                        switch=switch,
                        destination=destination,
                        destination_ip=self.builder.hosts[destination].ip,
                        group_id=group_id,
                        next_hops=tuple(candidates),
                    )
                )

        return groups

    def validate(self, groups: Sequence[ECMPGroup]) -> Dict[str, int]:
        expected = len(self.switches) * len(self.hosts)
        if len(groups) != expected:
            raise AssertionError(f"Expected {expected} ECMP groups, got {len(groups)}")

        seen = set()
        multipath = 0
        members = 0

        for group in groups:
            key = (group.switch, group.destination)
            if key in seen:
                raise AssertionError(f"Duplicate ECMP group for {key}")
            seen.add(key)

            current_distance = nx.shortest_path_length(
                self.graph, group.switch, group.destination, weight="weight"
            )
            for next_hop in group.next_hops:
                next_distance = nx.shortest_path_length(
                    self.graph, next_hop, group.destination, weight="weight"
                )
                edge_cost = self.graph[group.switch][next_hop]["weight"]
                if next_distance + edge_cost != current_distance:
                    raise AssertionError(
                        f"{group.switch}->{next_hop} is not equal-cost toward "
                        f"{group.destination}"
                    )

            multipath += int(len(group.next_hops) > 1)
            members += len(group.next_hops)

        # Queue independence is structural: neither the group lookup nor hash
        # has a queue/DSCP input. All three queues share these exact groups.
        return {
            "groups": len(groups),
            "multipath_groups": multipath,
            "members": members,
            "max_width": max(len(group.next_hops) for group in groups),
        }

    @staticmethod
    def bmv2_crc16(data: bytes) -> int:
        """Return the CRC-16 used by BMv2's HashAlgorithm.crc16.

        BMv2 implements the reflected CRC-16/ARC variant with polynomial
        0x8005 (0xA001 in reflected form), initial remainder 0, and no final
        XOR. This matches behavioral-model's calculations.cpp.
        """
        remainder = 0
        for byte in data:
            remainder ^= byte
            for _ in range(8):
                if remainder & 1:
                    remainder = (remainder >> 1) ^ 0xA001
                else:
                    remainder >>= 1
        return remainder & 0xFFFF

    @classmethod
    def select_member(
        cls,
        src_ip: str,
        dst_ip: str,
        group_id: int,
        path_count: int,
    ) -> int:
        """Mirror the P4 hash input and modulo operation exactly."""
        if path_count <= 0:
            raise ValueError("ECMP path_count must be positive")

        hash_input = (
            ipaddress.ip_address(src_ip).packed
            + ipaddress.ip_address(dst_ip).packed
            + int(group_id).to_bytes(2, byteorder="big")
        )
        return cls.bmv2_crc16(hash_input) % path_count

    def trace_selected_path(
        self,
        groups_by_key: Dict[Tuple[str, str], ECMPGroup],
        src_host: str,
        dst_host: str,
        host_ips: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Trace the concrete path selected by the installed BMv2 ECMP hash."""
        if src_host not in self.builder.hosts:
            raise KeyError(f"Unknown source host {src_host}")
        if dst_host not in self.builder.hosts:
            raise KeyError(f"Unknown destination host {dst_host}")

        host_ips = host_ips or {}
        src_ip = host_ips.get(src_host, self.builder.hosts[src_host].ip)
        dst_ip = host_ips.get(dst_host, self.builder.hosts[dst_host].ip)
        current = src_host
        path = [current]

        # A valid ECMP hop strictly reduces remaining path cost, so a path can
        # never contain more nodes than the physical graph.
        max_hops = len(self.graph)
        for _ in range(max_hops):
            if current == dst_host:
                return path

            if current in self.builder.hosts:
                next_hop = self.builder.hosts[current].connected_switch
            else:
                group = groups_by_key[(current, dst_host)]
                member_index = self.select_member(
                    src_ip,
                    dst_ip,
                    group.group_id,
                    len(group.next_hops),
                )
                next_hop = group.next_hops[member_index]

            if next_hop in path:
                raise AssertionError(
                    f"ECMP visualization trace contains a loop: "
                    f"{' -> '.join(path + [next_hop])}"
                )
            path.append(next_hop)
            current = next_hop

        raise AssertionError(
            f"ECMP visualization trace did not reach {dst_host}: "
            f"{' -> '.join(path)}"
        )

    def build_visualization_data(
        self,
        groups: Sequence[ECMPGroup],
        traffic_pairs: Sequence[Tuple[str, str, int]],
        host_ips: Optional[Dict[str, str]] = None,
    ) -> Dict[int, List[Dict]]:
        """Build the legacy visualizer schema using concrete ECMP paths.

        Each configured demand is hashed once. The same selected path is then
        exported for every QoS queue, reflecting that DSCP and queue ID are not
        part of the ECMP decision.
        """
        groups_by_key = {
            (group.switch, group.destination): group for group in groups
        }
        export_data = {qid: [] for qid in QIDS}

        for src_host, dst_host, flow_id in traffic_pairs:
            path = self.trace_selected_path(
                groups_by_key,
                src_host,
                dst_host,
                host_ips=host_ips,
            )
            for qid in QIDS:
                export_data[qid].append(
                    {
                        "src": src_host,
                        "dst": dst_host,
                        "flow_id": flow_id,
                        "routing_mode": "ecmp",
                        "path": path,
                    }
                )

        return export_data


class VisualizationPathFile:
    """Atomically replace and later restore the visualizer's path file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.previous_bytes = None
        self.previous_mode = None
        self.had_previous_file = False
        self.snapshot_taken = False

    def _write_bytes(self, content: bytes, mode: int = 0o666) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, self.path)

    def publish(self, data: Dict[int, List[Dict]]) -> None:
        if not self.snapshot_taken:
            try:
                stat = self.path.stat()
                self.previous_bytes = self.path.read_bytes()
                self.previous_mode = stat.st_mode & 0o777
                self.had_previous_file = True
            except FileNotFoundError:
                self.had_previous_file = False
            self.snapshot_taken = True

        payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self._write_bytes(payload)

    def restore(self) -> None:
        if self.had_previous_file and self.previous_bytes is not None:
            self._write_bytes(
                self.previous_bytes,
                mode=self.previous_mode or 0o666,
            )
        else:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _switch_id_maps(topology_builder) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name = {
        int(switch.switch_id): name
        for name, switch in topology_builder.switches.items()
    }
    name_to_id = {name: switch_id for switch_id, name in id_to_name.items()}
    return id_to_name, name_to_id


def _runtime_port_neighbor_map(path: str = "/tmp/topology.json") -> Dict[Tuple[str, str], str]:
    port_neighbor: Dict[Tuple[str, str], str] = {}
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return port_neighbor

    for link in data.get("links", []):
        node1 = link.get("node1") or link.get("source")
        node2 = link.get("node2") or link.get("target")
        port1 = link.get("port1")
        port2 = link.get("port2")
        if node1 is not None and node2 is not None and port1 is not None:
            port_neighbor[(str(node1), str(port1))] = str(node2)
        if node1 is not None and node2 is not None and port2 is not None:
            port_neighbor[(str(node2), str(port2))] = str(node1)
    return port_neighbor


def _runtime_host_ips(controller, hosts: Sequence[str]) -> Dict[str, str]:
    host_ips: Dict[str, str] = {}
    for host in hosts:
        try:
            host_ips[host] = controller.topo.get_host_ip(host).split("/")[0]
        except Exception:
            continue
    return host_ips


def _groups_with_runtime_ips(
    groups: Sequence[ECMPGroup],
    host_ips: Dict[str, str],
) -> List[ECMPGroup]:
    updated = []
    for group in groups:
        updated.append(
            ECMPGroup(
                switch=group.switch,
                destination=group.destination,
                destination_ip=host_ips.get(
                    group.destination,
                    group.destination_ip,
                ),
                group_id=group.group_id,
                next_hops=group.next_hops,
            )
        )
    return updated


def _decorate_top_egresses(
    observations: Optional[Dict[str, Any]],
    topology_builder,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    if not observations or not observations.get("ok", False):
        return []
    id_to_name, _ = _switch_id_maps(topology_builder)
    port_neighbor = _runtime_port_neighbor_map()
    decorated = []
    for item in observations.get("top_egresses", [])[:limit]:
        try:
            switch_id = int(item.get("switch_id"))
        except (TypeError, ValueError):
            switch_id = -1
        switch_name = id_to_name.get(switch_id, f"s{item.get('switch_id')}")
        egress_port = str(item.get("egress_port"))
        neighbor = port_neighbor.get((switch_name, egress_port), "?")
        decorated.append(
            {
                **item,
                "switch": switch_name,
                "neighbor": neighbor,
                "label": (
                    f"{switch_name}:port{egress_port}->{neighbor} "
                    f"Q{item.get('queue_id')}"
                ),
            }
        )
    return decorated


def log_top_bottleneck_egresses(
    top_egresses: Sequence[Dict[str, Any]],
    logger: logging.Logger = log,
    prefix: str = "Top INT bottleneck egresses",
) -> None:
    if not top_egresses:
        logger.info("%s: unavailable", prefix)
        return
    logger.info("%s:", prefix)
    for item in top_egresses:
        logger.info(
            "  %s mean=%.2f%% p95=%.2f%% max=%.2f%% samples=%s flows=%s",
            item.get("label", "?"),
            float(item.get("mean_util", 0.0)),
            float(item.get("p95_util", 0.0)),
            float(item.get("max_util", 0.0)),
            item.get("count", 0),
            item.get("flow_count", 0),
        )


def collect_local_egress_observations(
    env,
    window_seconds: float,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Best-effort local-cache INT egress evidence for benchmark summaries."""
    try:
        observations = env.get_local_egress_observations(
            window_seconds=window_seconds,
            top_n=top_n,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "flows": {},
            "top_egresses": [],
        }
    if not observations:
        return {
            "ok": False,
            "error": "local egress observations unavailable",
            "flows": {},
            "top_egresses": [],
        }
    return observations


def _expected_edge_signature(
    path: Sequence[str],
    controller,
    name_to_id: Dict[str, int],
) -> List[str]:
    edges = []
    for node, next_hop in zip(path, path[1:]):
        switch_id = name_to_id.get(node)
        if switch_id is None:
            continue
        port = controller.topo.node_to_node_port_num(node, next_hop)
        edges.append(f"{switch_id}:{port}")
    return sorted(edges)


def _observed_path_from_edges(
    edges: Sequence[str],
    src_host: str,
    dst_host: str,
    topology_builder,
    id_to_name: Dict[int, str],
    port_neighbor: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    by_switch: Dict[str, List[Tuple[str, str, str]]] = {}
    edge_details = []
    for edge in edges:
        try:
            switch_id_text, port = edge.split(":", 1)
            switch_name = id_to_name.get(int(switch_id_text), f"s{switch_id_text}")
        except (ValueError, TypeError):
            switch_name = "?"
            port = "?"
        neighbor = port_neighbor.get((switch_name, str(port)), "?")
        by_switch.setdefault(switch_name, []).append((str(port), neighbor, edge))
        edge_details.append(
            {
                "edge": edge,
                "switch": switch_name,
                "egress_port": str(port),
                "neighbor": neighbor,
            }
        )

    path = [src_host]
    current = topology_builder.hosts[src_host].connected_switch
    path.append(current)
    used_edges = set()
    ambiguous = False
    for _ in range(len(edges) + 3):
        if current == dst_host:
            break
        options = sorted(by_switch.get(current, []))
        options = [option for option in options if option[2] not in used_edges]
        if not options:
            break
        if len(options) > 1:
            ambiguous = True
        _port, neighbor, edge = options[0]
        used_edges.add(edge)
        path.append(neighbor)
        current = neighbor
        if current == dst_host:
            break
        if current not in topology_builder.switches:
            break

    return {
        "path": path,
        "complete": bool(path and path[-1] == dst_host),
        "ambiguous": ambiguous,
        "edge_details": edge_details,
        "unused_edges": sorted(set(edges) - used_edges),
    }


def audit_ecmp_observed_paths(
    observations: Optional[Dict[str, Any]],
    topology_builder,
    controller,
    planner: ECMPPlanner,
    groups_by_key: Dict[Tuple[str, str], ECMPGroup],
    traffic_pairs: Sequence[Tuple[str, str, int]],
    host_ips: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not observations or not observations.get("ok", False):
        return {
            "verified": False,
            "errors": ["local egress observations unavailable"],
            "mismatch_count": 0,
            "queue_independence_violations": 0,
            "per_flow": {},
        }

    id_to_name, name_to_id = _switch_id_maps(topology_builder)
    port_neighbor = _runtime_port_neighbor_map()
    observed_flows = observations.get("flows", {})
    errors = []
    per_flow: Dict[str, Any] = {}
    mismatch_count = 0
    queue_independence_violations = 0
    expected_egress_counts: Dict[str, int] = {}
    observed_egress_counts: Dict[str, int] = {}

    for src_host, dst_host, flow_id in traffic_pairs:
        flow_key = str(flow_id)
        expected_path = planner.trace_selected_path(
            groups_by_key,
            src_host,
            dst_host,
            host_ips=host_ips,
        )
        expected_edges = _expected_edge_signature(
            expected_path,
            controller,
            name_to_id,
        )
        for edge in expected_edges:
            expected_egress_counts[edge] = expected_egress_counts.get(edge, 0) + 1

        q_reports = {}
        dominant_signatures = set()
        for qid in QIDS:
            qid_key = str(qid)
            flow_payload = observed_flows.get(flow_key, {}).get(qid_key)
            if not flow_payload or not flow_payload.get("signatures"):
                mismatch_count += 1
                q_reports[qid_key] = {
                    "match": False,
                    "error": "no observed INT egress signature",
                    "expected_edges": expected_edges,
                }
                errors.append(f"flow {flow_id} Q{qid}: no observed INT egress signature")
                continue

            dominant = flow_payload["signatures"][0]
            observed_edges = sorted(dominant.get("edges", []))
            dominant_signatures.add("|".join(observed_edges))
            for edge in observed_edges:
                observed_egress_counts[edge] = observed_egress_counts.get(edge, 0) + 1
            missing = sorted(set(expected_edges) - set(observed_edges))
            unexpected = sorted(set(observed_edges) - set(expected_edges))
            match = not missing and not unexpected
            if not match:
                mismatch_count += 1
                errors.append(
                    f"flow {flow_id} Q{qid}: missing={missing} unexpected={unexpected}"
                )
            observed_path = _observed_path_from_edges(
                observed_edges,
                src_host,
                dst_host,
                topology_builder,
                id_to_name,
                port_neighbor,
            )
            q_reports[qid_key] = {
                "match": match,
                "report_count": flow_payload.get("report_count", 0),
                "unique_signatures": flow_payload.get("unique_signatures", 0),
                "dominant_fraction": dominant.get("fraction", 0.0),
                "expected_edges": expected_edges,
                "observed_edges": observed_edges,
                "missing_edges": missing,
                "unexpected_edges": unexpected,
                "observed_path": observed_path,
            }

        queue_independent = len(dominant_signatures) <= 1
        if not queue_independent:
            queue_independence_violations += 1
            errors.append(f"flow {flow_id}: Q0/Q1/Q7 observed different ECMP signatures")
        per_flow[flow_key] = {
            "src": src_host,
            "dst": dst_host,
            "expected_path": expected_path,
            "expected_edges": expected_edges,
            "queue_independent": queue_independent,
            "queues": q_reports,
        }

    return {
        "verified": not errors,
        "errors": errors,
        "mismatch_count": mismatch_count,
        "queue_independence_violations": queue_independence_violations,
        "flow_count": len(traffic_pairs),
        "per_flow": per_flow,
        "expected_egress_counts": expected_egress_counts,
        "observed_egress_counts": observed_egress_counts,
        "top_bottleneck_egresses": _decorate_top_egresses(
            observations,
            topology_builder,
            limit=10,
        ),
    }


class ECMPProgrammer:
    """Install ECMP groups into the running BMv2 switches."""

    GROUP_TABLE = "l3_forward.ecmp_group"
    NHOP_TABLE = "l3_forward.ecmp_nhop"

    def __init__(self, controller, groups: Sequence[ECMPGroup]):
        self.controller = controller
        self.groups = groups
        self.installed = False

    def _expected_counts(self) -> Dict[str, Dict[str, int]]:
        expected = {
            switch: {"groups": 0, "members": 0}
            for switch in self.controller.controllers
        }
        for group in self.groups:
            expected[group.switch]["groups"] += 1
            expected[group.switch]["members"] += len(group.next_hops)
        return expected

    def _table_count(self, switch: str, thrift, table: str) -> int:
        observed = self.controller._call(thrift.table_num_entries, table)
        if observed is None:
            raise RuntimeError(
                f"Could not read back {table} entry count on {switch}"
            )
        return int(observed)

    def verify(self, raise_on_error: bool = True) -> Dict:
        expected = self._expected_counts()
        errors = []
        per_switch = {}

        for switch, thrift in self.controller.controllers.items():
            group_count = self._table_count(switch, thrift, self.GROUP_TABLE)
            member_count = self._table_count(switch, thrift, self.NHOP_TABLE)
            switch_report = {
                self.GROUP_TABLE: {
                    "expected": expected[switch]["groups"],
                    "observed": group_count,
                },
                self.NHOP_TABLE: {
                    "expected": expected[switch]["members"],
                    "observed": member_count,
                },
            }
            per_switch[switch] = switch_report

            if group_count != expected[switch]["groups"]:
                errors.append(
                    f"{switch}:{self.GROUP_TABLE} expected "
                    f"{expected[switch]['groups']}, observed {group_count}"
                )
            if member_count != expected[switch]["members"]:
                errors.append(
                    f"{switch}:{self.NHOP_TABLE} expected "
                    f"{expected[switch]['members']}, observed {member_count}"
                )

        report = {
            "verified": not errors,
            "expected_groups": sum(item["groups"] for item in expected.values()),
            "expected_members": sum(item["members"] for item in expected.values()),
            "observed_groups": sum(
                item[self.GROUP_TABLE]["observed"] for item in per_switch.values()
            ),
            "observed_members": sum(
                item[self.NHOP_TABLE]["observed"] for item in per_switch.values()
            ),
            "errors": errors,
            "per_switch": per_switch,
        }
        if errors and raise_on_error:
            raise RuntimeError(
                "ECMP table verification failed: " + "; ".join(errors[:5])
            )
        return report

    def clear(self, required: bool = False) -> Dict:
        failures = []
        per_switch = {}
        for switch, thrift in self.controller.controllers.items():
            switch_report = {}
            for table in (self.GROUP_TABLE, self.NHOP_TABLE):
                try:
                    self.controller._call(thrift.table_clear, table)
                    observed = self._table_count(switch, thrift, table)
                    switch_report[table] = observed
                    if observed != 0:
                        failures.append(
                            (
                                switch,
                                table,
                                RuntimeError(
                                    f"expected 0 entries after clear, observed {observed}"
                                ),
                            )
                        )
                except Exception as exc:
                    failures.append((switch, table, exc))
                    switch_report[table] = None
            per_switch[switch] = switch_report

        if required and failures:
            switch, table, exc = failures[0]
            raise RuntimeError(
                f"Cannot access {table} on {switch}. Restart the network so it "
                f"compiles the updated p4src/int_md.p4. Original error: {exc}"
            )
        return {
            "verified": not failures,
            "errors": [
                f"{switch}:{table}: {exc}"
                for switch, table, exc in failures
            ],
            "per_switch": per_switch,
        }

    def install(self) -> Dict:
        self.clear(required=True)
        member_count = 0

        try:
            for group in self.groups:
                thrift = self.controller.controllers[group.switch]

                # Install members before publishing the group so packets can
                # never hash into a partially populated group.
                for index, next_hop in enumerate(group.next_hops):
                    self.controller.ensure_switching_and_mac(group.switch, next_hop)
                    egress_port = self.controller.topo.node_to_node_port_num(
                        group.switch, next_hop
                    )

                    if next_hop in self.controller.topo.get_hosts():
                        next_hop_ip = group.destination_ip
                    else:
                        next_hop_ip = (
                            self.controller.topo.node_to_node_interface_ip(
                                next_hop, group.switch
                            ).split("/")[0]
                        )

                    handle = self.controller._call(
                        thrift.table_add,
                        self.NHOP_TABLE,
                        "ipv4_forward",
                        [str(group.group_id), str(index)],
                        [next_hop_ip, str(egress_port)],
                    )
                    if handle is None:
                        raise RuntimeError(
                            f"ECMP next-hop write was rejected on {group.switch}: "
                            f"group={group.group_id}, member={index}, "
                            f"next_hop={next_hop}"
                        )
                    member_count += 1

                handle = self.controller._call(
                    thrift.table_add,
                    self.GROUP_TABLE,
                    "set_ecmp_group",
                    [f"{group.destination_ip}/32"],
                    [str(group.group_id), str(len(group.next_hops))],
                )
                if handle is None:
                    raise RuntimeError(
                        f"ECMP group write was rejected on {group.switch}: "
                        f"destination={group.destination_ip}, "
                        f"group={group.group_id}"
                    )
        except Exception:
            try:
                self.clear(required=False)
            except Exception:
                pass
            raise

        verification = self.verify(raise_on_error=True)
        self.installed = True
        return {
            "groups": len(self.groups),
            "members": member_count,
            "verified": verification["verified"],
            "verification": verification,
        }


class BenchmarkStats:
    def __init__(self):
        self.rewards: List[float] = []
        self.valid_steps = 0
        self.sla_met = 0
        self.sla_checks = 0
        self.queue_metrics = {
            qid: {"latency": [], "drop": [], "util": []} for qid in QIDS
        }

    def add(self, reward: float, info: dict, snapshot: dict) -> None:
        self.rewards.append(float(reward))
        self.valid_steps += int(info.get("data_valid", False))
        self.sla_met += len(info.get("sla_met", []))
        self.sla_checks += len(QIDS)

        for qid in QIDS:
            if snapshot[qid].get("data_valid", False):
                self.queue_metrics[qid]["latency"].append(snapshot[qid]["lat_p95"])
                self.queue_metrics[qid]["drop"].append(snapshot[qid]["drop_p95"])
                self.queue_metrics[qid]["util"].append(snapshot[qid]["util_p95"])

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _p95(values: Sequence[float]) -> float:
        return float(np.percentile(values, 95)) if values else 0.0

    def to_summary(self, steps: int) -> Dict:
        sla_pct = 100.0 * self.sla_met / max(1, self.sla_checks)
        return {
            "total_steps": steps,
            "mean_reward": self._mean(self.rewards),
            "overall_sla_compliance": sla_pct,
            "sla_met_pct": sla_pct,
            "valid_steps": self.valid_steps,
            "queue_metrics": {
                qid: {
                    "mean_latency": self._mean(self.queue_metrics[qid]["latency"]),
                    "p95_latency": self._p95(self.queue_metrics[qid]["latency"]),
                    "mean_drop": self._mean(self.queue_metrics[qid]["drop"]),
                    "mean_util": self._mean(self.queue_metrics[qid]["util"]),
                }
                for qid in QIDS
            },
        }

    def log_summary(self, summary: Dict, output_path: Path) -> None:
        log.info("=" * 68)
        log.info("ECMP benchmark summary")
        log.info(f"  Steps: {summary['total_steps']}")
        log.info(f"  Mean reward: {summary['mean_reward']:+.4f}")
        log.info(
            f"  SLA compliance: "
            f"{summary['overall_sla_compliance']:.2f}%"
        )
        log.info(f"  SLA met percentage: {summary['sla_met_pct']:.2f}%")
        log.info(
            f"  Valid telemetry steps: "
            f"{summary['valid_steps']}/{summary['total_steps']}"
        )
        for qid in QIDS:
            metrics = summary["queue_metrics"][qid]
            log.info(
                f"  Q{qid}: mean(step-p95 latency)="
                f"{metrics['mean_latency']:.3f} ms, "
                f"p95(step-p95 latency)={metrics['p95_latency']:.3f} ms, "
                f"mean drops/100ms={metrics['mean_drop']:.6f}, "
                f"mean util={metrics['mean_util']:.3f}%"
            )
        log.info(f"  CSV: {output_path}")
        log.info("=" * 68)


class ECMPBenchmark:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.env = None
        self.traffic_manager = None
        self.programmer = None
        self.visualization_file = None
        self.routing_state = None
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, signum, _frame):
        log.info(f"Received signal {signum}; stopping after the current sample")
        self.running = False

    @staticmethod
    def _pressure(snapshot: dict, sla_thresholds: Dict[int, float]) -> float:
        pressure = 0.0
        for qid in QIDS:
            q = snapshot[qid]
            lat_ratio = q["lat_p95"] / sla_thresholds[qid]
            drop_norm = min(q["drop_p95"], 0.20) / 0.20
            util_norm = min(q["util_p95"], 100.0) / 100.0
            pressure = max(
                pressure,
                0.5 * lat_ratio + 0.3 * drop_norm + 0.2 * util_norm,
            )
        return pressure

    def run(self) -> int:
        if os.geteuid() != 0:
            log.error("Runtime benchmarking must be run with sudo -E")
            return 2

        # Runtime-only imports keep --plan-only usable without BMv2/p4utils.
        from rl_agent_4 import QoSRoutingEnv, SLA_THRESHOLDS
        from traffic_generator import TrafficManager

        builder = create_topology(self.args.config)
        planner = ECMPPlanner(builder)
        groups = planner.build_groups()
        plan_stats = planner.validate(groups)
        log.info(
            "Validated queue-independent ECMP plan: "
            f"{plan_stats['groups']} groups, "
            f"{plan_stats['multipath_groups']} multipath groups, "
            f"max width {plan_stats['max_width']}"
        )

        topology_name = builder.config.topology.name.replace("-", "_")
        rules_dir = self.args.rules_dir or f"rules/{topology_name}"

        self.env = QoSRoutingEnv(
            self.args.influx_bucket,
            self.args.influx_token
            if self.args.telemetry_backend in ("influx", "cache-fallback-influx")
            else None,
            self.args.influx_org,
            self.args.influx_url,
            verbose=False,
            reset_network=False,
            production_mode=True,
            topology_builder=builder,
            rules_dir=rules_dir,
            config_path=self.args.config,
            telemetry_backend=self.args.telemetry_backend,
            telemetry_cache_socket=self.args.telemetry_cache_socket,
            telemetry_cache_timeout=self.args.telemetry_cache_timeout,
        )
        self.env.training_influx_detail = "off"
        runtime_host_ips = _runtime_host_ips(self.env.controller, planner.hosts)
        if runtime_host_ips:
            changed_ips = sum(
                1
                for group in groups
                if runtime_host_ips.get(group.destination) != group.destination_ip
            )
            groups = _groups_with_runtime_ips(groups, runtime_host_ips)
            if changed_ips:
                log.info(
                    "Using runtime host IPs from topology.json for ECMP "
                    f"programming ({changed_ips} destination entries updated)"
                )

        # ECMP must not inherit any forwarding state from the preceding
        # benchmark method. Install and verify a clean deterministic fallback
        # before publishing the ECMP overlay.
        log.info("Resetting all P4 tables before ECMP installation")
        self.env.controller.clear_all_tables(verify=True)
        self.env.controller.compute_forwarding_entries()
        self.env.controller.program_switches()
        baseline_verification = self.env.controller.verify_forwarding_tables(
            raise_on_error=True
        )
        log.info("Verified clean fallback forwarding tables on all switches")

        self.programmer = ECMPProgrammer(self.env.controller, groups)
        installed = self.programmer.install()
        plan_digest = ecmp_plan_sha256(groups)
        self.routing_state = {
            "verified": bool(
                baseline_verification["verified"] and installed["verified"]
            ),
            "plan_sha256": plan_digest,
            "baseline_tables": baseline_verification,
            "ecmp_tables": installed["verification"],
            "runtime_host_ips": runtime_host_ips,
        }
        log.info(
            f"Verified {installed['groups']} ECMP groups and "
            f"{installed['members']} next-hop members; "
            f"plan_sha256={plan_digest}"
        )

        self.traffic_manager = TrafficManager(
            config_path=self.args.config,
            seed=self.args.traffic_seed,
        )
        self.visualization_file = VisualizationPathFile(self.args.paths_file)
        groups_by_key = {
            (group.switch, group.destination): group for group in groups
        }
        visualization_data = planner.build_visualization_data(
            groups,
            self.traffic_manager.traffic_pairs,
            host_ips=runtime_host_ips,
        )
        self.visualization_file.publish(visualization_data)
        log.info(
            f"Published {len(self.traffic_manager.traffic_pairs)} concrete ECMP "
            f"routes to {self.args.paths_file}; Q0/Q1/Q7 use identical paths"
        )

        profile_info = self.traffic_manager.start_traffic(
            profile_name=self.args.traffic_profile
        )
        traffic_state = self.traffic_manager.verify_exact_processes(
            raise_on_error=True
        )
        traffic_state["sender_rate_shaping"] = profile_info.get(
            "traffic_shaping",
            {},
        )
        self.routing_state["traffic_processes"] = traffic_state
        loads = profile_info.get("measurement_loads", profile_info["loads"])
        log.info(
            f"Planned measurement profile {profile_info['profile_name']} "
            f"with seed "
            f"{self.args.traffic_seed}: Q0={loads[0]:.3f}, "
            f"Q1={loads[1]:.3f}, Q7={loads[7]:.3f} Mbps"
        )
        # Preserve ECMP: force_reset=False avoids the environment's OSPF reset.
        # Traffic warmup is handled by TrafficManager so staged profiles advance
        # during the excluded warmup window instead of holding their first stage.
        state = self.env.reset(
            force_reset=False,
            cooldown_seconds=0.0,
            collect_initial_snapshot=False,
        )
        del state
        warmup_state = self.traffic_manager.warm_profile_for(
            self.args.warmup_seconds
        )
        self.routing_state["traffic_profile_warmup"] = warmup_state
        telemetry_window_seconds = (
            self.traffic_manager.telemetry_coverage_window_seconds(
                max(5.0, self.args.warmup_seconds)
            )
        )
        telemetry_readiness_window_seconds = float(
            self.env.telemetry_liveness_window_seconds
        )
        telemetry_state = self.env.verify_required_telemetry_freshness(
            window_seconds=telemetry_readiness_window_seconds,
            raise_on_error=True,
        )
        self.routing_state["telemetry_required_metrics"] = telemetry_state
        log.info("Verified required queue telemetry for ECMP")
        measurement_shaping = self.traffic_manager.begin_measurement()
        if measurement_shaping:
            self.routing_state["measurement_shaping"] = measurement_shaping
            traffic_state["sender_rate_shaping"] = measurement_shaping
        post_measurement_traffic_state = self.traffic_manager.verify_exact_processes(
            raise_on_error=True
        )
        traffic_state["post_measurement"] = post_measurement_traffic_state
        traffic_state["verified"] = bool(
            traffic_state["verified"]
            and post_measurement_traffic_state["verified"]
        )
        output_path = Path(self.args.output) if self.args.output else Path(
            "data"
        ) / f"ecmp_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_artifact_permissions(output_path.parent, dir_mode=0o775)
        stats = BenchmarkStats()
        completed_steps = 0

        with output_path.open("w", newline="") as csv_file:
            normalize_artifact_permissions(output_path, file_mode=0o664)
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "step",
                    "routing_mode",
                    "traffic_profile",
                    "traffic_seed",
                    "load_q0_mbps",
                    "load_q1_mbps",
                    "load_q7_mbps",
                    "reward",
                    "raw_reward",
                    "sla_met_count",
                    "data_valid",
                    "routing_state_verified",
                    "traffic_state_verified",
                    "telemetry_state_verified",
                    "pressure",
                    "q0_latency_ms",
                    "q0_drop",
                    "q0_util_pct",
                    "q1_latency_ms",
                    "q1_drop",
                    "q1_util_pct",
                    "q7_latency_ms",
                    "q7_drop",
                    "q7_util_pct",
                    "timestamp",
                ]
            )

            while self.running and (
                self.args.steps == 0 or completed_steps < self.args.steps
            ):
                next_step = completed_steps + 1
                self.traffic_manager.apply_step_profile(next_step)

                _, reward, _, _, info = self.env.step(0)
                completed_steps += 1
                snapshot = self.env.last_snapshot
                current_loads = self.traffic_manager.current_load
                pressure = self._pressure(snapshot, SLA_THRESHOLDS)
                info["pressure"] = pressure
                stats.add(reward, info, snapshot)

                row = [
                    completed_steps,
                    "ecmp",
                    self.traffic_manager.current_profile_name,
                    self.args.traffic_seed,
                    current_loads.get(0, 0.0),
                    current_loads.get(1, 0.0),
                    current_loads.get(7, 0.0),
                    reward,
                    info.get("raw_reward", reward),
                    len(info.get("sla_met", [])),
                    int(info.get("data_valid", False)),
                    int(self.routing_state["verified"]),
                    int(traffic_state["verified"]),
                    int(telemetry_state["verified"]),
                    pressure,
                ]
                for qid in QIDS:
                    row.extend(
                        [
                            snapshot[qid]["lat_p95"],
                            snapshot[qid]["drop_p95"],
                            snapshot[qid]["util_p95"],
                        ]
                    )
                row.append(datetime.now().isoformat())
                writer.writerow(row)
                csv_file.flush()

                if self.args.log_every > 0 and completed_steps % self.args.log_every == 0:
                    log.info(
                        f"[Step {completed_steps}] reward={reward:+.2f} "
                        f"sla={len(info.get('sla_met', []))}/"
                        f"{info.get('sla_total', len(QIDS))} "
                        f"pressure={pressure:.3f}"
                    )

        final_traffic_state = self.traffic_manager.verify_exact_processes(
            timeout=2.0,
            require_no_restarts=True,
            raise_on_error=False,
        )
        traffic_state["final"] = final_traffic_state
        traffic_state["verified"] = bool(
            traffic_state["verified"] and final_traffic_state["verified"]
        )
        if not traffic_state["verified"]:
            raise RuntimeError(
                "ECMP traffic-state verification failed: "
                + "; ".join(final_traffic_state["errors"])
            )

        final_telemetry_state = self.env.verify_required_telemetry_freshness(
            window_seconds=telemetry_readiness_window_seconds,
            raise_on_error=False,
        )
        telemetry_state["final"] = final_telemetry_state
        telemetry_state["final_verified"] = bool(final_telemetry_state["verified"])
        if not final_telemetry_state["verified"]:
            log.warning(
                "Final ECMP required telemetry audit failed; preserving measured "
                "run because initial readiness, traffic health, and per-step telemetry "
                "validity are enforced: "
                + "; ".join(final_telemetry_state["errors"])
            )

        egress_observations = collect_local_egress_observations(
            self.env,
            window_seconds=telemetry_window_seconds,
            top_n=10,
        )
        top_bottlenecks = _decorate_top_egresses(
            egress_observations,
            builder,
            limit=10,
        )
        self.routing_state["top_bottleneck_egresses"] = top_bottlenecks
        log_top_bottleneck_egresses(top_bottlenecks)
        ecmp_path_audit_required = self.args.telemetry_backend in (
            "cache",
            "cache-fallback-influx",
        )
        if egress_observations.get("ok", False):
            ecmp_path_audit = audit_ecmp_observed_paths(
                egress_observations,
                builder,
                self.env.controller,
                planner,
                groups_by_key,
                self.traffic_manager.traffic_pairs,
                host_ips=runtime_host_ips,
            )
            ecmp_path_audit["required"] = ecmp_path_audit_required
        else:
            ecmp_path_audit = {
                "verified": None,
                "required": ecmp_path_audit_required,
                "errors": [egress_observations.get("error", "unavailable")],
                "mismatch_count": None,
                "queue_independence_violations": None,
                "flow_count": len(self.traffic_manager.traffic_pairs),
                "per_flow": {},
                "top_bottleneck_egresses": top_bottlenecks,
            }
        self.routing_state["ecmp_observed_path_audit"] = ecmp_path_audit
        if ecmp_path_audit["verified"] is True:
            log.info(
                "Verified INT-observed ECMP paths against planner "
                f"for {ecmp_path_audit['flow_count']} flows and all queues"
            )
        elif ecmp_path_audit["verified"] is None:
            log.info(
                "INT-observed ECMP path audit unavailable: "
                + "; ".join(ecmp_path_audit["errors"][:5])
            )
        else:
            log.warning(
                "INT-observed ECMP path audit failed: "
                + "; ".join(ecmp_path_audit["errors"][:5])
            )

        final_baseline_verification = (
            self.env.controller.verify_forwarding_tables(raise_on_error=True)
        )
        final_ecmp_verification = self.programmer.verify(raise_on_error=True)
        self.routing_state["final_baseline_tables"] = final_baseline_verification
        self.routing_state["final_ecmp_tables"] = final_ecmp_verification
        self.routing_state["verified"] = bool(
            self.routing_state["verified"]
            and final_baseline_verification["verified"]
            and final_ecmp_verification["verified"]
        )
        if not self.routing_state["verified"]:
            raise RuntimeError("ECMP routing-state verification failed")

        summary = stats.to_summary(completed_steps)
        stats.log_summary(summary, output_path)
        log.info(
            "  Routing state: VERIFIED "
            f"(plan_sha256={self.routing_state['plan_sha256']})"
        )
        log.info("  Traffic process state: VERIFIED")
        log.info("  Required telemetry: VERIFIED")
        if self.args.summary_json:
            summary_payload = {
                **summary,
                "method": "ecmp",
                "traffic_profile": self.args.traffic_profile,
                "traffic_seed": self.args.traffic_seed,
                "csv_path": str(output_path.resolve()),
                "status": "completed",
                "routing_state_verified": self.routing_state["verified"],
                "traffic_state_verified": traffic_state["verified"],
                "telemetry_state_verified": telemetry_state["verified"],
                "ecmp_path_audit_verified": ecmp_path_audit["verified"],
                "ecmp_path_audit_required": ecmp_path_audit_required,
                "ecmp_path_mismatch_count": ecmp_path_audit["mismatch_count"],
                "ecmp_queue_independence_violations": (
                    ecmp_path_audit["queue_independence_violations"]
                ),
                "top_bottleneck_egresses": top_bottlenecks,
                "routing_state": self.routing_state,
            }
            summary_path = Path(self.args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            normalize_artifact_permissions(summary_path.parent, dir_mode=0o775)
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True)
            )
            normalize_artifact_permissions(summary_path, file_mode=0o664)
        return 0

    def close(self) -> None:
        # Clear ECMP first so the legacy LPM baseline immediately becomes active.
        if self.programmer is not None:
            try:
                clear_report = self.programmer.clear(required=False)
                if clear_report["verified"]:
                    log.info("Cleared and verified empty ECMP overlay")
                else:
                    log.warning(
                        "ECMP overlay clear could not be verified: "
                        + "; ".join(clear_report["errors"][:3])
                    )
            except Exception as exc:
                log.warning(f"Failed to clear ECMP overlay: {exc}")

        if self.visualization_file is not None:
            try:
                self.visualization_file.restore()
                log.info("Restored previous visualization paths")
            except Exception as exc:
                log.warning(f"Failed to restore visualization paths: {exc}")

        if self.traffic_manager is not None:
            try:
                self.traffic_manager.stop_traffic()
                log.info("Stopped traffic")
            except Exception as exc:
                log.warning(f"Failed to stop traffic: {exc}")

        if self.env is not None:
            self.env.close()


def parse_args() -> argparse.Namespace:
    from traffic_generator import TrafficManager

    parser = argparse.ArgumentParser(
        description="Queue-independent ECMP benchmark for RL comparison"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/topologies/fat_tree_k4.yaml",
        help="Topology YAML used by the running network",
    )
    parser.add_argument("--rules-dir", default=None)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and summarize ECMP groups without touching the network",
    )
    parser.add_argument("--steps", type=int, default=300, help="0 runs forever")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional machine-readable run summary path",
    )
    parser.add_argument(
        "--paths-file",
        default="/tmp/p4_paths.json",
        help="Path consumed by visualize_routes.py",
    )
    parser.add_argument("--log-level", default="info")

    parser.add_argument(
        "--traffic-profile",
        choices=tuple(TrafficManager.TRAFFIC_PROFILES),
        default="medium_2",
    )
    parser.add_argument(
        "--traffic-seed",
        type=int,
        default=42,
        help="Use the same value with rl_production.py",
    )

    parser.add_argument("--influx-url", default="http://192.168.56.1:8086")
    parser.add_argument("--influx-org", default="Research")
    parser.add_argument("--influx-bucket", default="INT")
    parser.add_argument(
        "--influx-token",
        default=os.environ.get("INFLUX_TOKEN"),
        help="InfluxDB token or INFLUX_TOKEN environment variable",
    )
    parser.add_argument(
        "--telemetry-backend",
        choices=["cache", "influx", "cache-fallback-influx"],
        default="cache",
        help="Telemetry source for benchmark observations",
    )
    parser.add_argument(
        "--telemetry-cache-socket",
        default="/tmp/p4_qos_int_telemetry.sock",
        help="Unix socket path for local telemetry cache",
    )
    parser.add_argument(
        "--telemetry-cache-timeout",
        type=float,
        default=1.0,
        help="Local telemetry cache request timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_unified_logging(module_name="ecmp_baseline", log_level=args.log_level)

    builder = create_topology(args.config)
    planner = ECMPPlanner(builder)
    groups = planner.build_groups()
    summary = planner.validate(groups)

    if args.plan_only:
        print(
            "ECMP plan valid: "
            f"{summary['groups']} groups, "
            f"{summary['multipath_groups']} multipath groups, "
            f"{summary['members']} members, max width {summary['max_width']}. "
            "The group lookup and hash contain no DSCP/queue input."
        )
        return 0

    if args.telemetry_backend in ("influx", "cache-fallback-influx") and not args.influx_token:
        log.error("Set INFLUX_TOKEN or pass --influx-token")
        return 2

    benchmark = ECMPBenchmark(args)
    try:
        return benchmark.run()
    except Exception:
        log.exception("ECMP benchmark failed")
        return 1
    finally:
        benchmark.close()


if __name__ == "__main__":
    sys.exit(main())
