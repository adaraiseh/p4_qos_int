#!/usr/bin/env python3
"""
Test script for smart loop detection in controller.py
"""

import sys
from pathlib import Path

import networkx as nx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Mock the controller's loop detection method
def _detect_routing_loop(path: list[str]) -> tuple[bool, str]:
    """
    Detect if a path is unsafe for destination-only LPM forwarding.

    The installed overlay is keyed by destination prefix and DSCP only; it
    does not include ingress port or path position. Therefore any repeated
    switch in a computed path is unsafe even when the directed edges differ.

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


def _erase_repeated_nodes(path):
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


def _plan_safe_reroute_path(original_path, worst_name, alt_switch_name, graph):
    def _validate_simple_path(candidate_path, label):
        if not candidate_path or len(candidate_path) < 3:
            return None, f"{label} path too short"

        for node in candidate_path[1:-1]:
            if node.startswith('h'):
                return None, f"{label} uses non-switch transit node {node}"

        for i in range(len(candidate_path) - 1):
            if not graph.has_edge(candidate_path[i], candidate_path[i + 1]):
                return (
                    None,
                    f"{label} lacks link "
                    f"{candidate_path[i]} -> {candidate_path[i + 1]}",
                )

        has_loop, loop_msg = _detect_routing_loop(candidate_path)
        if has_loop:
            return None, f"{label} is unsafe: {loop_msg}"

        return candidate_path, "ok"

    if not original_path or len(original_path) < 3:
        return None, "path too short for local replacement"

    has_loop, loop_msg = _detect_routing_loop(original_path)
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
    new_path = _erase_repeated_nodes(new_path)

    planned, msg = _validate_simple_path(new_path, "local replacement")
    if planned is not None:
        return planned, msg

    if worst_name not in graph:
        return None, f"{msg}; bottleneck node not in graph"
    if alt_switch_name not in graph:
        return None, f"{msg}; alternative node not in graph"

    graph_view = nx.restricted_view(graph, nodes=[worst_name], edges=[])
    try:
        p1 = nx.shortest_path(graph_view, original_path[0], alt_switch_name, weight='weight')
        p2 = nx.shortest_path(graph_view, alt_switch_name, original_path[-1], weight='weight')
    except nx.NetworkXNoPath:
        return None, f"{msg}; no safe detour via {alt_switch_name}"

    detour_path = p1 + p2[1:]
    planned, detour_msg = _validate_simple_path(detour_path, "multi-hop detour")
    if planned is None:
        return None, f"{msg}; {detour_msg}"

    return planned, "ok"


def _graph(edges):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def run_tests():
    """Run test cases for loop detection."""

    print("=" * 70)
    print("Testing Smart Loop Detection")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: No duplicates - should pass
    print("\n[Test 1] Simple path with no duplicates")
    path1 = ['h1', 't1', 'a1', 'c1', 'a5', 't6', 'h16']
    print(f"Path: {' -> '.join(path1)}")
    has_loop, msg = _detect_routing_loop(path1)
    if not has_loop:
        print("✓ PASS: No loop detected (correct)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: False positive - {msg}")
        tests_failed += 1

    # Test 2: Node revisit with different edges - unsafe for LPM overlays
    print("\n[Test 2] Repeated core switch with different edges")
    path2 = ['h1', 't1', 'a1', 'c1', 'a3', 'c1', 'a7', 't8', 'h16']
    print(f"Path: {' -> '.join(path2)}")
    print("Edges: h1->t1, t1->a1, a1->c1, c1->a3, a3->c1, c1->a7, a7->t8, t8->h16")
    print("Analysis: c1 appears at positions 3 and 5. That is unsafe because")
    print("the P4 table cannot select next hop by path position:")
    print("  - First: a1->c1 (ingress from a1)")
    print("  - Second: a3->c1 (ingress from a3)")
    has_loop, msg = _detect_routing_loop(path2)
    if has_loop:
        print(f"✓ PASS: Unsafe repeated node rejected - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Repeated node was incorrectly accepted")
        tests_failed += 1

    # Test 3: True loop (2-node cycle) - should fail
    print("\n[Test 3] True loop: 2-node cycle (a1->c1->a1)")
    path3 = ['h1', 't1', 'a1', 'c1', 'a1', 'c2', 't2', 'h2']
    print(f"Path: {' -> '.join(path3)}")
    print("Edges: h1->t1, t1->a1, a1->c1, c1->a1, a1->c2, c2->t2, t2->h2")
    print("Analysis: Edge a1->c1 followed later by c1->a1 creates cycle")
    has_loop, msg = _detect_routing_loop(path3)
    if has_loop:
        print(f"✓ PASS: Loop detected (correct) - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Failed to detect actual loop")
        tests_failed += 1

    # Test 4: Multiple node revisits - unsafe for LPM overlays
    print("\n[Test 4] Multiple node revisits")
    path4 = ['h1', 't1', 'a1', 'c1', 'a3', 'c2', 'a1', 'c3', 'a5', 't6', 'h16']
    print(f"Path: {' -> '.join(path4)}")
    print("Analysis: a1 appears at positions 2 and 6, which is unsafe")
    has_loop, msg = _detect_routing_loop(path4)
    if has_loop:
        print(f"✓ PASS: Unsafe repeated node rejected - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Repeated node was incorrectly accepted")
        tests_failed += 1

    # Test 5: Self-loop - should fail
    print("\n[Test 5] Self-loop (node connects to itself)")
    path5 = ['h1', 't1', 'a1', 'a1', 'c1', 't2', 'h2']
    print(f"Path: {' -> '.join(path5)}")
    print("Edges: h1->t1, t1->a1, a1->a1, ...")
    has_loop, msg = _detect_routing_loop(path5)
    if has_loop:
        print(f"✓ PASS: Loop detected (correct) - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Failed to detect self-loop")
        tests_failed += 1

    # Test 6: Empty path - should pass
    print("\n[Test 6] Edge case: Empty path")
    path6 = []
    has_loop, msg = _detect_routing_loop(path6)
    if not has_loop:
        print("✓ PASS: No loop detected (correct)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: False positive on empty path - {msg}")
        tests_failed += 1

    # Test 7: Single node path - should pass
    print("\n[Test 7] Edge case: Single node")
    path7 = ['h1']
    has_loop, msg = _detect_routing_loop(path7)
    if not has_loop:
        print("✓ PASS: No loop detected (correct)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: False positive on single node - {msg}")
        tests_failed += 1

    # Test 8: Original user example 1
    print("\n[Test 8] Original repeated-core example 1 (should be rejected)")
    p1 = ['h1', 't1', 'a1', 'c1', 'a3']
    p2 = ['a3', 'c1', 'a7', 't8', 'h16']
    merged = p1 + p2[1:]
    print(f"P1: {' -> '.join(p1)}")
    print(f"P2: {' -> '.join(p2)}")
    print(f"Merged: {' -> '.join(merged)}")
    has_loop, msg = _detect_routing_loop(merged)
    if has_loop:
        print(f"✓ PASS: Unsafe repeated node rejected - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Repeated node was incorrectly accepted")
        tests_failed += 1

    # Test 9: Original user example 2
    print("\n[Test 9] Original repeated-core example 2 (should be rejected)")
    p1 = ['h11', 't6', 'a6', 'c3', 'a4']
    p2 = ['a4', 'c3', 'a2', 't1', 'h2']
    merged = p1 + p2[1:]
    print(f"P1: {' -> '.join(p1)}")
    print(f"P2: {' -> '.join(p2)}")
    print(f"Merged: {' -> '.join(merged)}")
    has_loop, msg = _detect_routing_loop(merged)
    if has_loop:
        print(f"✓ PASS: Unsafe repeated node rejected - {msg}")
        tests_passed += 1
    else:
        print("✗ FAIL: Repeated node was incorrectly accepted")
        tests_failed += 1

    # Test 10: Safe one-hop replacement
    print("\n[Test 10] Safe one-hop replacement")
    graph10 = _graph([
        ('h1', 't1'), ('t1', 'a1'), ('a1', 'c1'), ('c1', 'a5'),
        ('a5', 't6'), ('t6', 'h16'), ('a1', 'c2'), ('c2', 'a5'),
    ])
    path10 = ['h1', 't1', 'a1', 'c1', 'a5', 't6', 'h16']
    planned, msg = _plan_safe_reroute_path(path10, 'c1', 'c2', graph10)
    expected10 = ['h1', 't1', 'a1', 'c2', 'a5', 't6', 'h16']
    if planned == expected10:
        print("PASS: Local replacement accepted")
        tests_passed += 1
    else:
        print(f"FAIL: Safe local replacement rejected - {msg}")
        tests_failed += 1

    # Test 11: Allow safe non-local aggregation detour
    print("\n[Test 11] Allow safe non-local aggregation detour")
    graph11 = _graph([
        ('h11', 't6'), ('t6', 'a5'), ('a5', 'c1'), ('c1', 'a1'),
        ('a1', 't1'), ('t1', 'h2'), ('t6', 'a6'), ('a6', 'c3'),
        ('c3', 'a2'), ('a2', 't1'),
    ])
    path11 = ['h11', 't6', 'a5', 'c1', 'a1', 't1', 'h2']
    planned, msg = _plan_safe_reroute_path(path11, 'a1', 'a2', graph11)
    expected11 = ['h11', 't6', 'a6', 'c3', 'a2', 't1', 'h2']
    if planned == expected11:
        print("PASS: Safe multi-hop detour accepted")
        tests_passed += 1
    else:
        print(f"FAIL: Safe multi-hop detour rejected - {msg}, planned={planned}")
        tests_failed += 1

    # Test 12: Allow same-pod aggregation shortcut via loop erasure
    print("\n[Test 12] Allow same-pod aggregation shortcut via loop erasure")
    graph12 = _graph([
        ('h1', 't1'), ('t1', 'a1'), ('a1', 'c1'), ('c1', 'a2'),
        ('a2', 't2'), ('t2', 'h4'), ('a1', 't2'),
    ])
    path12 = ['h1', 't1', 'a1', 'c1', 'a2', 't2', 'h4']
    planned, msg = _plan_safe_reroute_path(path12, 'a2', 'a1', graph12)
    expected12 = ['h1', 't1', 'a1', 't2', 'h4']
    if planned == expected12:
        print("PASS: Repeated raw route collapsed to safe simple path")
        tests_passed += 1
    else:
        print(f"FAIL: Same-pod shortcut rejected - {msg}, planned={planned}")
        tests_failed += 1

    # Test 13: Reject loop-erased shortcut if remaining edge is missing
    print("\n[Test 13] Reject invalid loop-erased shortcut")
    graph13 = _graph([
        ('h1', 't1'), ('t1', 'a1'), ('a1', 'c1'), ('c1', 'a2'),
        ('a2', 't2'), ('t2', 'h4'),
    ])
    planned, msg = _plan_safe_reroute_path(path12, 'a2', 'a1', graph13)
    if planned is None and ("lacks link" in msg or "no safe detour" in msg):
        print("PASS: Shortcut rejected when collapsed path has no physical link")
        tests_passed += 1
    else:
        print(f"FAIL: Invalid shortcut accepted - {planned}")
        tests_failed += 1

    # Test 14: Reject detour that produces a repeated-node path
    print("\n[Test 14] Reject unsafe repeated-node detour")
    graph14 = _graph([
        ('h1', 'x'), ('x', 'w'), ('w', 'y'), ('y', 'h2'),
        ('h1', 'a'), ('a', 'b'), ('b', 'alt'), ('alt', 'b'), ('b', 'h2'),
    ])
    path14 = ['h1', 'x', 'w', 'y', 'h2']
    planned, msg = _plan_safe_reroute_path(path14, 'w', 'alt', graph14)
    if planned is None and "unsafe" in msg:
        print("PASS: Repeated-node detour rejected")
        tests_passed += 1
    else:
        print(f"FAIL: Unsafe repeated-node detour accepted - {planned}")
        tests_failed += 1

    # Test 15: Reject detour using a host as transit
    print("\n[Test 15] Reject host-transit detour")
    graph15 = _graph([
        ('h1', 'x'), ('x', 'w'), ('w', 'y'), ('y', 'h2'),
        ('h1', 'h100'), ('h100', 'alt'), ('alt', 'h2'),
    ])
    path15 = ['h1', 'x', 'w', 'y', 'h2']
    planned, msg = _plan_safe_reroute_path(path15, 'w', 'alt', graph15)
    if planned is None and "non-switch transit" in msg:
        print("PASS: Host-transit detour rejected")
        tests_passed += 1
    else:
        print(f"FAIL: Host-transit detour accepted - {planned}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/15")
    print(f"Tests failed: {tests_failed}/15")

    if tests_failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
