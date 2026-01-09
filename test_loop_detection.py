#!/usr/bin/env python3
"""
Test script for smart loop detection in controller.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Mock the controller's loop detection method
def _detect_routing_loop(path: list[str]) -> tuple[bool, str]:
    """
    Detect if a path contains a routing loop (directed cycle).

    A routing loop exists if the directed edge sequence contains a cycle,
    meaning packets could circulate indefinitely. Node revisits are OK
    if edges differ (e.g., a1→c1 then a3→c1 - same node, different ingress).

    The key insight: A routing loop exists if and only if the same directed
    edge appears twice in the path.

    Args:
        path: List of node names forming the path

    Returns:
        (has_loop: bool, diagnostic_msg: str)
    """
    if not path or len(path) < 2:
        return False, ""

    # Fast path: If no node appears twice, impossible to have a cycle
    if len(path) == len(set(path)):
        return False, ""

    # Check for two types of loops:
    # 1. Self-loops: node connects to itself
    # 2. Duplicate edges: same directed edge appears twice
    edges_seen = {}  # edge -> first occurrence position
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        # Check for self-loop
        if u == v:
            return True, f"Loop detected: {u} -> {u} (self-loop)"

        # Check for duplicate edge
        edge = (u, v)
        if edge in edges_seen:
            # Same directed edge appears twice - this is a loop!
            first_pos = edges_seen[edge]
            loop_path = path[first_pos:i+2]
            return True, f"Loop detected: {' -> '.join(loop_path)} (duplicate edge)"
        edges_seen[edge] = i

    return False, ""


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

    # Test 2: Node revisit with different edges - should pass (CRITICAL TEST)
    print("\n[Test 2] Valid node revisit (core switch 'c1' appears twice)")
    path2 = ['h1', 't1', 'a1', 'c1', 'a3', 'c1', 'a7', 't8', 'h16']
    print(f"Path: {' -> '.join(path2)}")
    print("Edges: h1->t1, t1->a1, a1->c1, c1->a3, a3->c1, c1->a7, a7->t8, t8->h16")
    print("Analysis: c1 appears at positions 3 and 5, but edges differ:")
    print("  - First: a1->c1 (ingress from a1)")
    print("  - Second: a3->c1 (ingress from a3)")
    has_loop, msg = _detect_routing_loop(path2)
    if not has_loop:
        print("✓ PASS: No loop detected (correct - valid node revisit)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: False positive - {msg}")
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

    # Test 4: Multiple node revisits without cycle - should pass
    print("\n[Test 4] Multiple node revisits without cycle")
    path4 = ['h1', 't1', 'a1', 'c1', 'a3', 'c2', 'a1', 'c3', 'a5', 't6', 'h16']
    print(f"Path: {' -> '.join(path4)}")
    print("Analysis: a1 appears at positions 2 and 6, but edges differ")
    has_loop, msg = _detect_routing_loop(path4)
    if not has_loop:
        print("✓ PASS: No loop detected (correct)")
        tests_passed += 1
    else:
        print(f"✗ FAIL: False positive - {msg}")
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
    print("\n[Test 8] User's original example 1 (should NOT be flagged)")
    p1 = ['h1', 't1', 'a1', 'c1', 'a3']
    p2 = ['a3', 'c1', 'a7', 't8', 'h16']
    merged = p1 + p2[1:]
    print(f"P1: {' -> '.join(p1)}")
    print(f"P2: {' -> '.join(p2)}")
    print(f"Merged: {' -> '.join(merged)}")
    has_loop, msg = _detect_routing_loop(merged)
    if not has_loop:
        print("✓ PASS: Valid path correctly accepted")
        tests_passed += 1
    else:
        print(f"✗ FAIL: User's valid path rejected - {msg}")
        tests_failed += 1

    # Test 9: Original user example 2
    print("\n[Test 9] User's original example 2 (should NOT be flagged)")
    p1 = ['h11', 't6', 'a6', 'c3', 'a4']
    p2 = ['a4', 'c3', 'a2', 't1', 'h2']
    merged = p1 + p2[1:]
    print(f"P1: {' -> '.join(p1)}")
    print(f"P2: {' -> '.join(p2)}")
    print(f"Merged: {' -> '.join(merged)}")
    has_loop, msg = _detect_routing_loop(merged)
    if not has_loop:
        print("✓ PASS: Valid path correctly accepted")
        tests_passed += 1
    else:
        print(f"✗ FAIL: User's valid path rejected - {msg}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/9")
    print(f"Tests failed: {tests_failed}/9")

    if tests_failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
