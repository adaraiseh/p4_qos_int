#!/usr/bin/env python3
import sys
import os
import argparse
from scapy.all import sniff

def get_if():
    """
    Returns the first interface matching 'eth'.
    """
    interfaces = [i for i in os.listdir('/sys/class/net/') if 'eth' in i]
    if not interfaces:
        print("Cannot find any 'eth' interface")
        sys.exit(1)
    return interfaces[0]

def parse_ports(val: str):
    """
    Parse a port value: either a single number or a START-END range.
    Returns a tuple (start, end).
    """
    if '-' in val:
        start, end = val.split('-', 1)
        return int(start), int(end)
    else:
        p = int(val)
        return p, p

def build_bpf(proto: str, port_range: tuple[int, int]):
    """
    Build BPF filter string based on proto and port range.
    """
    if proto == "tcp":
        proto_expr = "tcp"
    elif proto == "udp":
        proto_expr = "udp"
    else:
        proto_expr = "tcp or udp"

    start, end = port_range
    if start == end:
        return f"{proto_expr} and port {start}"
    else:
        return f"{proto_expr} and portrange {start}-{end}"

def handle_pkt(pkt):
    """
    Handles each packet (print summary).
    """
    print(pkt.summary())

def main():
    ap = argparse.ArgumentParser(description="Simple packet receiver")
    ap.add_argument("--proto", choices=["tcp", "udp", "all"], default="all",
                    help="Protocol to sniff (default: all).")
    ap.add_argument("--ports", type=str, default="5000-5999",
                    help="Port number or range (default: 5000-5999).")
    args = ap.parse_args()

    port_range = parse_ports(args.ports)
    bpf = build_bpf(args.proto, port_range)

    iface = get_if()
    print(f"Sniffing on {iface} with filter: {bpf}")
    sys.stdout.flush()

    sniff(
        iface=iface,
        filter=bpf,
        prn=handle_pkt,
        store=False
    )

if __name__ == "__main__":
    main()
