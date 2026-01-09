#!/usr/bin/env python3
"""
INT Report Collector and InfluxDB Exporter.

Sniffs INT reports from leaf switch interfaces and exports metrics to InfluxDB.
Supports dynamic interface discovery from topology configuration.
"""

import sys
import os
import signal
import argparse
from pathlib import Path

from scapy.all import AsyncSniffer, conf
from influxdb_client import InfluxDBClient
from collector import *

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

INFLUX_URL = "http://192.168.56.1:8086"
INFLUX_TOKEN = os.environ.get('INFLUX_TOKEN')
INFLUX_ORG = "Research"
INFLUX_BUCKET = "INT"

# Default interfaces (legacy Fat-Tree topology with 4 ToR switches)
DEFAULT_INTERFACES = ['t1-eth10', 't2-eth10', 't3-eth10', 't4-eth10']

BPF = "udp and dst port 1234"


def get_interfaces_from_config(config_path: str) -> list:
    """
    Get INT collector interfaces from topology configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        List of interface names (e.g., ['leaf1-eth10', 'leaf2-eth10'])
    """
    try:
        from topology.factory import create_topology

        builder = create_topology(config_path)
        interfaces = builder.get_collector_interfaces()

        if interfaces:
            print(f"Discovered {len(interfaces)} collector interfaces from topology")
            return interfaces
        else:
            print("No collector interfaces found in topology, using defaults")
            return DEFAULT_INTERFACES

    except Exception as e:
        print(f"Error loading topology: {e}")
        print("Falling back to default interfaces")
        return DEFAULT_INTERFACES


def handle_pkt(pkt, c: Collector):
    if INTREP in pkt:
        fi = c.parser_int_pkt(pkt)
        if fi:
            c.export_influxdb(fi)


def main():
    parser = argparse.ArgumentParser(
        description="INT Report Collector and InfluxDB Exporter"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML topology configuration for dynamic interface discovery'
    )
    parser.add_argument(
        '--interfaces', '-i',
        type=str,
        default=None,
        help='Comma-separated list of interfaces (overrides config)'
    )
    parser.add_argument(
        '--influx-url',
        default=INFLUX_URL,
        help='InfluxDB URL'
    )
    parser.add_argument(
        '--influx-token',
        default=INFLUX_TOKEN,
        help='InfluxDB token (or set INFLUX_TOKEN env var)'
    )
    parser.add_argument(
        '--influx-org',
        default=INFLUX_ORG,
        help='InfluxDB organization'
    )
    parser.add_argument(
        '--influx-bucket',
        default=INFLUX_BUCKET,
        help='InfluxDB bucket'
    )

    args = parser.parse_args()

    # Determine interfaces to sniff
    if args.interfaces:
        # Explicit interface list overrides everything
        iface = [i.strip() for i in args.interfaces.split(',')]
        print(f"Using explicit interfaces: {iface}")
    elif args.config:
        # Load from topology configuration
        iface = get_interfaces_from_config(args.config)
    else:
        # Fall back to defaults
        iface = DEFAULT_INTERFACES
        print(f"Using default interfaces: {iface}")

    print(f"Sniffing on {iface} with BPF: {BPF}")
    sys.stdout.flush()

    if not args.influx_token:
        print("Error: InfluxDB token not configured. Set INFLUX_TOKEN environment variable or use --influx-token argument.")
        sys.exit(1)

    # Scapy performance knobs
    conf.use_pcap = True
    conf.sniff_promisc = 0

    influx_client = InfluxDBClient(
        url=args.influx_url,
        token=args.influx_token,
        org=args.influx_org
    )

    # Async writer
    c = Collector(
        influx_client, args.influx_org, args.influx_bucket,
        write_async=True, flush_interval_ms=50, batch_size=1000,
        use_device_time=False,
        aggregate_enabled=False
    )

    stop = False

    def signal_handler(sig, frame):
        nonlocal stop
        stop = True
        print("\nStopping...")
        c.flush_buffer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # CPU Optimization: Create one sniffer per interface for parallel packet parsing
    # This distributes Scapy parsing load across multiple threads instead of one
    sniffers = []
    for iface_name in iface:
        s = AsyncSniffer(
            iface=iface_name, filter=BPF, store=False,
            prn=lambda x: handle_pkt(x, c)
        )
        s.start()
        sniffers.append(s)
        print(f"Started sniffer on {iface_name}")

    print(f"Total {len(sniffers)} sniffer threads running")
    sys.stdout.flush()

    while not stop:
        try:
            signal.pause()
        except InterruptedError:
            # Normal signal interruption, continue
            continue

    # Cleanup - stop all sniffers
    for s in sniffers:
        try:
            s.stop()
        except Exception:
            pass
    c.flush_buffer()
    try:
        influx_client.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
