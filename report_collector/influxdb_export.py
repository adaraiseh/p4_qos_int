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
import logging
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import setup_unified_logging

from scapy.all import AsyncSniffer, conf
from influxdb_client import InfluxDBClient
from collector import *
from local_telemetry_cache import (
    DEFAULT_SOCKET_PATH,
    DEFAULT_TRAINING_STATE_PATH,
    LineProtocolSpoolWriter,
    LocalTelemetryCache,
    LocalTelemetryCacheServer,
)

log = logging.getLogger(__name__)

INFLUX_URL = "http://192.168.56.1:8086"
INFLUX_TOKEN = os.environ.get('INFLUX_TOKEN')
INFLUX_ORG = "Research"
INFLUX_BUCKET = "INT"
DEFAULT_EXTERNAL_ARTIFACT_ROOT = "/media/sf_amjad/p4_qos_int/training_runs"
DEFAULT_COLLECTOR_SPOOL_SPLIT_STEPS = 5000

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
            log.info(f"Discovered {len(interfaces)} collector interfaces from topology")
            return interfaces
        else:
            log.warning("No collector interfaces found in topology, using defaults")
            return DEFAULT_INTERFACES

    except Exception as e:
        log.error(f"Error loading topology: {e}")
        log.warning("Falling back to default interfaces")
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
    parser.add_argument(
        '--influx-write',
        choices=['on', 'off'],
        default='on',
        help='Write collector telemetry to InfluxDB (default: on)'
    )
    parser.add_argument(
        '--telemetry-cache-socket',
        default=DEFAULT_SOCKET_PATH,
        help='Unix socket path for the local RL telemetry cache'
    )
    parser.add_argument(
        '--disable-telemetry-cache',
        action='store_true',
        help='Disable the local in-memory telemetry cache'
    )
    parser.add_argument(
        '--telemetry-cache-retention-seconds',
        type=float,
        default=20.0,
        help='Seconds of recent collector records to keep in local memory'
    )
    parser.add_argument(
        '--telemetry-cache-max-points',
        type=int,
        default=120000,
        help='Maximum cached points per measurement'
    )
    parser.add_argument(
        '--local-spool',
        choices=['on', 'off'],
        default='on',
        help='Persist exact collector line-protocol telemetry locally (default: on)'
    )
    parser.add_argument(
        '--local-spool-dir',
        default='training_files/collector_spool',
        help='Fallback directory for local collector telemetry line-protocol artifacts'
    )
    parser.add_argument(
        '--artifact-root',
        default=DEFAULT_EXTERNAL_ARTIFACT_ROOT,
        help='External media root containing training run folders'
    )
    parser.add_argument(
        '--training-state-file',
        default=DEFAULT_TRAINING_STATE_PATH,
        help='Training state JSON file used to organize collector spool splits'
    )
    parser.add_argument(
        '--local-spool-split-steps',
        type=int,
        default=DEFAULT_COLLECTOR_SPOOL_SPLIT_STEPS,
        help='Rotate collector line-protocol spool every N training steps'
    )
    parser.add_argument(
        '--local-spool-flush-interval',
        type=float,
        default=1.0,
        help='Seconds between local telemetry spool flushes'
    )
    parser.add_argument(
        '--local-spool-queue-batches',
        type=int,
        default=8192,
        help='Buffered collector batches before the local spool applies backpressure'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='debug',
        choices=['debug', 'info', 'warning', 'error'],
        help='File log level (console always shows INFO)'
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        help='Optional directory for collector process logs'
    )

    args = parser.parse_args()

    # Set up unified logging
    setup_unified_logging(
        module_name="collector",
        log_level=args.log_level,
        log_dir=args.log_dir,
    )

    # Suppress Scapy's noisy internal loggers (Rx timeout spam every 50ms)
    logging.getLogger("Rx").setLevel(logging.WARNING)
    logging.getLogger("scapy.runtime").setLevel(logging.WARNING)

    # Determine interfaces to sniff
    if args.interfaces:
        # Explicit interface list overrides everything
        iface = [i.strip() for i in args.interfaces.split(',')]
        log.info(f"Using explicit interfaces: {iface}")
    elif args.config:
        # Load from topology configuration
        iface = get_interfaces_from_config(args.config)
    else:
        # Fall back to defaults
        iface = DEFAULT_INTERFACES
        log.info(f"Using default interfaces: {iface}")

    log.info(f"Sniffing on {iface} with BPF: {BPF}")

    influx_enabled = args.influx_write == 'on'
    if influx_enabled and not args.influx_token:
        log.error("InfluxDB token not configured. Set INFLUX_TOKEN environment variable or use --influx-token argument.")
        sys.exit(1)

    # Scapy performance knobs
    conf.use_pcap = True
    conf.sniff_promisc = 0

    influx_client = None
    if influx_enabled:
        influx_client = InfluxDBClient(
            url=args.influx_url,
            token=args.influx_token,
            org=args.influx_org
        )
    else:
        log.info("InfluxDB writes disabled; collector will serve local telemetry cache only")

    telemetry_cache = None
    telemetry_server = None
    if not args.disable_telemetry_cache:
        telemetry_cache = LocalTelemetryCache(
            retention_seconds=args.telemetry_cache_retention_seconds,
            max_points_per_measurement=args.telemetry_cache_max_points,
        )
        telemetry_server = LocalTelemetryCacheServer(
            telemetry_cache,
            args.telemetry_cache_socket,
        )
        telemetry_server.start()
    else:
        log.info("Local telemetry cache disabled")

    telemetry_spool = None
    if args.local_spool == 'on':
        telemetry_spool = LineProtocolSpoolWriter(
            output_dir=args.local_spool_dir,
            flush_interval_seconds=args.local_spool_flush_interval,
            max_queue_batches=args.local_spool_queue_batches,
            run_state_path=args.training_state_file,
            external_artifact_root=args.artifact_root,
            split_every_steps=args.local_spool_split_steps,
        )
    else:
        log.info("Local telemetry spool disabled")

    # Async writer
    c = Collector(
        influx_client, args.influx_org, args.influx_bucket,
        write_async=True, flush_interval_ms=50, batch_size=1000,
        use_device_time=False,  # P4 device timestamps are NOT Unix epoch - must use system time
        aggregate_enabled=False,
        telemetry_cache=telemetry_cache,
        telemetry_spool=telemetry_spool,
        influx_enabled=influx_enabled,
    )

    stop = False

    def signal_handler(sig, frame):
        nonlocal stop
        stop = True
        log.info("Stopping...")
        c.flush_buffer()
        if telemetry_server is not None:
            telemetry_server.stop()

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
        log.info(f"Started sniffer on {iface_name}")

    log.info(f"Total {len(sniffers)} sniffer threads running")

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
    if telemetry_spool is not None:
        telemetry_spool.close()
    if telemetry_server is not None:
        telemetry_server.stop()
    try:
        if influx_client is not None:
            influx_client.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
