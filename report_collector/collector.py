# collector.py

import sys
import os
import io
import time
import struct
import threading
import heapq
import logging


# Add parent directory to path for logging_config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import setup_unified_logging

log = logging.getLogger(__name__)

from scapy.all import Packet
from scapy.all import BitField, ShortField
from scapy.layers.inet import Ether, IP, TCP, UDP, bind_layers
from influxdb_client import Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.write.point import WritePrecision
from reactivex.scheduler import ThreadPoolScheduler


class INTREP(Packet):
    name = "INT Report Header v2.0"
    fields_desc = [
        BitField("version", 0, 4),
        BitField("hw_id", 0, 6),
        BitField("seq_number", 0, 22),
        BitField("node_id", 0, 32)
    ]


class INTIndiviREP(Packet):
    name = "INT Report Individual Header v2.0"
    fields_desc = [
        BitField("rep_type", 0, 4),
        BitField("in_type", 0, 4),
        BitField("rep_len", 0, 8),
        BitField("md_len", 0, 8),
        BitField("flag", 0, 4),
        BitField("rsvd", 0, 4),
        ShortField("RepMdBits", 0),
        ShortField("DomainID", 0),
        ShortField("DSMdBits", 0),
        ShortField("DSMdstatus", 0)
    ]


class INTShim(Packet):
    name = "INT Shim header v2.1"
    fields_desc = [
        BitField("type", 0, 4),
        BitField("next_protocol", 0, 2),
        BitField("rsvd", 0, 2),
        BitField("int_length", 0, 8),
        ShortField("NPT_Dependent_Field", 0)
    ]


class INTMD(Packet):
    name = "INT-MD Header v2.1"
    fields_desc = [
        BitField("version", 0, 4),
        BitField("flags", 0, 3),
        BitField("reserved", 0, 12),
        BitField("HopMetaLength", 0, 5),
        BitField("RemainingHopCount", 0, 8),
        BitField("instruction_mask_0003", 0, 4),
        BitField("instruction_mask_0407", 0, 4),
        BitField("instruction_mask_0811", 0, 4),
        BitField("instruction_mask_1215", 0, 4),
        ShortField("DomainID", 0),
        ShortField("DomainInstructions", 0),
        ShortField("DomainFlags", 0)
    ]


bind_layers(UDP, INTREP, dport=1234)
bind_layers(INTREP, INTIndiviREP)
bind_layers(INTIndiviREP, Ether, in_type=3)
bind_layers(INTShim, INTMD, type=1)

SWITCH_ID_BIT =           0b10000000
L1_PORT_IDS_BIT =         0b01000000
HOP_LATENCY_BIT =         0b00100000
QUEUE_BIT =               0b00010000
INGRESS_TSTAMP_BIT =      0b00001000
EGRESS_TSTAMP_BIT =       0b00000100
L2_PORT_IDS_BIT =         0b00000010
EGRESS_PORT_TX_UTIL_BIT = 0b00000001

# Pre-compiled struct formats for faster INT metadata parsing (CPU optimization)
# Using Struct objects avoids format string compilation on each unpack call
STRUCT_UINT32 = struct.Struct('>I')      # 4-byte unsigned int (big-endian)
STRUCT_UINT16_PAIR = struct.Struct('>HH')  # Two 2-byte unsigned ints
STRUCT_UINT64 = struct.Struct('>Q')      # 8-byte unsigned int
STRUCT_BYTE = struct.Struct('>B')        # 1-byte unsigned int
STRUCT_UINT32_PAIR = struct.Struct('>II')  # Two 4-byte unsigned ints


class FlowInfo:
    """Flow metadata container with __slots__ for reduced memory/CPU overhead."""
    __slots__ = ('src_ip', 'dst_ip', 'src_port', 'dst_port', 'ip_proto',
                 'hop_cnt', 'flow_latency', 'switch_ids', 'l1_ingress_ports',
                 'l1_egress_ports', 'hop_latencies', 'queue_ids', 'queue_occups',
                 'queue_drops', 'ingress_tstamps', 'egress_tstamps',
                 'l2_ingress_ports', 'l2_egress_ports', 'egress_tx_utils',
                 'e_new_flow', 'e_flow_latency', 'e_sw_latency',
                 'e_link_latency', 'e_q_occupancy')

    def __init__(self):
        self.src_ip = None
        self.dst_ip = None
        self.src_port = None
        self.dst_port = None
        self.ip_proto = None

        self.hop_cnt = 0
        self.flow_latency = 0

        self.switch_ids = []
        self.l1_ingress_ports = []
        self.l1_egress_ports = []
        self.hop_latencies = []
        self.queue_ids = []
        self.queue_occups = []
        self.queue_drops = []
        self.ingress_tstamps = []
        self.egress_tstamps = []
        self.l2_ingress_ports = []
        self.l2_egress_ports = []
        self.egress_tx_utils = []

        self.e_new_flow = None
        self.e_flow_latency = None
        self.e_sw_latency = None
        self.e_link_latency = None
        self.e_q_occupancy = None

    def show(self):
        log.debug(f"src_ip {self.src_ip}")
        log.debug(f"dst_ip {self.dst_ip}")
        log.debug(f"src_port {self.src_port}")
        log.debug(f"dst_port {self.dst_port}")
        log.debug(f"ip_proto {self.ip_proto}")
        log.debug(f"hop_cnt {self.hop_cnt}")
        log.debug(f"flow_latency {self.flow_latency}")
        if len(self.switch_ids) > 0:
            log.debug(f"switch_ids {self.switch_ids}")
        if len(self.l1_ingress_ports) > 0:
            log.debug(f"l1_ingress_ports {self.l1_ingress_ports}")
            log.debug(f"l1_egress_ports {self.l1_egress_ports}")
        if len(self.hop_latencies) > 0:
            log.debug(f"hop_latencies {self.hop_latencies}")
        if len(self.queue_ids) > 0:
            log.debug(f"queue_ids {self.queue_ids}")
            log.debug(f"queue_occups {self.queue_occups}")
        if len(self.ingress_tstamps) > 0:
            log.debug(f"ingress_tstamps {self.ingress_tstamps}")
            log.debug(f"egress_tstamps {self.egress_tstamps}")
        if len(self.l2_ingress_ports) > 0:
            log.debug(f"l2_ingress_ports {self.l2_ingress_ports}")
            log.debug(f"l2_egress_ports {self.l2_egress_ports}")
        if len(self.egress_tx_utils) > 0:
            log.debug(f"egress_tx_utils {self.egress_tx_utils}")

    def clear_metadata(self):
        self.switch_ids.clear()
        self.l1_ingress_ports.clear()
        self.l1_egress_ports.clear()
        self.hop_latencies.clear()
        self.queue_ids.clear()
        self.queue_occups.clear()
        self.ingress_tstamps.clear()
        self.egress_tstamps.clear()
        self.l2_ingress_ports.clear()
        self.l2_egress_ports.clear()
        self.egress_tx_utils.clear()

    def __str__(self) -> str:
        return f"Flow {self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port} hops={self.hop_cnt}"


class Collector:
    """
    Per-report writes with batched async option.
    - write_async=True: ~0.5s flush cadence, much lower CPU/latency.
    - use_device_time=True: use device timestamps for accurate latency measurement.
    - aggregate_enabled=True caps each series at <= 10 pts/sec via 100ms averaging.
    """
    def __init__(self, influx_client, org, bucket,
                 write_async=True, flush_interval_ms=500, batch_size=1000,
                 use_device_time=False,  # P4 device timestamps are NOT Unix epoch - must use system time
                 aggregate_enabled=True,        # knob to enable/disable averaging
                 aggregate_window_ms=500,       # aggregation window in ms
                 # Optional metrics (disabled for CPU efficiency, enable when needed)
                 enable_link_latency=False,     # inter-switch link latency
                 enable_queue_occupancy=False,  # queue depth metrics
                 telemetry_cache=None,
                 telemetry_spool=None,
                 influx_enabled=True):
        self.influx_client = influx_client
        self.telemetry_cache = telemetry_cache
        self.telemetry_spool = telemetry_spool
        self.influx_enabled = bool(influx_enabled and influx_client is not None)
        self.counter = 0            # packets parsed
        self.records_exported = 0   # points written to Influx (total)
        self.records_per_queue = {0: 0, 1: 0, 7: 0}  # per-queue counts
        self._lock = threading.Lock()
        self._last_log = time.time()
        self._log_check_counter = 0  # rate-limit logging checks
        self._log_check_interval = 500  # only check time every N packets

        # Aggregation controls/state
        self.aggregate_enabled = bool(aggregate_enabled)
        self.bucket_ns = int(max(1, int(aggregate_window_ms)) * 1_000_000)  # ms -> ns
        # key=(measurement, sorted(tags)) -> state dict
        self._agg = {}
        self._max_agg_entries = 5000  # Limit aggregation entries to prevent memory growth

        # Optional metrics flags
        self.enable_link_latency = bool(enable_link_latency)
        self.enable_queue_occupancy = bool(enable_queue_occupancy)

        if not self.influx_enabled:
            self.write_api = None
        elif write_async:
            self.write_api = influx_client.write_api(write_options=WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval_ms,
                jitter_interval=0,
                retry_interval=1000,
                max_retries=3,
                max_retry_delay=5000,
                exponential_base=2,
                write_scheduler=ThreadPoolScheduler(max_workers=4)  # 4 threads for parallel writes
            ))
        else:
            self.write_api = influx_client.write_api(write_options=SYNCHRONOUS)

        self.use_device_time = bool(use_device_time)
        self.org = org
        self.bucket = bucket
        # (flow_id, switch_id, queue_id, egress_port) -> (last_count, last_ts_ns)
        self.last_drop_data = {}
        self._max_drop_entries = 10000  # Limit drop data entries to prevent memory growth

    def flush_buffer(self):
        if self.telemetry_spool is not None:
            try:
                self.telemetry_spool.flush()
            except Exception:
                log.warning("Local telemetry spool flush failed", exc_info=True)

        if self.write_api is None:
            return
        try:
            self.write_api.flush()
        except Exception:
            pass

    # ---------- Logging ----------
    def log_export_rate(self):
        # Rate-limit: only check time every N packets to reduce overhead
        self._log_check_counter += 1
        if self._log_check_counter < self._log_check_interval:
            return
        self._log_check_counter = 0

        now = time.time()
        # Lock-free: stats are approximate, no need for synchronization
        if now - self._last_log >= 1.0:   # once per second
            q0, q1, q7 = self.records_per_queue.get(0, 0), self.records_per_queue.get(1, 0), self.records_per_queue.get(7, 0)
            total = self.records_exported
            drop_cache = len(self.last_drop_data)
            agg_cache = len(self._agg)
            lat_cnt = getattr(self, '_lat_count', 0)
            lat_skip = getattr(self, '_lat_skip_count', 0)
            # Per-queue lat breakdown for diagnostics (shows which queue_ids flow_latency is tagged with)
            lat_per_q = getattr(self, '_lat_per_queue', {})
            # Format all queue_ids found (not just 0,1,7 - to detect mismatches)
            lat_per_q_str = " ".join(f"Q{k}:{v}" for k, v in sorted(lat_per_q.items()))
            lat_str = f"lat:{lat_cnt} ({lat_per_q_str})" if lat_per_q_str else f"lat:{lat_cnt}"
            # Per-queue skip breakdown for diagnostics
            skip_per_q = getattr(self, '_lat_skip_per_queue', {0: 0, 1: 0, 7: 0})
            skip_q0, skip_q1, skip_q7 = skip_per_q.get(0, 0), skip_per_q.get(1, 0), skip_per_q.get(7, 0)
            skip_str = f" (Q0:{skip_q0} Q1:{skip_q1} Q7:{skip_q7})" if lat_skip > 0 else ""

            sink_parts = [
                f"influx={'on' if self.write_api is not None else 'off'}",
                f"cache={'on' if self.telemetry_cache is not None else 'off'}",
                f"spool={'on' if self.telemetry_spool is not None else 'off'}",
            ]
            if self.telemetry_cache is not None:
                cache_stats = self.telemetry_cache.stats()
                sink_parts.append(f"cache_cached={cache_stats.get('records_cached', 0)}")
            if self.telemetry_spool is not None:
                spool_stats = self.telemetry_spool.stats()
                sink_parts.append(f"spool_written={spool_stats.get('records_written', 0)}")
                queued = spool_stats.get("queued_batches", 0)
                if queued:
                    sink_parts.append(f"spool_queue={queued}")
                if spool_stats.get("error"):
                    sink_parts.append("spool_error=1")

            log.info(f"[Collector] Exported {total} records (Q0:{q0} Q1:{q1} Q7:{q7}) "
                     f"| sinks: {' '.join(sink_parts)} "
                     f"| caches: drop={drop_cache} agg={agg_cache} | {lat_str} skip:{lat_skip}{skip_str}")
            self.records_exported = 0
            self.records_per_queue = {0: 0, 1: 0, 7: 0}
            self._lat_count = 0
            self._lat_skip_count = 0
            self._lat_skip_per_queue = {0: 0, 1: 0, 7: 0}
            self._lat_per_queue = {}  # Reset per-queue lat counter
            self._last_log = now

    # ---------- Drop-rate (structured return for aggregation) ----------
    def record_drop_rate_instant(self, flow_id, src_ip, dst_ip, switch_id, egress_port, queue_id,
                                 drop_count, report_time_ns):
        """
        Compute an instantaneous drop rate (per 100ms) using elapsed time
        between samples of the same series. Returns a dict suitable for aggregation:
          {"measurement": ..., "tags": {...}, "value": float, "ts_ns": int}
        or None if not enough info yet.
        """
        tag_key = (flow_id, switch_id, queue_id, egress_port)
        current_time = int(report_time_ns)

        last = self.last_drop_data.get(tag_key)
        self.last_drop_data[tag_key] = (int(drop_count), current_time)

        # Evict oldest entries if dict grows too large (memory leak prevention)
        # CPU Optimization: Use heapq.nsmallest for O(n) partial sort instead of O(n log n) full sort
        if len(self.last_drop_data) > self._max_drop_entries:
            evict_count = len(self.last_drop_data) // 10
            oldest_keys = heapq.nsmallest(evict_count, self.last_drop_data.keys(),
                                           key=lambda k: self.last_drop_data[k][1])
            for k in oldest_keys:
                del self.last_drop_data[k]

        if last is None:
            return None

        last_count, last_ts = last
        elapsed_ms = max(0.0, (current_time - last_ts) / 1_000_000.0)
        if elapsed_ms <= 0.0:
            return None

        diff = int(drop_count) - int(last_count)
        if diff < 0:
            # counter reset/wrap
            diff = 0

        per100ms = float(diff) * (100.0 / elapsed_ms)

        return {
            "measurement": "q_drop_rate_100ms",
            "tags": {
                "flow_id": flow_id,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "switch_id": switch_id,
                "egress_port": egress_port,
                "queue_id": queue_id,
            },
            "value": per100ms,
            "ts_ns": current_time
        }

    # ---------- Aggregation helpers ----------
    def _emit_point_tuple(self, measurement: str, tag_tuple: tuple, avg_value: float, ts_ns: int):
        """Fast line protocol generation from pre-sorted tag tuple."""
        # Format: measurement,tag1=val1,tag2=val2 value=X timestamp
        tag_str = ','.join(f'{k}={v}' for k, v in tag_tuple)
        return f"{measurement},{tag_str} value={avg_value} {int(ts_ns)}"

    def _emit_point(self, measurement: str, tags: dict, avg_value: float, ts_ns: int):
        # Legacy method for compatibility - uses dict, sorts once
        tag_str = ','.join(f'{k}={v}' for k, v in sorted(tags.items()))
        return f"{measurement},{tag_str} value={avg_value} {int(ts_ns)}"

    def _emit_or_aggregate_fast(self, measurement: str, tag_tuple: tuple, value: float, timestamp_ns: int, out_points: list):
        """
        Optimized version using pre-sorted tag tuples.
        Either append the raw point, or aggregate into bucket to emit 1 averaged point per bucket.
        """
        if not self.aggregate_enabled:
            out_points.append(self._emit_point_tuple(measurement, tag_tuple, value, timestamp_ns))
            return

        key = (measurement, tag_tuple)
        bucket = int(timestamp_ns // self.bucket_ns)
        state = self._agg.get(key)

        if state is None:
            self._agg[key] = {
                "bucket": bucket,
                "sum": float(value),
                "count": 1,
                "tag_tuple": tag_tuple,
                "measurement": measurement,
            }
            return

        if state["bucket"] == bucket:
            state["sum"] += float(value)
            state["count"] += 1
            return

        # bucket changed: flush previous bucket
        prev_bucket = state["bucket"]
        avg = state["sum"] / max(1, state["count"])
        ts_emit = (prev_bucket + 1) * self.bucket_ns  # end-of-bucket timestamp
        out_points.append(self._emit_point_tuple(state["measurement"], state["tag_tuple"], avg, ts_emit))

        # start new bucket
        state["bucket"] = bucket
        state["sum"] = float(value)
        state["count"] = 1

    def _emit_or_aggregate(self, measurement: str, tags: dict, value: float, timestamp_ns: int, out_points: list):
        """Legacy method for compatibility - converts dict to tuple once."""
        tag_tuple = tuple(sorted(tags.items()))
        self._emit_or_aggregate_fast(measurement, tag_tuple, value, timestamp_ns, out_points)

    def _flush_agg_due(self, now_ns: int, out_points: list):
        """
        Flush buckets older than the current bucket to prevent points from getting stuck.
        Also evicts oldest entries if _agg grows too large (memory leak prevention).
        """
        if not self.aggregate_enabled or not self._agg:
            return
        current_bucket = now_ns // self.bucket_ns
        to_delete = []
        for key, st in self._agg.items():
            if st["bucket"] < current_bucket:
                avg = st["sum"] / max(1, st["count"])
                ts_emit = (st["bucket"] + 1) * self.bucket_ns
                out_points.append(self._emit_point_tuple(st["measurement"], st["tag_tuple"], avg, ts_emit))
                to_delete.append(key)
        for key in to_delete:
            del self._agg[key]

        # Safety eviction if _agg grows too large (shouldn't happen with proper flushing)
        # CPU Optimization: Use heapq.nsmallest for O(n) partial sort instead of O(n log n) full sort
        if len(self._agg) > self._max_agg_entries:
            evict_count = len(self._agg) // 10
            oldest_keys = heapq.nsmallest(evict_count, self._agg.keys(), key=lambda k: self._agg[k]["bucket"])
            for k in oldest_keys:
                del self._agg[k]

    # ---------- Export ----------
    def export_influxdb(self, flow_info):
        if not flow_info:
            return
        
        try:
            # Reusable buffer for points
            # Pre-allocating somewhat helps avoid resizing overhead
            points = [] 
            cache_points = []
            
            # --- Extract Flow Metadata Once ---
            dst_ip = flow_info.dst_ip
            src_ip = flow_info.src_ip
            
            # Optimization: Pre-calculate integers to avoid repeated access/calc in loop
            # flow_id = (dst_port // 10) % 100
            # queue_id = dst_port % 10
            dst_port = flow_info.dst_port
            flow_id = (dst_port // 10) % 100
            expected_queue_id = dst_port % 10

            # ---- Robust guard for partial/empty hop metadata ----
            # Only check lengths of arrays we actually iterate over or index into
            # Hoisting len() calls out of the loop/list comprehension for speed
            hop_latency_len = len(flow_info.hop_latencies)
            egress_port_len = len(flow_info.l1_egress_ports)
            
            # Fast-path check: if essential arrays are empty, abort
            if hop_latency_len == 0 or egress_port_len == 0:
                return

            # Calc safe_hops based on minimum length of core per-hop data arrays
            # NOTE: Timestamps (ingress_tstamps, egress_tstamps) are NOT included here
            # because they're optional and only used for flow_latency calculation.
            # Per-hop metrics (drop_rate, tx_util, switch_latency) don't require timestamps.
            safe_hops = min(
                len(flow_info.switch_ids),
                len(flow_info.l1_ingress_ports),
                egress_port_len,
                hop_latency_len,
                len(flow_info.queue_ids),
                len(flow_info.queue_occups),
                len(flow_info.queue_drops),
                len(flow_info.egress_tx_utils),
                flow_info.hop_cnt
            )

            if safe_hops <= 0:
                return

            # Choose a unified timestamp in ns for InfluxDB record time
            # Note: use_device_time=False is strongly recommended (default) because
            # P4 device timestamps are switch uptime, not Unix epoch.
            if self.use_device_time:
                 # Prefer egress ts from first hop, fallback to ingress, fallback to now()
                 if len(flow_info.egress_tstamps) >= 1:
                     report_time = int(flow_info.egress_tstamps[-1])
                 elif len(flow_info.ingress_tstamps) >= 1:
                     report_time = int(flow_info.ingress_tstamps[-1])
                 else:
                     report_time = time.time_ns()
            else:
                report_time = time.time_ns()

            # --- OPTIMIZATION PATH: Direct String Construction ---
            # If aggregation is disabled (default in production), avoid tuple creation overhead
            if not self.aggregate_enabled:
                # CPU Optimization: Pre-format static tag prefix that's common across all measurements
                # Tag Keys sorted: dst_ip, egress_port, flow_id, queue_id, src_ip, switch_id
                # Note: InfluxDB requires tags sorted by key.
                # Common prefix for most measurements: dst_ip, flow_id (then queue_id, src_ip, switch_id vary)
                base_tags = f"dst_ip={dst_ip},flow_id={flow_id}"

                # Cache local vars for loop speed
                sw_ids = flow_info.switch_ids
                q_ids = flow_info.queue_ids
                eg_ports = flow_info.l1_egress_ports
                hop_lats = flow_info.hop_latencies
                tx_utils = flow_info.egress_tx_utils
                q_occups = flow_info.queue_occups
                q_drops = flow_info.queue_drops

                enable_occupancy = self.enable_queue_occupancy

                for i in range(safe_hops):
                    switch_id = sw_ids[i]
                    queue_id = q_ids[i]
                    egress_port = eg_ports[i]

                    # Pre-build per-hop tag suffix (CPU optimization - computed once per hop)
                    hop_tags_no_port = f"queue_id={queue_id},src_ip={src_ip},switch_id={switch_id}"
                    hop_tags_with_port = f"egress_port={egress_port},{hop_tags_no_port}"

                    # 1. switch_latency
                    # measurement=switch_latency,dst_ip=...,flow_id=...,queue_id=...,src_ip=...,switch_id=... value=... ts
                    points.append(
                        f"switch_latency,{base_tags},{hop_tags_no_port} value={hop_lats[i] / 1000.0} {report_time}"
                    )
                    cache_points.append((
                        "switch_latency",
                        {
                            "dst_ip": dst_ip,
                            "flow_id": flow_id,
                            "queue_id": queue_id,
                            "src_ip": src_ip,
                            "switch_id": switch_id,
                        },
                        hop_lats[i] / 1000.0,
                        report_time,
                    ))

                    # 2. tx_utilization
                    # measurement=tx_utilization,dst_ip=...,egress_port=...,flow_id=...,queue_id=...,src_ip=...,switch_id=... value=... ts
                    points.append(
                        f"tx_utilization,{base_tags},{hop_tags_with_port} value={tx_utils[i]} {report_time}"
                    )
                    cache_points.append((
                        "tx_utilization",
                        {
                            "dst_ip": dst_ip,
                            "egress_port": egress_port,
                            "flow_id": flow_id,
                            "queue_id": queue_id,
                            "src_ip": src_ip,
                            "switch_id": switch_id,
                        },
                        tx_utils[i],
                        report_time,
                    ))

                    # 3. queue_occupancy (Optional)
                    if enable_occupancy:
                        points.append(
                            f"queue_occupancy,{base_tags},{hop_tags_no_port} value={q_occups[i]} {report_time}"
                        )

                    # 4. drop_rate (Special logic)
                    # We still need calculations from record_drop_rate_instant, but we can avoid
                    # the dict return if we inline the logic or parse the result fast.
                    # For safety, let's keep the logic encapsulated but unpack efficiently.
                    dr = self.record_drop_rate_instant(
                        flow_id, src_ip, dst_ip, switch_id, egress_port, queue_id,
                        q_drops[i], report_time
                    )

                    if dr is not None:
                        # measurement=q_drop_rate_100ms,dst_ip=...,egress_port=...,flow_id=...,queue_id=...,src_ip=...,switch_id=... value=... ts
                        points.append(
                           f"q_drop_rate_100ms,{base_tags},{hop_tags_with_port} value={dr['value']} {dr['ts_ns']}"
                        )
                        cache_points.append((
                            dr["measurement"],
                            dr["tags"],
                            dr["value"],
                            dr["ts_ns"],
                        ))

            else:
                # --- AGGREGATION PATH (Legacy/Slow) ---
                for i in range(safe_hops):
                    switch_id = flow_info.switch_ids[i]
                    queue_id = flow_info.queue_ids[i]
                    egress_port = flow_info.l1_egress_ports[i]

                    # switch_latency tags
                    switch_latency_tags = (
                        ("dst_ip", dst_ip),
                        ("flow_id", flow_id),
                        ("queue_id", queue_id),
                        ("src_ip", src_ip),
                        ("switch_id", switch_id),
                    )
                    self._emit_or_aggregate_fast(
                        "switch_latency",
                        switch_latency_tags,
                        float(flow_info.hop_latencies[i] / 1000.0),
                        report_time,
                        points,
                    )

                    # tx_utilization tags
                    tx_util_tags = (
                        ("dst_ip", dst_ip),
                        ("egress_port", egress_port),
                        ("flow_id", flow_id),
                        ("queue_id", queue_id),
                        ("src_ip", src_ip),
                        ("switch_id", switch_id),
                    )
                    self._emit_or_aggregate_fast(
                        "tx_utilization",
                        tx_util_tags,
                        float(flow_info.egress_tx_utils[i]),
                        report_time,
                        points,
                    )

                    # queue_occupancy
                    if self.enable_queue_occupancy:
                        queue_occup_tags = (
                            ("dst_ip", dst_ip),
                            ("flow_id", flow_id),
                            ("queue_id", queue_id),
                            ("src_ip", src_ip),
                            ("switch_id", switch_id),
                        )
                        self._emit_or_aggregate_fast(
                           "queue_occupancy",
                            queue_occup_tags,
                            float(flow_info.queue_occups[i]),
                            report_time,
                            points,
                        )

                    # drop-rate
                    dr = self.record_drop_rate_instant(
                        flow_id, src_ip, dst_ip, switch_id, egress_port, queue_id,
                        flow_info.queue_drops[i], report_time,
                    )
                    if dr is not None:
                        dr_tags = tuple(sorted(dr["tags"].items()))
                        self._emit_or_aggregate_fast(
                            dr["measurement"], dr_tags, float(dr["value"]), dr["ts_ns"], points
                        )

            # --- Flow Latency (Only once per packet) ---
            # INT metadata is prepended by each switch, so:
            #   - Index 0 = last hop (most recent metadata)
            #   - Index -1 = first hop (oldest metadata)
            # Flow latency = (last hop egress time) - (first hop ingress time)
            # No dependency on safe_hops - timestamps are independent of per-hop data
            egress_ts_len = len(flow_info.egress_tstamps)
            ingress_ts_len = len(flow_info.ingress_tstamps)

            # Only need at least 1 egress and 1 ingress timestamp
            if egress_ts_len >= 1 and ingress_ts_len >= 1:
                # egress[0] = last hop egress time, ingress[-1] = first hop ingress time
                flow_latency = (
                    flow_info.egress_tstamps[0] - flow_info.ingress_tstamps[-1]
                ) / 1_000_000.0

                # Use queue_id from first hop (index -1, oldest metadata)
                int_queue_id = q_ids[-1] if len(q_ids) > 0 else expected_queue_id

                # Debug: Log when INT queue_id differs from expected (from dst_port)
                if int_queue_id != expected_queue_id:
                    log.debug(f"[Latency Queue Mismatch] expected={expected_queue_id} actual={int_queue_id} "
                              f"flow={flow_id} q_ids={list(q_ids)}")

                # Debug: log if timestamp arrays differ in length (could indicate parsing issues)
                if egress_ts_len != ingress_ts_len:
                    log.debug(f"[Latency] Timestamp array mismatch: egress={egress_ts_len}, ingress={ingress_ts_len}, "
                              f"flow={flow_id}, queue={int_queue_id}")

                # Sanity check: reject negative latency or latency > 10 seconds (10000ms)
                # This catches timestamp wraparound issues and stale packet data
                if flow_latency < 0 or flow_latency > 10000:
                    log.warning(f"[Latency] Rejected insane value: {flow_latency:.2f}ms "
                                f"(egr[0]={flow_info.egress_tstamps[0]}, "
                                f"ing[-1]={flow_info.ingress_tstamps[-1]}) "
                                f"flow={flow_id} queue={int_queue_id}")
                elif not self.aggregate_enabled:
                     points.append(
                        f"flow_latency,dst_ip={dst_ip},flow_id={flow_id},queue_id={int_queue_id},src_ip={src_ip} value={float(flow_latency)} {report_time}"
                     )
                     cache_points.append((
                        "flow_latency",
                        {
                            "dst_ip": dst_ip,
                            "flow_id": flow_id,
                            "queue_id": int_queue_id,
                            "src_ip": src_ip,
                        },
                        float(flow_latency),
                        report_time,
                     ))
                     self._lat_count = getattr(self, '_lat_count', 0) + 1
                     # Track per-queue flow_latency distribution for diagnostics
                     if not hasattr(self, '_lat_per_queue'):
                         self._lat_per_queue = {}
                     self._lat_per_queue[int_queue_id] = self._lat_per_queue.get(int_queue_id, 0) + 1
                else:
                     self._emit_or_aggregate(
                        "flow_latency",
                        {
                            "flow_id": flow_id,
                            "src_ip": flow_info.src_ip,
                            "dst_ip": flow_info.dst_ip,
                            "queue_id": int_queue_id,
                        },
                        float(flow_latency),
                        report_time,
                        points,
                    )
                     # Track per-queue flow_latency distribution for diagnostics (aggregation path)
                     if not hasattr(self, '_lat_per_queue'):
                         self._lat_per_queue = {}
                     self._lat_per_queue[int_queue_id] = self._lat_per_queue.get(int_queue_id, 0) + 1
            else:
                # Count skipped latency records
                self._lat_skip_count = getattr(self, '_lat_skip_count', 0) + 1

                # Track skips per queue for diagnostics
                if not hasattr(self, '_lat_skip_per_queue'):
                    self._lat_skip_per_queue = {0: 0, 1: 0, 7: 0}
                self._lat_skip_per_queue[expected_queue_id] = self._lat_skip_per_queue.get(expected_queue_id, 0) + 1

                # Log detailed info for EVERY skip at debug level
                log.debug(f"[Latency Skip] queue={expected_queue_id} flow={flow_id} "
                          f"egress_ts={egress_ts_len} ingress_ts={ingress_ts_len} "
                          f"hop_cnt={flow_info.hop_cnt}")

                # Also log first occurrence at warning level (existing behavior)
                if not hasattr(self, '_lat_skip_logged'):
                    self._lat_skip_logged = set()
                skip_key = (egress_ts_len, ingress_ts_len)
                if skip_key not in self._lat_skip_logged:
                    self._lat_skip_logged.add(skip_key)
                    log.warning(f"[Latency Skip] FIRST: egress_ts={egress_ts_len}, ingress_ts={ingress_ts_len}, "
                               f"hop_cnt={flow_info.hop_cnt}, queue={expected_queue_id}")

            # Flush aggregation buckets if enabled
            self._flush_agg_due(report_time, points)

            # Write batch
            if points:
                if self.telemetry_cache is not None:
                    if cache_points:
                        self.telemetry_cache.add_preparsed_points(cache_points)
                    else:
                        self.telemetry_cache.add_line_points(points)
                if self.telemetry_spool is not None:
                    self.telemetry_spool.write_lines(points)

                # Use low-level write call if possible to avoid Point object validation overhead?
                # The generic client.write_api.write() handles strings well.
                if self.write_api is not None:
                    self.write_api.write(
                        bucket=self.bucket,
                        org=self.org,
                        record=points,
                        write_precision=WritePrecision.NS,
                    )
                self.records_exported += len(points)
                if expected_queue_id in self.records_per_queue:
                    self.records_per_queue[expected_queue_id] += len(points)

        finally:
            flow_info.clear_metadata()
            self.log_export_rate()

    # ---------- Parsing ----------
    def parse_flow_info(self, flow_info, ip_pkt):
        flow_info.src_ip = ip_pkt.src
        flow_info.dst_ip = ip_pkt.dst
        flow_info.ip_proto = ip_pkt.proto

        if UDP in ip_pkt:
            flow_info.src_port = ip_pkt[UDP].sport
            flow_info.dst_port = ip_pkt[UDP].dport
        elif TCP in ip_pkt:
            flow_info.src_port = ip_pkt[TCP].sport
            flow_info.dst_port = ip_pkt[TCP].dport

    # Debug counter for instruction mask logging (avoid flooding)
    _ins_map_logged = set()

    def parse_int_metadata(self, flow_info, int_pkt):
        if INTShim not in int_pkt:
            return

        ins_map = (int_pkt[INTMD].instruction_mask_0003 << 4) + int_pkt[INTMD].instruction_mask_0407
        int_len = int_pkt.int_length - 3
        hop_meta_len_bytes = int_pkt[INTMD].HopMetaLength << 2
        int_metadata = int_pkt.load[:int_len << 2]
        hop_count = int(int_len / (hop_meta_len_bytes >> 2))
        flow_info.hop_cnt = hop_count

        # Pre-compute instruction presence flags (avoid repeated bitwise AND in loop)
        has_switch_id = bool(ins_map & SWITCH_ID_BIT)
        has_l1_ports = bool(ins_map & L1_PORT_IDS_BIT)
        has_hop_latency = bool(ins_map & HOP_LATENCY_BIT)
        has_queue = bool(ins_map & QUEUE_BIT)
        has_ingress_ts = bool(ins_map & INGRESS_TSTAMP_BIT)
        has_egress_ts = bool(ins_map & EGRESS_TSTAMP_BIT)
        has_l2_ports = bool(ins_map & L2_PORT_IDS_BIT)
        has_tx_util = bool(ins_map & EGRESS_PORT_TX_UTIL_BIT)

        # Debug: Log instruction mask once per unique value (to understand what's being received)
        if ins_map not in Collector._ins_map_logged:
            Collector._ins_map_logged.add(ins_map)
            log.info(f"[INT Debug] ins_map=0x{ins_map:02X}, hops={hop_count}, hop_meta_len={hop_meta_len_bytes}B, "
                     f"sw={has_switch_id}, l1={has_l1_ports}, lat={has_hop_latency}, q={has_queue}, "
                     f"ing_ts={has_ingress_ts}, eg_ts={has_egress_ts}, l2={has_l2_ports}, tx={has_tx_util}")

        # Local references for faster access
        switch_ids = flow_info.switch_ids
        l1_ingress_ports = flow_info.l1_ingress_ports
        l1_egress_ports = flow_info.l1_egress_ports
        hop_latencies = flow_info.hop_latencies
        queue_ids = flow_info.queue_ids
        queue_occups = flow_info.queue_occups
        queue_drops = flow_info.queue_drops
        ingress_tstamps = flow_info.ingress_tstamps
        egress_tstamps = flow_info.egress_tstamps
        l2_ingress_ports = flow_info.l2_ingress_ports
        l2_egress_ports = flow_info.l2_egress_ports
        egress_tx_utils = flow_info.egress_tx_utils

        # Use pre-compiled struct objects for faster parsing (CPU optimization)
        for i in range(hop_count):
            offset = i * hop_meta_len_bytes

            if has_switch_id:
                switch_ids.append(STRUCT_UINT32.unpack_from(int_metadata, offset)[0])
                offset += 4
            if has_l1_ports:
                in_port, eg_port = STRUCT_UINT16_PAIR.unpack_from(int_metadata, offset)
                l1_ingress_ports.append(in_port)
                l1_egress_ports.append(eg_port)
                offset += 4
            if has_hop_latency:
                hop_latencies.append(STRUCT_UINT32.unpack_from(int_metadata, offset)[0])
                offset += 4
            if has_queue:
                # queue_id: 1 byte, queue_occup: 3 bytes (24-bit), queue_drops: 4 bytes
                q_id = STRUCT_BYTE.unpack_from(int_metadata, offset)[0]
                queue_ids.append(q_id)
                offset += 1
                # 3-byte value: unpack as 4 bytes with leading zero
                q_occ = STRUCT_UINT32.unpack_from(b'\x00' + int_metadata[offset:offset + 3], 0)[0]
                queue_occups.append(q_occ)
                offset += 3
                queue_drops.append(STRUCT_UINT32.unpack_from(int_metadata, offset)[0])
                offset += 4
            if has_ingress_ts:
                ingress_tstamps.append(STRUCT_UINT64.unpack_from(int_metadata, offset)[0] * 1000)
                offset += 8
            if has_egress_ts:
                egress_tstamps.append(STRUCT_UINT64.unpack_from(int_metadata, offset)[0] * 1000)
                offset += 8
            if has_l2_ports:
                l2_in, l2_eg = STRUCT_UINT32_PAIR.unpack_from(int_metadata, offset)
                l2_ingress_ports.append(l2_in)
                l2_egress_ports.append(l2_eg)
                offset += 8
            if has_tx_util:
                tx_util = STRUCT_UINT32.unpack_from(int_metadata, offset)[0]
                egress_tx_utils.append(round(tx_util / 10**4, 2))

    def parser_int_pkt(self, pkt):
        if INTREP not in pkt:
            return
        int_rep_pkt = pkt[INTREP]
        flow_info = FlowInfo()
        self.parse_flow_info(flow_info, int_rep_pkt[IP])
        int_shim_pkt = INTShim(int_rep_pkt.load)
        self.parse_int_metadata(flow_info, int_shim_pkt)

        # Count parsed packets (FYI)
        self.counter += 1
        return flow_info
