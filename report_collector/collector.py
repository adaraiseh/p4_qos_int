# collector.py

import sys
import io
import time
import threading

from scapy.all import Packet
from scapy.all import BitField, ShortField
from scapy.layers.inet import Ether, IP, TCP, UDP, bind_layers
from influxdb_client import Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.write.point import WritePrecision


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


class FlowInfo():
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
        print("src_ip %s" % (self.src_ip))
        print("dst_ip %s" % (self.dst_ip))
        print("src_port %s" % (self.src_port))
        print("dst_port %s" % (self.dst_port))
        print("ip_proto %s" % (self.ip_proto))
        print("hop_cnt %s" % (self.hop_cnt))
        print("flow_latency %s" % (self.flow_latency))
        if len(self.switch_ids) > 0:
            print("switch_ids %s" % (self.switch_ids))
        if len(self.l1_ingress_ports) > 0:
            print("l1_ingress_ports %s" % (self.l1_ingress_ports))
            print("l1_egress_ports %s" % (self.l1_egress_ports))
        if len(self.hop_latencies) > 0:
            print("hop_latencies %s" % (self.hop_latencies))
        if len(self.queue_ids) > 0:
            print("queue_ids %s" % (self.queue_ids))
            print("queue_occups %s" % (self.queue_occups))
        if len(self.ingress_tstamps) > 0:
            print("ingress_tstamps %s" % (self.ingress_tstamps))
            print("egress_tstamps %s" % (self.egress_tstamps))
        if len(self.l2_ingress_ports) > 0:
            print("l2_ingress_ports %s" % (self.l2_ingress_ports))
            print("l2_egress_ports %s" % (self.l2_egress_ports))
        if len(self.egress_tx_utils) > 0:
            print("egress_tx_utils %s" % (self.egress_tx_utils))
        print("\n")

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
    - use_device_time=False: use server now() to avoid device clock skew.
    - aggregate_enabled=True caps each series at <= 10 pts/sec via 100ms averaging.
    """
    def __init__(self, influx_client, org, bucket,
                 write_async=True, flush_interval_ms=500, batch_size=1000,
                 use_device_time=False,
                 aggregate_enabled=True,        # NEW: knob to enable/disable averaging
                 aggregate_window_ms=500):      # NEW: 500ms -> <=20 pts/sec
        self.influx_client = influx_client
        self.counter = 0            # packets parsed
        self.records_exported = 0   # points written to Influx
        self._lock = threading.Lock()
        self._last_log = time.time()

        # Aggregation controls/state
        self.aggregate_enabled = bool(aggregate_enabled)
        self.bucket_ns = int(max(1, int(aggregate_window_ms)) * 1_000_000)  # ms -> ns
        # key=(measurement, sorted(tags)) -> state dict
        self._agg = {}

        if write_async:
            self.write_api = influx_client.write_api(write_options=WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval_ms,
                jitter_interval=0,
                retry_interval=1000,
                max_retries=3,
                max_retry_delay=5000,
                exponential_base=2
            ))
        else:
            self.write_api = influx_client.write_api(write_options=SYNCHRONOUS)

        self.use_device_time = bool(use_device_time)
        self.org = org
        self.bucket = bucket
        # (flow_id, switch_id, queue_id, egress_port) -> (last_count, last_ts_ns)
        self.last_drop_data = {}

    def flush_buffer(self):
        try:
            self.write_api.flush()
        except Exception:
            pass

    # ---------- Logging ----------
    def log_export_rate(self):
        now = time.time()
        with self._lock:
            if now - self._last_log >= 1.0:   # once per second
                print(f"[INFO] Exported {self.records_exported} records in the last second")
                sys.stdout.flush()
                self.records_exported = 0
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
    def _agg_key(self, measurement: str, tags: dict):
        return (measurement, tuple(sorted(tags.items())))

    def _emit_point(self, measurement: str, tags: dict, avg_value: float, ts_ns: int):
        p = Point(measurement)
        for k, v in tags.items():
            p = p.tag(k, v)
        return p.field("value", float(avg_value)).time(int(ts_ns))

    def _emit_or_aggregate(self, measurement: str, tags: dict, value: float, timestamp_ns: int, out_points: list):
        """
        Either append the raw point, or aggregate into 100ms bucket to emit 1 averaged point per bucket.
        """
        if not self.aggregate_enabled:
            out_points.append(self._emit_point(measurement, tags, value, timestamp_ns))
            return

        key = self._agg_key(measurement, tags)
        bucket = int(timestamp_ns // self.bucket_ns)
        state = self._agg.get(key)

        if state is None:
            self._agg[key] = {
                "bucket": bucket,
                "sum": float(value),
                "count": 1,
                "tags": tags,
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
        out_points.append(self._emit_point(state["measurement"], state["tags"], avg, ts_emit))

        # start new bucket
        state["bucket"] = bucket
        state["sum"] = float(value)
        state["count"] = 1

    def _flush_agg_due(self, now_ns: int, out_points: list):
        """
        Flush buckets older than the current bucket to prevent points from getting stuck.
        """
        if not self.aggregate_enabled or not self._agg:
            return
        current_bucket = now_ns // self.bucket_ns
        to_delete = []
        for key, st in self._agg.items():
            if st["bucket"] < current_bucket:
                avg = st["sum"] / max(1, st["count"])
                ts_emit = (st["bucket"] + 1) * self.bucket_ns
                out_points.append(self._emit_point(st["measurement"], st["tags"], avg, ts_emit))
                to_delete.append(key)
        for key in to_delete:
            del self._agg[key]

    # ---------- Export ----------
    def export_influxdb(self, flow_info):
        if not flow_info:
            return
        try:
            points = []
            flow_id = (flow_info.dst_port // 10) % 100  # digit 2 & 3
            expected_queue_id = flow_info.dst_port % 10  # digit 4

            # ---- Robust guard for partial/empty hop metadata ----
            arrays = [
                flow_info.switch_ids,
                flow_info.l1_ingress_ports,
                flow_info.l1_egress_ports,
                flow_info.hop_latencies,
                flow_info.queue_ids,
                flow_info.queue_occups,
                flow_info.queue_drops,
                flow_info.ingress_tstamps,
                flow_info.egress_tstamps,
                flow_info.egress_tx_utils,
            ]
            present_lengths = [len(a) for a in arrays]
            safe_hops = min([flow_info.hop_cnt] + present_lengths) if flow_info.hop_cnt else min(present_lengths + [0])

            if safe_hops <= 0:
                return

            # Choose a unified timestamp in ns
            if self.use_device_time and len(flow_info.egress_tstamps) >= safe_hops:
                report_time = int(flow_info.egress_tstamps[safe_hops - 1])
            elif self.use_device_time and len(flow_info.ingress_tstamps) >= safe_hops:
                report_time = int(flow_info.ingress_tstamps[safe_hops - 1])
            else:
                report_time = int(time.time_ns())

            # Per-hop metrics
            for i in range(safe_hops):
                # switch_latency (us -> ms)
                self._emit_or_aggregate(
                    "switch_latency",
                    {
                        "flow_id": flow_id,
                        "src_ip": flow_info.src_ip,
                        "dst_ip": flow_info.dst_ip,
                        "queue_id": flow_info.queue_ids[i],
                        "switch_id": flow_info.switch_ids[i],
                    },
                    float(flow_info.hop_latencies[i] / 1000.0),
                    report_time,
                    points,
                )

                # tx_utilization
                self._emit_or_aggregate(
                    "tx_utilization",
                    {
                        "flow_id": flow_id,
                        "src_ip": flow_info.src_ip,
                        "dst_ip": flow_info.dst_ip,
                        "switch_id": flow_info.switch_ids[i],
                        "egress_port": flow_info.l1_egress_ports[i],
                        "queue_id": flow_info.queue_ids[i],
                    },
                    float(flow_info.egress_tx_utils[i]),
                    report_time,
                    points,
                )

                # queue_occupancy
                self._emit_or_aggregate(
                    "queue_occupancy",
                    {
                        "flow_id": flow_id,
                        "src_ip": flow_info.src_ip,
                        "dst_ip": flow_info.dst_ip,
                        "switch_id": flow_info.switch_ids[i],
                        "queue_id": flow_info.queue_ids[i],
                    },
                    float(flow_info.queue_occups[i]),
                    report_time,
                    points,
                )

                # drop-rate (structured)
                dr = self.record_drop_rate_instant(
                    flow_id,
                    flow_info.src_ip,
                    flow_info.dst_ip,
                    flow_info.switch_ids[i],
                    flow_info.l1_egress_ports[i],
                    flow_info.queue_ids[i],
                    flow_info.queue_drops[i],
                    report_time,
                )
                if dr is not None:
                    self._emit_or_aggregate(
                        dr["measurement"], dr["tags"], float(dr["value"]), dr["ts_ns"], points
                    )

            # Link latency (device stamps), same time domain
            link_pairs = max(safe_hops - 1, 0)
            for i in range(link_pairs):
                if i + 1 < len(flow_info.egress_tstamps) and i < len(flow_info.ingress_tstamps):
                    link_latency = abs(
                        flow_info.egress_tstamps[i + 1] - flow_info.ingress_tstamps[i]
                    ) / 1_000_000.0

                    self._emit_or_aggregate(
                        "link_latency",
                        {
                            "flow_id": flow_id,
                            "src_ip": flow_info.src_ip,
                            "dst_ip": flow_info.dst_ip,
                            "queue_id": expected_queue_id,
                            "egress_switch_id": flow_info.switch_ids[i + 1],
                            "egress_port_id": flow_info.l1_egress_ports[i + 1],
                            "ingress_switch_id": flow_info.switch_ids[i],
                            "ingress_port_id": flow_info.l1_ingress_ports[i],
                        },
                        float(link_latency),
                        report_time,
                        points,
                    )

            # Flow latency
            if len(flow_info.ingress_tstamps) >= 1 and len(flow_info.egress_tstamps) >= safe_hops:
                flow_latency = (
                    flow_info.ingress_tstamps[0] - flow_info.egress_tstamps[safe_hops - 1]
                ) / 1_000_000.0

                self._emit_or_aggregate(
                    "flow_latency",
                    {
                        "flow_id": flow_id,
                        "src_ip": flow_info.src_ip,
                        "dst_ip": flow_info.dst_ip,
                        "queue_id": expected_queue_id,
                    },
                    float(flow_latency),
                    report_time,
                    points,
                )

            # Flush any buckets that are due (older than current bucket)
            self._flush_agg_due(report_time, points)

            # Write what we have (may be empty if everything stayed in current bucket)
            if points:
                self.write_api.write(
                    bucket=self.bucket,
                    org=self.org,
                    record=points,
                    write_precision=WritePrecision.NS,
                )
                with self._lock:
                    self.records_exported += len(points)

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

    def parse_int_metadata(self, flow_info, int_pkt):
        if INTShim not in int_pkt:
            return

        ins_map = (int_pkt[INTMD].instruction_mask_0003 << 4) + int_pkt[INTMD].instruction_mask_0407
        int_len = int_pkt.int_length - 3
        hop_meta_len_bytes = int_pkt[INTMD].HopMetaLength << 2
        int_metadata = int_pkt.load[:int_len << 2]
        hop_count = int(int_len / (hop_meta_len_bytes >> 2))
        flow_info.hop_cnt = hop_count

        for i in range(hop_count):
            index = i * hop_meta_len_bytes
            hop_metadata = int_metadata[index:index + hop_meta_len_bytes]
            offset = 0

            if ins_map & SWITCH_ID_BIT:
                flow_info.switch_ids.append(int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big'))
                offset += 4
            if ins_map & L1_PORT_IDS_BIT:
                flow_info.l1_ingress_ports.append(int.from_bytes(hop_metadata[offset:offset + 2], byteorder='big'))
                offset += 2
                flow_info.l1_egress_ports.append(int.from_bytes(hop_metadata[offset:offset + 2], byteorder='big'))
                offset += 2
            if ins_map & HOP_LATENCY_BIT:
                flow_info.hop_latencies.append(int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big'))
                offset += 4
            if ins_map & QUEUE_BIT:
                flow_info.queue_ids.append(int.from_bytes(hop_metadata[offset:offset + 1], byteorder='big'))
                offset += 1
                flow_info.queue_occups.append(int.from_bytes(hop_metadata[offset:offset + 3], byteorder='big'))
                offset += 3
                flow_info.queue_drops.append(int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big'))
                offset += 4
            if ins_map & INGRESS_TSTAMP_BIT:
                flow_info.ingress_tstamps.append(int.from_bytes(hop_metadata[offset:offset + 8], byteorder='big') * 1000)
                offset += 8
            if ins_map & EGRESS_TSTAMP_BIT:
                flow_info.egress_tstamps.append(int.from_bytes(hop_metadata[offset:offset + 8], byteorder='big') * 1000)
                offset += 8
            if ins_map & L2_PORT_IDS_BIT:
                flow_info.l2_ingress_ports.append(int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big'))
                offset += 4
                flow_info.l2_egress_ports.append(int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big'))
                offset += 4
            if ins_map & EGRESS_PORT_TX_UTIL_BIT:
                tx_util = int.from_bytes(hop_metadata[offset:offset + 4], byteorder='big')
                tx_util_normalized = round(tx_util / 10**4, 2)
                flow_info.egress_tx_utils.append(tx_util_normalized)

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
        self.log_export_rate()
        sys.stdout.flush()
        return flow_info
