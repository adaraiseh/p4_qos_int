# collector.py

import sys
import io
import time

from scapy.all import Packet
from scapy.all import BitField, ShortField
from scapy.layers.inet import Ether, IP, TCP, UDP, bind_layers
from influxdb_client import Point
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
        pass


class Collector:
    """
    Per-report atomic writes using the sink-hop egress timestamp as a unified time.
    Emits a drop-rate point on every report (except the first per tag).
    """
    def __init__(self, influx_client, org, bucket):
        self.influx_client = influx_client
        self.write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        self.org = org
        self.bucket = bucket
        # (flow_id, switch_id, queue_id, egress_port) -> (last_count, last_ts_ns)
        self.last_drop_data = {}

    # Kept for compatibility with influx_export.py
    def flush_buffer(self):
        return

    def record_drop_rate_instant(self, flow_id, src_ip, dst_ip, switch_id, egress_port, queue_id,
                                 drop_count, report_time_ns):
        """
        Emit a per-100ms instantaneous drop rate for every report.
        - Uses elapsed time between consecutive samples for this tag.
        - If the counter resets (wrap/restart), treats negative diffs as 0.
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

        # Scale to "per 100 ms"
        per100ms = float(diff) * (100.0 / elapsed_ms)

        return (
            Point("q_drop_rate_100ms")
            .tag("flow_id", flow_id)
            .tag("src_ip", src_ip)
            .tag("dst_ip", dst_ip)
            .tag("switch_id", switch_id)
            .tag("egress_port", egress_port)
            .tag("queue_id", queue_id)
            .field("value", per100ms)
            .time(current_time)
        )

    def export_influxdb(self, flow_info):
        if not flow_info:
            return
        try:
            points = []
            flow_id = (flow_info.dst_port // 10) % 100  # digit 2 & 3
            expected_queue_id = flow_info.dst_port % 10  # digit 4

            # ---- Robust guard for partial/empty hop metadata ----
            # Determine a safe number of hops present across all arrays.
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
                # Nothing consistent to write; just drop this report.
                return

            # Use sink-hop egress timestamp as unified time if available; else fallback
            if len(flow_info.egress_tstamps) >= safe_hops:
                report_time = int(flow_info.egress_tstamps[safe_hops - 1])
            elif len(flow_info.ingress_tstamps) >= safe_hops:
                report_time = int(flow_info.ingress_tstamps[safe_hops - 1])
            else:
                report_time = int(time.time_ns())

            # Per-hop metrics
            for i in range(safe_hops):
                points.append(
                    Point("switch_latency")
                    .tag("flow_id", flow_id)
                    .tag("src_ip", flow_info.src_ip)
                    .tag("dst_ip", flow_info.dst_ip)
                    .tag("queue_id", flow_info.queue_ids[i])
                    .tag("switch_id", flow_info.switch_ids[i])
                    .field("value", flow_info.hop_latencies[i] / 1000.0)
                    .time(report_time)
                )
                points.append(
                    Point("tx_utilization")
                    .tag("flow_id", flow_id)
                    .tag("src_ip", flow_info.src_ip)
                    .tag("dst_ip", flow_info.dst_ip)
                    .tag("switch_id", flow_info.switch_ids[i])
                    .tag("egress_port", flow_info.l1_egress_ports[i])
                    .tag("queue_id", flow_info.queue_ids[i])
                    .field("value", flow_info.egress_tx_utils[i])
                    .time(report_time)
                )
                points.append(
                    Point("queue_occupancy")
                    .tag("flow_id", flow_id)
                    .tag("src_ip", flow_info.src_ip)
                    .tag("dst_ip", flow_info.dst_ip)
                    .tag("switch_id", flow_info.switch_ids[i])
                    .tag("queue_id", flow_info.queue_ids[i])
                    .field("value", flow_info.queue_occups[i])
                    .time(report_time)
                )

                # NEW: emit a drop-rate point on every report (except first sample)
                drp = self.record_drop_rate_instant(
                    flow_id,
                    flow_info.src_ip,
                    flow_info.dst_ip,
                    flow_info.switch_ids[i],
                    flow_info.l1_egress_ports[i],
                    flow_info.queue_ids[i],
                    flow_info.queue_drops[i],
                    report_time,
                )
                if drp is not None:
                    points.append(drp)

            # Link latency (device stamps), same unified time
            link_pairs = max(safe_hops - 1, 0)
            for i in range(link_pairs):
                # Guard against mismatched stamp lengths
                if i + 1 < len(flow_info.egress_tstamps) and i < len(flow_info.ingress_tstamps):
                    link_latency = abs(
                        flow_info.egress_tstamps[i + 1] - flow_info.ingress_tstamps[i]
                    ) / 1_000_000.0
                    points.append(
                        Point("link_latency")
                        .tag("flow_id", flow_id)
                        .tag("src_ip", flow_info.src_ip)
                        .tag("dst_ip", flow_info.dst_ip)
                        .tag("queue_id", expected_queue_id)
                        .tag("egress_switch_id", flow_info.switch_ids[i + 1])
                        .tag("egress_port_id", flow_info.l1_egress_ports[i + 1])
                        .tag("ingress_switch_id", flow_info.switch_ids[i])
                        .tag("ingress_port_id", flow_info.l1_ingress_ports[i])
                        .field("value", link_latency)
                        .time(report_time)
                    )

            # Flow latency, unified time (only if we have both ends)
            if len(flow_info.ingress_tstamps) >= 1 and len(flow_info.egress_tstamps) >= safe_hops:
                flow_latency = (
                    flow_info.ingress_tstamps[0]
                    - flow_info.egress_tstamps[safe_hops - 1]
                ) / 1_000_000.0
                points.append(
                    Point("flow_latency")
                    .tag("flow_id", flow_id)
                    .tag("src_ip", flow_info.src_ip)
                    .tag("dst_ip", flow_info.dst_ip)
                    .tag("queue_id", expected_queue_id)
                    .field("value", flow_latency)
                    .time(report_time)
                )

            # Single, atomic write for this report
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=points,
                write_precision=WritePrecision.NS,
            )

        finally:
            flow_info.clear_metadata()

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

        #flow_info.show()

        sys.stdout.flush()
        return flow_info
