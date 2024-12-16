#!/usr/bin/env python3

import sys
import signal
from scapy.all import sniff
from influxdb_client import InfluxDBClient
from collector import *

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "pkyJUX9Itrw-y8YuTx3kLDAQ_VYyR_MxnyvtFmHwnRQOjDb7n2QBUFt7piMNgl9TU6IujEJpi8cMEKnwGs77dA=="  # Replace with your InfluxDB 2 token
INFLUX_ORG = "research"
INFLUX_BUCKET = "INT"

def handle_pkt(pkt, c):
    if INTREP in pkt:
        flow_info = c.parser_int_pkt(pkt)
        if flow_info:
            c.export_influxdb(flow_info)
            flow_info.clear_metadata()  # Custom cleanup for FlowInfo

def main():
    iface = ['t1-eth10', 't2-eth10', 't3-eth10', 't4-eth10']
    print(f"Sniffing on {iface}")
    sys.stdout.flush()

    # Initialize the InfluxDB client and the Collector
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    c = Collector(influx_client, INFLUX_ORG, INFLUX_BUCKET)

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        print("\nStopping sniffing and shutting down...")
        c.flush_buffer()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start sniffing
    sniff(
        iface=iface,
        filter='inbound and tcp or udp',
        prn=lambda x: handle_pkt(x, c),
        store=False  # Avoid storing packets in memory
    )

if __name__ == '__main__':
    main()