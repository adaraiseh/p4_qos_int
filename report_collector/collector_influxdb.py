#!/usr/bin/env python3

import sys
from scapy.all import sniff
from influxdb_client import InfluxDBClient
from collector import *

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "pkyJUX9Itrw-y8YuTx3kLDAQ_VYyR_MxnyvtFmHwnRQOjDb7n2QBUFt7piMNgl9TU6IujEJpi8cMEKnwGs77dA=="  # Replace with your InfluxDB 2 token
INFLUX_ORG = "research"
INFLUX_BUCKET = "INT"

def handle_pkt(pkt, c):
    if INTREP in pkt:
        print("\n\n********* Receiving Telemetry Report ********")
        flow_info = c.parser_int_pkt(pkt)
        flow_info.show()
        c.export_influxdb(flow_info)

def main():
    iface = ['t1-eth10','t2-eth10','t3-eth10','t4-eth10']
    print("Sniffing on %s" % iface)
    sys.stdout.flush()

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    c = Collector(influx_client, INFLUX_ORG, INFLUX_BUCKET)
    sniff(
        iface=iface,
        filter='inbound and tcp or udp',
        prn=lambda x: handle_pkt(x, c)
    )

if __name__ == '__main__':
    main()