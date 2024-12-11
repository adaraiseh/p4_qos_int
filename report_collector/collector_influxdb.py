#!/usr/bin/env python3

import sys
from scapy.all import sniff
from influxdb_client import InfluxDBClient
from collector import *
from queue import Queue
from threading import Thread

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "pkyJUX9Itrw-y8YuTx3kLDAQ_VYyR_MxnyvtFmHwnRQOjDb7n2QBUFt7piMNgl9TU6IujEJpi8cMEKnwGs77dA=="  # Replace with your InfluxDB 2 token
INFLUX_ORG = "research"
INFLUX_BUCKET = "INT"

# Packet processing thread function
def packet_processor(queue, collector):
    while True:
        pkt = queue.get()
        if pkt is None:  # Exit signal
            break
        if INTREP in pkt:
            flow_info = collector.parser_int_pkt(pkt)
            collector.export_influxdb(flow_info)
        queue.task_done()

def main():
    iface = 't4-eth1'
    print("Sniffing on %s" % iface)
    sys.stdout.flush()

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    collector = Collector(influx_client, INFLUX_ORG, INFLUX_BUCKET)

    # Create a thread-safe queue
    packet_queue = Queue(maxsize=1000)  # Adjust maxsize based on expected packet volume

    # Create and start worker threads
    num_threads = 4  # Number of threads to use
    threads = []
    for _ in range(num_threads):
        t = Thread(target=packet_processor, args=(packet_queue, collector))
        t.daemon = True
        t.start()
        threads.append(t)

    # Start sniffing
    sniff(
        iface=iface,
        filter='inbound and tcp or udp',
        prn=lambda pkt: packet_queue.put(pkt),
        store=False  # Prevent storing packets in memory
    )

    # Wait for all packets in the queue to be processed
    packet_queue.join()

    # Stop threads
    for _ in threads:
        packet_queue.put(None)  # Signal threads to exit
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
