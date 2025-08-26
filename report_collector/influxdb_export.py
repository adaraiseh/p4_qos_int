#!/usr/bin/env python3
import sys
import signal
from scapy.all import AsyncSniffer, conf
from influxdb_client import InfluxDBClient
from collector import *

INFLUX_URL = "http://192.168.201.1:8086"
INFLUX_TOKEN = "0fO0ojKAANp-7aEehJHRDWEKE-cSNoIEHY2aK8dd1KI0VWpmO1GAsMJhRh_B1U8bXDIaozHMDVv1yEkCPm230w=="
INFLUX_ORG = "research"
INFLUX_BUCKET = "INT"

BPF = "udp and dst port 1234"   # <<<< narrowed; huge CPU win

def handle_pkt(pkt, c: Collector):
    if INTREP in pkt:
        fi = c.parser_int_pkt(pkt)
        if fi:
            c.export_influxdb(fi)

def main():
    iface = ['t1-eth10', 't2-eth10', 't3-eth10', 't4-eth10']
    print(f"Sniffing on {iface} with BPF: {BPF}")
    sys.stdout.flush()

    # Scapy performance knobs (optional, but helpful)
    conf.use_pcap = True          # prefer libpcap
    conf.sniff_promisc = 0        # no promiscuous unless needed

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # Async writer (flush ~0.5s). If your device clocks are skewed, set use_device_time=False.
    c = Collector(influx_client, INFLUX_ORG, INFLUX_BUCKET,
                  write_async=True, flush_interval_ms=500, batch_size=1000,
                  use_device_time=False,
                  aggregate_enabled=True,
                  aggregate_window_ms=500)

    stop = False
    def signal_handler(sig, frame):
        nonlocal stop
        stop = True
        print("\nStopping...")
        c.flush_buffer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sniffer = AsyncSniffer(iface=iface, filter=BPF, store=False,
                           prn=lambda x: handle_pkt(x, c))
    sniffer.start()
    try:
        while not stop:
            signal.pause()
    except Exception:
        pass
    finally:
        try:
            sniffer.stop()
        except Exception:
            pass
        c.flush_buffer()

if __name__ == '__main__':
    main()
