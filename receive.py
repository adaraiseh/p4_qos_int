#!/usr/bin/env python
import sys
import os
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

def handle_pkt(pkt):
    """
    Handles each packet. Replace the `pass` with packet processing logic if needed.
    """
    pass  # Replace with processing logic if necessary

def main():
    iface = get_if()
    print(f"Sniffing on {iface}")
    sys.stdout.flush()

    # Sniff packets and avoid retaining them in memory
    sniff(
        iface=iface,
        filter="inbound and tcp or udp",  # Only TCP or UDP inbound traffic
        prn=handle_pkt,  # Call handle_pkt for each packet
        store=False  # Do not store packets in memory
    )

if __name__ == "__main__":
    main()
