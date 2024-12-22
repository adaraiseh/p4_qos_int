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
    #print(pkt.summary())  # Print packet summary for debugging
    pass

def main():
    # Get the network interface
    iface = get_if()
    print(f"Sniffing on {iface}")
    sys.stdout.flush()

    # Define the ports to listen to
    specific_ports = [5010, 5011, 5012, 5013, 5020, 5021, 5022, 5023, 5030, 5031, 5032, 5033, 5040, 5041, 5042, 5043, 5050, 5051, 5052, 5053, 5060, 5061, 5062, 5063]
    port_filter = " or ".join([f"tcp port {p} or udp port {p}" for p in specific_ports])

    # Sniff packets and avoid retaining them in memory
    sniff(
        iface=iface,
        filter=port_filter,  # Filter packets to specific TCP/UDP ports
        prn=handle_pkt,  # Call handle_pkt for each packet
        store=False  # Do not store packets in memory
    )

if __name__ == "__main__":
    main()
