import argparse
from p4utils.mininetlib.network_API import NetworkAPI
from controller import *

default_rule = 'rules/test/'

def config_network(p4):
    net = NetworkAPI()

    # Network general options
    net.setLogLevel('info')
    net.disableCli()

    # Network definition
    host_nodes = 8
    tor_nodes = 4
    agg_nodes = 4
    core_nodes = 2

    host_tor_bw = 10
    tor_agg_bw = 5
    agg_core_bw = 10

    # Hosts
    hosts = []
    for i in range(1, host_nodes + 1):
        host = net.addHost(f'h{i}')
        hosts.append(host)

    # ToR (Edge) switches
    tor_switches = []
    for i in range(1, tor_nodes + 1):
        tor_switch = net.addP4Switch(f't{i}', priority_queues_num=8,
                                     max_link_bw=host_tor_bw,
                                     cli_input=default_rule + f't{i}-commands.txt')
        tor_switches.append(tor_switch)

    # Aggregate switches
    agg_switches = []
    for i in range(1, agg_nodes + 1):
        agg_switch = net.addP4Switch(f'a{i}', priority_queues_num=8,
                                     max_link_bw=tor_agg_bw,
                                     cli_input=default_rule + f'a{i}-commands.txt')
        agg_switches.append(agg_switch)

    # Core switches
    core_switches = []
    for i in range(1, core_nodes + 1):
        core_switch = net.addP4Switch(f'c{i}', priority_queues_num=8,
                                      max_link_bw=agg_core_bw,
                                      cli_input=default_rule + f'c{i}-commands.txt')
        core_switches.append(core_switch)

    net.setP4SourceAll(p4)
    # Add links with 10 Mbps bandwidth
    # Connect hosts to ToR switches
    for i in range(tor_nodes):
        for j in range(2):  # Each ToR switch connects to 2 hosts
            net.addLink(hosts[i * 2 + j], tor_switches[i], bw=host_tor_bw)

    # Connect ToR switches to Aggregate switches
    for i in range(4):  # Each Pod has 2 ToR switches and 2 Agg switches
        for tor in tor_switches[i * 2: i * 2 + 2]:
            for agg in agg_switches[i * 2: i * 2 + 2]:
                net.addLink(tor, agg, bw=tor_agg_bw)

    # Connect Aggregate switches to Core switches
    for i in range(4):  # Each Aggregate switch connects to all Core switches
        for agg in agg_switches[i * 2: i * 2 + 2]:
            for core in core_switches:
                net.addLink(agg, core, bw=agg_core_bw)

    # Assignment strategy
    net.l3()

    # INT reports reciever host
    host100 = net.addHost('h100')
    host101 = net.addHost('h101')
    net.addLink(host100, tor_switches[0], port1=10, port2=10)
    net.setIntfIp(host100, tor_switches[0], "172.16.10.101/24")
    net.setIntfIp(tor_switches[0], host100, "172.16.10.100/24")
    net.setIntfMac(host100, tor_switches[0], "10:10:10:10:10:11")
    net.setIntfMac(tor_switches[0], host100, "10:10:10:10:10:10")

    net.addLink(host100, tor_switches[1], port1=11, port2=10)
    net.setIntfIp(host100, tor_switches[1], "172.16.11.101/24")
    net.setIntfIp(tor_switches[1], host100, "172.16.11.100/24")
    net.setIntfMac(host100, tor_switches[1], "10:10:10:10:11:11")
    net.setIntfMac(tor_switches[1], host100, "10:10:10:10:11:10")

    net.addLink(host101, tor_switches[2], port1=10, port2=10)
    net.setIntfIp(host101, tor_switches[2], "172.16.12.101/24")
    net.setIntfIp(tor_switches[2], host101, "172.16.12.100/24")
    net.setIntfMac(host101, tor_switches[2], "10:10:10:10:12:11")
    net.setIntfMac(tor_switches[2], host101, "10:10:10:10:12:10")

    net.addLink(host101, tor_switches[3], port1=11, port2=10)
    net.setIntfIp(host101, tor_switches[3], "172.16.13.101/24")
    net.setIntfIp(tor_switches[3], host101, "172.16.13.100/24")
    net.setIntfMac(host101, tor_switches[3], "10:10:10:10:13:11")
    net.setIntfMac(tor_switches[3], host101, "10:10:10:10:13:10")


    # Generate traffic
    
    net.addTask("h8", "python3 receive.py", 1, 0, True)
    #net.addTask("h8", "iperf3 -s -p 6017 -i 1", 1, 0, True)
    #net.addTask("h8", "iperf3 -s -p 6016 -i 1", 1, 0, True)
    #net.addTask("h8", "iperf3 -s -p 6010 -i 1", 1, 0, True)

    hosts_ips = ["0","10.7.1.2","10.7.2.2","10.8.3.2","10.8.4.2","10.9.5.2","10.9.6.2","10.10.7.2","10.10.8.2"]
    
    net.addTask("h1", f'python3 send.py --ip {hosts_ips[8]} --l4 udp --port 5017 --tos 184 --m "ToS is 184" --c 0', 5, 0, True)
    net.addTask("h1", f'python3 send.py --ip {hosts_ips[8]} --l4 udp --port 5016 --tos 96 --m "ToS is 96" --c 0', 5, 0, True)
    net.addTask("h1", f'python3 send.py --ip {hosts_ips[8]} --l4 udp --port 5010 --tos 0 --m "ToS is 0" --c 0', 5, 0, True)
    #net.addTask("h1", f'iperf3 -c {hosts_ips[8]} -i 1 -t 0 -p 6017 -u -b 3M -l 128 --tos 184', 2.1, 0, True)
    #net.addTask("h1", f'iperf3 -c {hosts_ips[8]} -i 1 -t 0 -p 6016 -u -b 3M -l 1250 --tos 96', 2.1, 0, True)
    #net.addTask("h1", f'iperf3 -c {hosts_ips[8]} -i 1 -t 0 -p 6010 -u -b 4M -l 1750 --tos 0', 2.1, 0, True)



    # Nodes general options
    #net.enableCpuPortAll()
    #net.enablePcapDumpAll()
    net.enableLogAll()

    return net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p4', help='p4 src file.',
                        type=str, required=False, default='p4src/int_md.p4')
                        
    return parser.parse_args()


def main():
    args = get_args()
    net = config_network(args.p4)
    net.startNetwork()
    # start the P4 controller
    controller = Controller()
    print("\n\nSUMMARY:")
    print("\nOSPF Shortest Paths:")
    controller.print_paths()  # Print the stored paths
    #print("\nP4 Table Entries:")
    #controller.print_forwarding_entries()  # Print forwarding entries

    net.enableCli()

    
    net.start_net_cli()

if __name__ == '__main__':
    main()