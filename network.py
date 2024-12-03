import argparse
from p4utils.mininetlib.network_API import NetworkAPI

default_rule = 'rules/test/'

def config_network(p4):
    net = NetworkAPI()

    # Network general options
    net.setLogLevel('info')
    net.enableCli()

    # Network definition
    host_nodes = 8
    tor_nodes = 4
    agg_nodes = 4
    core_nodes = 2

    # Hosts
    hosts = []
    for i in range(1, host_nodes + 1):
        host = net.addHost(f'h{i}')
        hosts.append(host)

    # ToR (Edge) switches
    tor_switches = []
    for i in range(1, tor_nodes + 1):
        tor_switch = net.addP4Switch(f't{i}', priority_queues_num=4, cli_input=default_rule + f't{i}-commands.txt')
        tor_switches.append(tor_switch)

    # Aggregate switches
    agg_switches = []
    for i in range(1, agg_nodes + 1):
        agg_switch = net.addP4Switch(f'a{i}', priority_queues_num=4, cli_input=default_rule + f'a{i}-commands.txt')
        agg_switches.append(agg_switch)

    # Core switches
    core_switches = []
    for i in range(1, core_nodes + 1):
        core_switch = net.addP4Switch(f'c{i}', priority_queues_num=4, cli_input=default_rule + f'c{i}-commands.txt')
        core_switches.append(core_switch)

    net.setP4SourceAll(p4)
    # Add links with 1 Mbps bandwidth
    # Connect hosts to ToR switches
    for i in range(tor_nodes):
        for j in range(2):  # Each ToR switch connects to 2 hosts
            net.addLink(hosts[i * 2 + j], tor_switches[i], bw=1)

    # Connect ToR switches to Aggregate switches
    for i in range(4):  # Each Pod has 2 ToR switches and 2 Agg switches
        for tor in tor_switches[i * 2: i * 2 + 2]:
            for agg in agg_switches[i * 2: i * 2 + 2]:
                net.addLink(tor, agg, bw=1)

    # Connect Aggregate switches to Core switches
    for i in range(4):  # Each Aggregate switch connects to all Core switches
        for agg in agg_switches[i * 2: i * 2 + 2]:
            for core in core_switches:
                net.addLink(agg, core, bw=1)

    # Assignment strategy
    net.mixed()

    # Nodes general options
    net.enableCpuPortAll()
    net.enablePcapDumpAll()
    net.enableLogAll()

    return net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p4', help='p4 src file.',
                        type=str, required=False, default='p4src/int_mri.p4')
                        
    return parser.parse_args()


def main():
    args = get_args()
    net = config_network(args.p4)
    net.startNetwork()


if __name__ == '__main__':
    main()