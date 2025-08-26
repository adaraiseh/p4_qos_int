import argparse 
from p4utils.mininetlib.network_API import NetworkAPI
from controller import *

default_rule = 'rules/test/'

# ----------------------------
# Helpers for traffic creation
# ----------------------------

# Fixed mapping you asked for: qid -> ToS(DSCP)
QID_TOS = {
    0: 184,   # voice
    1: 96,    # video
    7: 0,     # best effort
}

ALL_QUEUES = [0, 1, 7]

def _ensure_dict_per_queue(val, queues):
    """Allow single scalar or per-queue dict; return dict{qid: val}."""
    if isinstance(val, dict):
        return {qid: val.get(qid, list(val.values())[0]) for qid in queues}
    return {qid: val for qid in queues}

def _traffic_dst_port(flow_id: int, qid: int, base: int = 6000) -> int:
    """
    Build dst port so that:
      flow_id = (port // 10) % 100
      expected_queue_id = port % 10
    Using scheme: base + flow_id*10 + qid
    """
    if not (0 <= flow_id <= 99):
        raise ValueError("flow_id must be 0..99")
    if not (0 <= qid <= 9):
        raise ValueError("queue_id must be 0..9")
    return base + flow_id * 10 + qid

def _probe_dst_port(flow_id: int, qid: int, base: int = 5000) -> int:
    """
    Build dst port so that:
      flow_id = (port // 10) % 100
      expected_queue_id = port % 10
    Using scheme: base + flow_id*10 + qid
    """
    if not (0 <= flow_id <= 99):
        raise ValueError("flow_id must be 0..99")
    if not (0 <= qid <= 9):
        raise ValueError("queue_id must be 0..9")
    return base + flow_id * 10 + qid

def _host_to_ip(hostname: str, hosts_ips: list[str]) -> str:
    """
    Map 'hN' -> hosts_ips[N]. Keeps your existing static map to avoid
    reliance on getHostIp(), which you mentioned is not correct.
    """
    if not hostname.startswith('h'):
        raise ValueError("Hostnames must look like 'h1', 'h2', ...")
    idx = int(hostname[1:])
    if idx < 1 or idx >= len(hosts_ips):
        raise IndexError(f"No IP for {hostname} in hosts_ips")
    return hosts_ips[idx]

def generate_traffic(
    net: NetworkAPI,
    src_host: str,
    dst_host: str,
    flow_id: int,
    queue_id: int | str = "all",
    per_queue_bw: float | int | dict = 2.0,
    per_queue_len: int | dict = 0,
):
    """
    Schedule receiver(s) on dst and sender(s) on src according to:
      - src_host, dst_host: e.g., 'h1', 'h8'
      - flow_id: 0..99 (encoded into dst ports)
      - queue_id: int (0..9) or 'all' (== [0,1,7] per your mapping)
      - per_queue_bw: Mbps as float/int or dict{qid: Mbps}
      - per_queue_len: iperf3 -t duration seconds as int or dict{qid: seconds} (0 = continuous)
    Creates both:
      - iperf3 UDP streams (with --tos corresponding to queue)
      - your send.py UDP control messages (matching port & ToS)
    """

    # exactly the three queues you requested when "all"
    queues = ALL_QUEUES if queue_id == "all" else [int(queue_id)]

    # Per-queue params normalized to dicts
    bw_map = _ensure_dict_per_queue(per_queue_bw, queues)       # Mbps
    len_map = _ensure_dict_per_queue(per_queue_len, queues)     # seconds

    # Your static IP map used elsewhere in this file
    hosts_ips = [
        "0",                 # dummy index 0
        "10.7.1.2",          # h1
        "10.7.2.2",          # h2
        "10.8.3.2",          # h3
        "10.8.4.2",          # h4
        "10.9.5.2",          # h5
        "10.9.6.2",          # h6
        "10.10.7.2",         # h7
        "10.10.8.2",         # h8
        "10.11.9.2",         # h9
        "10.11.10.2",        # h10
        "10.12.11.2",        # h11
        "10.12.12.2",        # h12
        "10.13.13.2",        # h13
        "10.13.14.2",        # h14
        "10.14.15.2",        # h15
        "10.14.16.2"         # h16
    ]

    dst_ip = _host_to_ip(dst_host, hosts_ips)

    # Start iperf3 servers for each selected queue/port
    for qid in queues:
        probe_port = _probe_dst_port(flow_id, qid)
        traffic_port = _traffic_dst_port(flow_id, qid)
        #net.addTask(dst_host, f"python3 receive.py --proto all --ports {probe_port}", 1, 0, True)
        # net.addTask(
        #     dst_host,
        #     f"""python3 receive.py --proto all --ports {probe_port} >>/tmp/{dst_host}_recv_{probe_port}.log""",
        #     1, 0, True
        # )
        # iperf3 server: keep it running forever, restart if it dies
        # net.addTask(
        #     dst_host,
        #     f"""iperf3 -s -p {traffic_port} -i 1 --logfile /tmp/{dst_host}_iperf3_s_{traffic_port}.log""",
        #     1, 0, True
        # )
        net.addTask(
        dst_host,
        (
            f"bash -lc '"
            f"while true; do "
            f"  iperf3 -s -p {traffic_port} -i 1 "
            f"    --logfile /tmp/{dst_host}_iperf3_s_{traffic_port}.log ; "
            f"  echo \"[RESTART][$(date +%F_%T)] iperf3 server {traffic_port} exited ($?)\" "
            f"    >> /tmp/{dst_host}_iperf3_s_{traffic_port}.log ; "
            f"  sleep 1 ; "
            f"done'"
        ),
        1, 0, True
    )


    # Send traffic from src -> dst for each selected queue
    for qid in queues:
        tos = QID_TOS.get(qid)
        if tos is None:
            raise ValueError(f"No ToS mapping defined for queue {qid}")

        probe_port = _probe_dst_port(flow_id, qid)
        traffic_port = _traffic_dst_port(flow_id, qid)
        bw_mbps = bw_map[qid]
        length = len_map[qid]

        # Your lightweight sender (control/marker packets), matches ToS and port
        # net.addTask(
        #     src_host,
        #     f'python3 send.py --ip {dst_ip} --l4 udp --port {probe_port} --tos {tos} --m "flow {flow_id}, q{qid}, ToS {tos}" --c 0',
        #     1.5, 0, True
        # )
        
        # iperf3 client: run forever, auto-restart on any exit, keep logs
        # net.addTask(
        #     src_host,
        #     f"""iperf3 -c {dst_ip} -p {traffic_port} -u -b {bw_mbps}M -l {length} --tos {tos} \
        #         -i 1 -t 3600 --connect-timeout 5000 --logfile /tmp/{src_host}_iperf3_c_{traffic_port}.log""",
        #     2.0, 0, True
        # )
        net.addTask(
            src_host,
            (
                f"bash -lc '"
                f"while true; do "
                f"  iperf3 -c {dst_ip} -p {traffic_port} -u "
                f"         -b {bw_mbps}M -l {length} --tos {tos} "
                f"         -i 1 -t 0 --connect-timeout 5000 "
                f"         >> /tmp/{src_host}_iperf3_c_{traffic_port}.log 2>&1 ; "
                f"  echo \"[RESTART][$(date +%F_%T)] iperf3 client {traffic_port} exited ($?)\" "
                f"    >> /tmp/{src_host}_iperf3_c_{traffic_port}.log ; "
                f"  sleep 1 ; "
                f"done'"
            ),
            2.0, 0, True
        )


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

    # INT reports receiver hosts
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

    # -----------------
    # Generate traffic (half-host senders, hosts 1..8 only)
    # -----------------

    # Per-flow, per-queue Mbps (kept same shape, scaled by LOAD_FACTOR)
    LOAD_FACTOR = 1.3
    PER_QUEUE_BW = {
        0: 0.3 * LOAD_FACTOR,
        1: 0.4 * LOAD_FACTOR,
        7: 0.5 * LOAD_FACTOR,
    }
    PER_QUEUE_LEN = {0: 1250, 1: 1250, 7: 1250}

    # Define pods as pairs and pick only half (first) as senders
    pods = [
        ["h1", "h2"],
        ["h3", "h4"],
        ["h5", "h6"],
        ["h7", "h8"],
    ]
    senders   = [pod[0] for pod in pods]  # h1, h3, h5, h7
    receivers = [pod[1] for pod in pods]  # h2, h4, h6, h8  (used as pure receivers)

    # Build src->dst pairs:
    # For each sender in pod i, send to the *second* host of every other pod.
    balanced_pairs = []
    next_flow_id = 10
    for i, s in enumerate(senders):
        for j, pod in enumerate(pods):
            if j == i:
                continue  # skip same pod
            dst = pod[1]  # the non-sender half from other pod
            balanced_pairs.append((s, dst, next_flow_id))
            next_flow_id += 1

    # Schedule traffic for the selected pairs
    for src, dst, fid in balanced_pairs:
        generate_traffic(
            net=net,
            src_host=src,
            dst_host=dst,
            flow_id=fid,
            queue_id="all",
            per_queue_bw=PER_QUEUE_BW,   # Mbps per queue
            per_queue_len=PER_QUEUE_LEN  # payload bytes
        )

    # Nodes general options
    #net.enableCpuPortAll()
    #net.enablePcapDumpAll()
    #net.enableLogAll()

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
    controller.print_paths()

    net.enableCli()
    net.start_net_cli()

if __name__ == '__main__':
    main()
