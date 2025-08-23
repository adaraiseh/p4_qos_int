#!/usr/bin/env python3
import argparse, glob, os, re, time
from collections import defaultdict, deque

# Parse typical iperf3 UDP server lines:
# [  6] 123.00-124.00 sec  70.8 KBytes   580 Kbits/sec  10.164 ms  0/58 (0%)
LINE_RE = re.compile(
    r"\[\s*\d+\]\s+(?P<t0>\d+(?:\.\d+)?)-(?P<t1>\d+(?:\.\d+)?)\s+sec\s+"
    r"(?P<size>\d+(?:\.\d+)?)\s+(?P<size_unit>[KMG]Bytes)\s+"
    r"(?P<rate>\d+(?:\.\d+)?)\s+(?P<rate_unit>[KMG]bits/sec)\s+"
    r"(?P<jitter_ms>\d+(?:\.\d+)?)\s+ms\s+"
    r"(?P<lost>\d+)\/(?P<total>\d+)\s+\((?P<loss_pct>\d+(?:\.\d+)?)%\)"
)

def rate_to_mbps(v: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("kbit"): return v / 1000.0
    if u.startswith("mbit"): return v
    if u.startswith("gbit"): return v * 1000.0
    return 0.0

def queue_from_port(port: int) -> int: return port % 10
def flow_from_port(port: int) -> int:  return (port // 10) % 100

def get_host_port_from_path(path: str):
    base = os.path.basename(path)
    # h7_iperf3_s_6140.log
    m = re.match(r"(h\d+)_iperf3_s_(\d+)\.log$", base)
    if not m: return None, None
    return m.group(1), int(m.group(2))

class Rolling:
    """Rolling window of samples per server: (mbps, jitter_ms, lost, total)."""
    def __init__(self, window: int):
        self.window = window
        self.samples = deque(maxlen=window)  # tuples

    def add(self, mbps, jitter_ms, lost, total):
        self.samples.append((mbps, jitter_ms, lost, total))

    def stats(self):
        if not self.samples: return 0, 0.0, 0.0, 0, 0
        n = len(self.samples)
        mbps = sum(s[0] for s in self.samples) / n
        jitter = sum(s[1] for s in self.samples) / n
        lost = sum(s[2] for s in self.samples)
        total = sum(s[3] for s in self.samples)
        return n, mbps, jitter, lost, total

class FileTail:
    """Efficient incremental reader with seek position & truncation handling."""
    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.inode = None

    def read_new_lines(self):
        lines = []
        try:
            st = os.stat(self.path)
            # Handle rotation/truncation/inode changes
            if self.inode is None or self.inode != st.st_ino or st.st_size < self.pos:
                self.pos = 0
                self.inode = st.st_ino
            with open(self.path, "r", errors="ignore") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return lines
        if not chunk:
            return lines
        # Ensure we split on lines; drop partial trailing line if any
        parts = chunk.splitlines()
        return parts

def print_table(rows):
    if not rows: return
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        print(line)
        if i == 0:
            print("-" * len(line))

def run(dirpath, pattern, window, refresh):
    # Discover files once, but also re‑scan occasionally (cheap)
    last_rescan = 0
    tails = {}         # path -> FileTail
    rolls = {}         # (host,port) -> Rolling
    latest_jitter = {} # (host,port) -> last jitter_ms seen

    def rescan():
        files = glob.glob(os.path.join(dirpath, pattern))
        for p in files:
            if p not in tails:
                tails[p] = FileTail(p)
                rolls[get_host_port_from_path(p)] = Rolling(window)

    while True:
        now = time.time()
        if now - last_rescan > 5:  # light periodic rescan
            rescan()
            last_rescan = now

        # Parse only new lines for each file
        for path, tail in list(tails.items()):
            host, port = get_host_port_from_path(path)
            if host is None: 
                continue
            for line in tail.read_new_lines():
                m = LINE_RE.search(line)
                if not m: 
                    continue
                mbps = rate_to_mbps(float(m.group("rate")), m.group("rate_unit"))
                jitter_ms = float(m.group("jitter_ms"))  # <-- this is jitter, not latency
                lost = int(m.group("lost"))
                total = int(m.group("total"))
                rolls[(host, port)].add(mbps, jitter_ms, lost, total)
                latest_jitter[(host, port)] = jitter_ms

        # Build summary
        server_rows = [["Host", "Port", "Flow", "Q", "Samples", "Avg Mbps", "Avg Jitter ms", "Loss %", "Lost/Total", "Latest ms"]]
        per_host = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})
        per_host_q = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})

        for (host, port), roll in sorted(rolls.items()):
            n, mbps, jitter, lost, total = roll.stats()
            if n == 0: 
                continue
            loss_pct = (lost / total * 100.0) if total > 0 else 0.0
            server_rows.append([
                host, str(port), str(flow_from_port(port)), str(queue_from_port(port)),
                str(n), f"{mbps:.3f}", f"{jitter:.3f}", f"{loss_pct:.2f}", f"{lost}/{total}",
                f"{latest_jitter.get((host, port), 0.0):.3f}"
            ])
            per_host[host]["mbps"] += mbps
            per_host[host]["lost"] += lost
            per_host[host]["total"] += total
            key = (host, queue_from_port(port))
            per_host_q[key]["mbps"] += mbps
            per_host_q[key]["lost"] += lost
            per_host_q[key]["total"] += total

        os.system("clear")
        print(f"iperf3 UDP server summaries (window={window} samples, refresh={refresh}s)")
        print("\nPer-server\n")
        print_table(server_rows)

        # host_rows = [["Host", "Sum Mbps", "Loss %", "Lost/Total"]]
        # for h, d in sorted(per_host.items()):
        #     loss_pct = (d["lost"]/d["total"]*100.0) if d["total"]>0 else 0.0
        #     host_rows.append([h, f"{d['mbps']:.3f}", f"{loss_pct:.2f}", f"{d['lost']}/{d['total']}"])
        # print("\nPer-host aggregate\n")
        # print_table(host_rows)

        # hq_rows = [["Host", "Q", "Sum Mbps", "Loss %", "Lost/Total"]]
        # for (h,q), d in sorted(per_host_q.items()):
        #     loss_pct = (d["lost"]/d["total"]*100.0) if d["total"]>0 else 0.0
        #     hq_rows.append([h, str(q), f"{d['mbps']:.3f}", f"{loss_pct:.2f}", f"{d['lost']}/{d['total']}"])
        # print("\nPer-host per-queue aggregate\n")
        # print_table(hq_rows)

        time.sleep(refresh)

def main():
    ap = argparse.ArgumentParser(description="Lightweight rolling summary for iperf3 UDP server logs.")
    ap.add_argument("--dir", default="/tmp", help="directory with *_iperf3_s_*.log")
    ap.add_argument("--pattern", default="*_iperf3_s_*.log", help="glob pattern")
    ap.add_argument("--window", type=int, default=60, help="rolling window (samples)")
    ap.add_argument("--refresh", type=float, default=2.0, help="refresh interval seconds")
    args = ap.parse_args()
    run(args.dir, args.pattern, args.window, args.refresh)

if __name__ == "__main__":
    main()
