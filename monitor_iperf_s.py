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
    m = re.match(r"(h\d+)_iperf3_s_(\d+)\.log$", base)
    if not m: return None, None
    return m.group(1), int(m.group(2))

class Rolling:
    """Rolling window of samples per server: (mbps, jitter_ms, lost, total)."""
    def __init__(self, window: int):
        self.window = window
        self.samples = deque(maxlen=window)

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
        return chunk.splitlines()

def print_table(rows):
    if not rows: return
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        print(line)
        if i == 0:
            print("-" * len(line))

def run(dirpath, pattern, window, refresh):
    last_rescan = 0
    tails = {}          # path -> FileTail
    rolls = {}          # (host,port) -> Rolling
    latest_jitter = {}  # (host,port) -> last jitter_ms seen

    # NEW: monotonic cumulative "total" and previous snapshot for stall detection
    cum_total = defaultdict(int)   # (host,port) -> cumulative total packets seen so far
    prev_cum_total = {}            # (host,port) -> value at last refresh

    def rescan():
        files = glob.glob(os.path.join(dirpath, pattern))
        for p in files:
            if p not in tails:
                tails[p] = FileTail(p)
                rolls[get_host_port_from_path(p)] = Rolling(window)

    while True:
        now = time.time()
        if now - last_rescan > 5:
            rescan()
            last_rescan = now

        # Parse only new lines for each file
        for path, tail in list(tails.items()):
            host, port = get_host_port_from_path(path)
            if host is None:
                continue
            key = (host, port)
            for line in tail.read_new_lines():
                m = LINE_RE.search(line)
                if not m:
                    continue
                mbps = rate_to_mbps(float(m.group("rate")), m.group("rate_unit"))
                jitter_ms = float(m.group("jitter_ms"))
                lost = int(m.group("lost"))
                total = int(m.group("total"))

                # Update rolling stats
                rolls[key].add(mbps, jitter_ms, lost, total)
                latest_jitter[key] = jitter_ms

                # Update monotonic cumulative total (used for stall detection)
                cum_total[key] += total

        # Build summary
        server_rows = [["Host", "Port", "Flow", "Q", "Samples", "Avg Mbps", "Avg Jitter ms", "Loss %", "Lost/Total", "Latest ms"]]
        per_host = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})
        per_host_q = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})

        for (host, port), roll in sorted(rolls.items()):
            n, mbps, jitter, lost, total = roll.stats()
            if n == 0:
                continue

            key = (host, port)
            # Stall if the cumulative total didn't increase since last refresh
            prev = prev_cum_total.get(key, None)
            stalled = (prev is not None and cum_total[key] <= prev)

            loss_pct = (lost / total * 100.0) if total > 0 else 0.0
            host_disp = f"{host}{'*' if stalled else ''}"
            server_rows.append([
                host_disp, str(port), str(flow_from_port(port)), str(queue_from_port(port)),
                str(n), f"{mbps:.3f}", f"{jitter:.3f}", f"{loss_pct:.2f}", f"{lost}/{total}",
                f"{latest_jitter.get((host, port), 0.0):.3f}"
            ])

            per_host[host]["mbps"] += mbps
            per_host[host]["lost"] += lost
            per_host[host]["total"] += total
            key_hq = (host, queue_from_port(port))
            per_host_q[key_hq]["mbps"] += mbps
            per_host_q[key_hq]["lost"] += lost
            per_host_q[key_hq]["total"] += total

        os.system("clear")
        print(f"iperf3 UDP server summaries (window={window} samples, refresh={refresh}s)")
        print("(* = no new packets since last refresh)")
        print("\nPer-server\n")
        print_table(server_rows)

        # After printing, snapshot current cumulative totals for the next refresh
        prev_cum_total = dict(cum_total)

        time.sleep(refresh)

def main():
    ap = argparse.ArgumentParser(description="Lightweight rolling summary for iperf3 UDP server logs.")
    ap.add_argument("--dir", default="/tmp", help="directory with *_iperf3_s_*.log")
    ap.add_argument("--pattern", default="*_iperf3_s_*.log", help="glob pattern")
    ap.add_argument("--window", type=int, default=60, help="rolling window (samples)")
    ap.add_argument("--refresh", type=float, default=1.0, help="refresh interval seconds")
    args = ap.parse_args()
    run(args.dir, args.pattern, args.window, args.refresh)

if __name__ == "__main__":
    main()
