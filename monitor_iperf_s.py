#!/usr/bin/env python3
import argparse, glob, os, re, time
from collections import defaultdict, deque

# Parse typical iperf3 UDP *server* interval lines, e.g.:
# [  6] 123.00-124.00 sec  70.8 KBytes   580 Kbits/sec  10.164 ms  0/58 (0%)
LINE_RE = re.compile(
    r"\[\s*\d+\]\s+(?P<t0>\d+(?:\.\d+)?)-(?P<t1>\d+(?:\.\d+)?)\s+sec\s+"
    r"(?P<size>\d+(?:\.\d+)?)\s+(?P<size_unit>[KMG]Bytes)\s+"
    r"(?P<rate>\d+(?:\.\d+)?)\s+(?P<rate_unit>[KMG]bits/sec)\s+"
    r"(?P<jitter_ms>\d+(?:\.\d+)?)\s+ms\s+"
    r"(?P<lost>\d+)\/(?P<total>\d+)\s+\((?P<loss_pct>\d+(?:\.\d+)?)%\)"
)

RESTART_RE = re.compile(r"\[RESTART\]")  # from the server wrapper we added

# qid -> DSCP label (for readability only)
QID_DSCP = {0: "EF(0x2E)", 1: "CS3(0x18)", 7: "BE(0x00)"}

# Simple “hot” thresholds to annotate bursts (!)
HOT_Mbps = {0: 0.30, 1: 4.0, 7: 6.0}

def rate_to_mbps(v: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("kbit"): return v / 1000.0
    if u.startswith("mbit"): return v
    if u.startswith("gbit"): return v * 1000.0
    return 0.0

def queue_from_port(port: int) -> int: return port % 10
def flow_from_port(port: int) -> int:  return (port // 10) % 100

def get_host_port_from_path(path: str):
    # Matches /tmp/hX_iperf3_s_<port>.log
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
            return []
        if not chunk:
            return []
        return chunk.splitlines()

def fmt_loss(lost, total):
    if total <= 0: return "0/0", "0.00"
    pct = 100.0 * (lost / total)
    return f"{lost}/{total}", f"{pct:.2f}"

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
    tails = {}           # path -> FileTail
    rolls = {}           # (host,port) -> Rolling
    latest_jitter = {}   # (host,port) -> last jitter_ms seen
    last_restart = {}    # (host,port) -> bool (seen restart marker recently)

    # Monotonic cumulative packet totals for stall detection
    cum_total = defaultdict(int)   # (host,port) -> cumulative total packets seen
    prev_cum_total = {}            # (host,port) -> value at last refresh

    def rescan():
        files = glob.glob(os.path.join(dirpath, pattern))
        for p in files:
            if p not in tails:
                host, port = get_host_port_from_path(p)
                if host is None:
                    continue
                tails[p] = FileTail(p)
                rolls[(host, port)] = Rolling(window)

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
                # restart marker from our persistent server wrapper
                if RESTART_RE.search(line):
                    last_restart[key] = True
                    continue
                m = LINE_RE.search(line)
                if not m:
                    continue
                mbps = rate_to_mbps(float(m.group("rate")), m.group("rate_unit"))
                jitter_ms = float(m.group("jitter_ms"))
                lost = int(m.group("lost"))
                total = int(m.group("total"))

                rolls[key].add(mbps, jitter_ms, lost, total)
                latest_jitter[key] = jitter_ms
                cum_total[key] += total

        # -------- summaries --------
        # Per-server (port encodes flow & queue)
        server_rows = [["Host", "Port", "Flow", "Q(DSCP)", "Samples", "Avg Mbps", "Avg Jit ms",
                        "Loss %", "Lost/Total", "Latest ms", "Flags"]]

        # Per-host totals & per-queue totals across fabric
        per_host = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})
        per_q = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})  # qid -> agg

        for (host, port), roll in sorted(rolls.items()):
            n, mbps, jitter, lost, total = roll.stats()
            if n == 0:
                continue

            qid = queue_from_port(port)
            flow = flow_from_port(port)
            loss_pair, loss_pct = fmt_loss(lost, total)
            prev = prev_cum_total.get((host, port), None)
            stalled = (prev is not None and cum_total[(host, port)] <= prev)

            # flags: '*' stall, 'R' restarted since last refresh, '!' hot rate
            flags = []
            if stalled: flags.append("*")
            if last_restart.pop((host, port), False): flags.append("R")
            if mbps >= HOT_Mbps.get(qid, 1e9): flags.append("!")

            server_rows.append([
                host, str(port), str(flow), f"{qid}({QID_DSCP.get(qid,'?')})",
                str(n), f"{mbps:.3f}", f"{jitter:.3f}",
                loss_pct, loss_pair, f"{latest_jitter.get((host, port), 0.0):.3f}",
                "".join(flags)
            ])

            per_host[host]["mbps"] += mbps
            per_host[host]["lost"] += lost
            per_host[host]["total"] += total

            per_q[qid]["mbps"] += mbps
            per_q[qid]["lost"] += lost
            per_q[qid]["total"] += total

        # ---------- render ----------
        os.system("clear")
        print(f"iperf3 UDP server summaries (window={window} samples, refresh={refresh}s)")
        print("Flags: * = no new packets since last refresh, R = server restarted, ! = hot rate threshold crossed\n")

        print("Per-server")
        print_table(server_rows)

        # Per-host totals
        host_rows = [["Host", "Total Mbps", "Loss %", "Lost/Total"]]
        for host in sorted(per_host.keys()):
            h = per_host[host]
            lp, pct = fmt_loss(h["lost"], h["total"])
            host_rows.append([host, f"{h['mbps']:.3f}", pct, lp])
        print("\nPer-host totals")
        print_table(host_rows)

        # Per-queue (fabric-wide)
        q_rows = [["Queue(DSCP)", "Total Mbps", "Loss %", "Lost/Total"]]
        for qid in sorted(per_q.keys()):
            q = per_q[qid]
            lp, pct = fmt_loss(q["lost"], q["total"])
            q_rows.append([f"{qid}({QID_DSCP.get(qid,'?')})", f"{q['mbps']:.3f}", pct, lp])
        print("\nPer-queue totals")
        print_table(q_rows)

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
