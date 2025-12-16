#!/usr/bin/env python3
"""
monitor_iperf_s.py - Real-time monitoring of iperf3 UDP server logs

Shows INSTANT (most recent) values for immediate visibility into current network state.
Optionally shows rolling averages for trend analysis.
"""
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

# Simple "hot" thresholds to annotate bursts (!)
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


class ServerStats:
    """Track both instant and rolling stats for a server."""
    def __init__(self, window: int):
        self.window = window
        self.samples = deque(maxlen=window)
        # Instant (most recent sample)
        self.instant_mbps = 0.0
        self.instant_jitter = 0.0
        self.instant_lost = 0
        self.instant_total = 0
        self.last_update = 0.0
        
    def add(self, mbps, jitter_ms, lost, total):
        # Update instant values
        self.instant_mbps = mbps
        self.instant_jitter = jitter_ms
        self.instant_lost = lost
        self.instant_total = total
        self.last_update = time.time()
        # Add to rolling window
        self.samples.append((mbps, jitter_ms, lost, total))
    
    def get_instant(self):
        """Return instant (most recent) values."""
        return self.instant_mbps, self.instant_jitter, self.instant_lost, self.instant_total
    
    def get_rolling(self):
        """Return rolling average stats."""
        if not self.samples:
            return 0, 0.0, 0.0, 0, 0
        n = len(self.samples)
        mbps = sum(s[0] for s in self.samples) / n
        jitter = sum(s[1] for s in self.samples) / n
        lost = sum(s[2] for s in self.samples)
        total = sum(s[3] for s in self.samples)
        return n, mbps, jitter, lost, total
    
    def age_seconds(self):
        """How long since last update."""
        if self.last_update == 0:
            return float('inf')
        return time.time() - self.last_update


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


def fmt_loss_pct(lost, total):
    """Return just the percentage string."""
    if total <= 0: return "0.0"
    return f"{100.0 * lost / total:.1f}"


def print_table(rows):
    if not rows: return
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        print(line)
        if i == 0:
            print("-" * len(line))


def run(dirpath, pattern, window, refresh, show_rolling):
    tails = {}           # path -> FileTail
    stats = {}           # (host,port) -> ServerStats
    last_restart = {}    # (host,port) -> bool
    last_rescan = 0.0

    def rescan():
        files = glob.glob(os.path.join(dirpath, pattern))
        for p in files:
            if p not in tails:
                host, port = get_host_port_from_path(p)
                if host is None:
                    continue
                tails[p] = FileTail(p)
                stats[(host, port)] = ServerStats(window)

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
                stats[key].add(mbps, jitter_ms, lost, total)

        # -------- Build summaries --------
        # Per-server instant stats
        if show_rolling:
            server_rows = [["Host", "Port", "Q", "Mbps", "Jit ms", "Loss%", 
                           "RollMbps", "RollJit", "RollLoss%", "Age", "Flags"]]
        else:
            server_rows = [["Host", "Port", "Flow", "Q(DSCP)", "Mbps", "Jitter ms", 
                           "Loss %", "Lost/Total", "Age(s)", "Flags"]]

        # Per-host and per-queue INSTANT totals
        per_host_instant = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})
        per_q_instant = defaultdict(lambda: {"mbps": 0.0, "lost": 0, "total": 0})

        for (host, port), srv in sorted(stats.items()):
            i_mbps, i_jitter, i_lost, i_total = srv.get_instant()
            n, r_mbps, r_jitter, r_lost, r_total = srv.get_rolling()
            
            if n == 0:
                continue

            qid = queue_from_port(port)
            flow = flow_from_port(port)
            age = srv.age_seconds()
            
            # Flags
            flags = []
            if age > 3.0: flags.append("*")  # stale
            if last_restart.pop((host, port), False): flags.append("R")
            if i_mbps >= HOT_Mbps.get(qid, 1e9): flags.append("!")
            
            if show_rolling:
                server_rows.append([
                    host, str(port), str(qid),
                    f"{i_mbps:.3f}", f"{i_jitter:.2f}", fmt_loss_pct(i_lost, i_total),
                    f"{r_mbps:.3f}", f"{r_jitter:.2f}", fmt_loss_pct(r_lost, r_total),
                    f"{age:.1f}", "".join(flags)
                ])
            else:
                loss_pair, loss_pct = fmt_loss(i_lost, i_total)
                server_rows.append([
                    host, str(port), str(flow), f"{qid}({QID_DSCP.get(qid,'?')})",
                    f"{i_mbps:.3f}", f"{i_jitter:.2f}",
                    loss_pct, loss_pair, f"{age:.1f}", "".join(flags)
                ])

            # Aggregate instant values (only if data is fresh)
            if age < 5.0:
                per_host_instant[host]["mbps"] += i_mbps
                per_host_instant[host]["lost"] += i_lost
                per_host_instant[host]["total"] += i_total
                
                per_q_instant[qid]["mbps"] += i_mbps
                per_q_instant[qid]["lost"] += i_lost
                per_q_instant[qid]["total"] += i_total

        # ---------- Render ----------
        os.system("clear")
        mode_str = "INSTANT + Rolling" if show_rolling else "INSTANT"
        print(f"iperf3 UDP server monitor [{mode_str}] (refresh={refresh}s)")
        print("Flags: * = stale (>3s), R = restarted, ! = hot rate\n")

        print("=== Per-Server (INSTANT values) ===")
        print_table(server_rows)

        # Per-host instant totals
        host_rows = [["Host", "Mbps", "Loss %", "Lost/Total"]]
        for host in sorted(per_host_instant.keys()):
            h = per_host_instant[host]
            lp, pct = fmt_loss(h["lost"], h["total"])
            host_rows.append([host, f"{h['mbps']:.3f}", pct, lp])
        print("\n=== Per-Host INSTANT Totals ===")
        print_table(host_rows)

        # Per-queue instant totals (fabric-wide)
        q_rows = [["Queue(DSCP)", "Mbps", "Loss %", "Lost/Total"]]
        for qid in sorted(per_q_instant.keys()):
            q = per_q_instant[qid]
            lp, pct = fmt_loss(q["lost"], q["total"])
            q_rows.append([f"{qid}({QID_DSCP.get(qid,'?')})", f"{q['mbps']:.3f}", pct, lp])
        print("\n=== Per-Queue INSTANT Totals ===")
        print_table(q_rows)

        time.sleep(refresh)


def main():
    ap = argparse.ArgumentParser(
        description="Real-time iperf3 UDP server monitor - shows INSTANT values")
    ap.add_argument("--dir", default="/tmp", 
                    help="directory with *_iperf3_s_*.log")
    ap.add_argument("--pattern", default="*_iperf3_s_*.log", 
                    help="glob pattern")
    ap.add_argument("--window", type=int, default=30, 
                    help="rolling window size (samples) for comparison")
    ap.add_argument("--refresh", type=float, default=1.0, 
                    help="refresh interval seconds")
    ap.add_argument("--rolling", action="store_true",
                    help="show rolling averages alongside instant values")
    args = ap.parse_args()
    run(args.dir, args.pattern, args.window, args.refresh, args.rolling)


if __name__ == "__main__":
    main()
