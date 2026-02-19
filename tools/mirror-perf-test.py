#!/usr/bin/env python3
"""Armbian mirror performance tester.

Fetches the mirror list and redirect info from apt.armbian.com in parallel,
then measures latency, jitter, and packet loss for every mirror.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

APT_BASE    = "http://apt.armbian.com/"
MIRRORS_URL = "https://apt.armbian.com/mirrors"
IS_TTY      = sys.stdout.isatty()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test Armbian mirror performance.")
    p.add_argument("-n", "--samples",  type=int,   default=10,    metavar="N",    help="Samples per mirror (default: 10)")
    p.add_argument("-p", "--parallel", type=int,   default=4,     metavar="N",    help="Parallel mirror tests (default: 4)")
    p.add_argument("-s", "--suite",    default=os.environ.get("SUITE", "trixie"), metavar="SUITE", help="Debian suite (default: trixie / $SUITE)")
    p.add_argument("-t", "--timeout",  type=float, default=10.0,  metavar="SEC",  help="Per-request timeout in seconds (default: 10)")
    return p.parse_args()


# ── Network ───────────────────────────────────────────────────────────────

def fetch_redirect_info():
    """HEAD apt.armbian.com and return the redirect headers (does not follow)."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)

    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(APT_BASE, method="HEAD")
    try:
        resp = opener.open(req, timeout=10)
        return resp.headers
    except urllib.error.HTTPError as e:
        return e.headers


def fetch_mirrors():
    """Return a list of {name, url} dicts from the Armbian mirrors API."""
    try:
        with urllib.request.urlopen(MIRRORS_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"ERROR: could not fetch mirror list: {e}", file=sys.stderr)
        sys.exit(1)

    seen, mirrors = set(), []
    for region, urls in data.items():
        for url in urls:
            if url not in seen:
                seen.add(url)
                host = url.split("//")[-1].split("/")[0]
                mirrors.append({"name": f"{host} [{region}]", "url": url.rstrip("/")})
    return mirrors


def fetch_apt_info():
    """Fetch redirect info and mirror list in parallel; return (redirect_headers, mirrors)."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_redir   = pool.submit(fetch_redirect_info)
        f_mirrors = pool.submit(fetch_mirrors)
        redirect  = f_redir.result()
        mirrors   = f_mirrors.result()
    return redirect, mirrors


def test_mirror(mirror, file_path, samples, timeout, progress_cb=None):
    """Fetch a file from a mirror N times; return a result dict."""
    url = f"{mirror['url']}/{file_path}"
    times, successes = [], 0

    for i in range(samples):
        latency = None
        start = time.monotonic()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="GET"), timeout=timeout
            ) as resp:
                resp.read()
                if resp.status == 200:
                    latency = (time.monotonic() - start) * 1000
                    times.append(latency)
                    successes += 1
        except Exception:
            pass
        if progress_cb:
            progress_cb(i + 1, latency)

    if not times:
        return {**mirror, "ok": "ERR", "avg": 99999,
                "min": 0, "max": 0, "jitter": 0, "loss": 100}

    avg    = sum(times) / len(times)
    jitter = sum(abs(t - avg) for t in times) / len(times)
    return {
        **mirror, "ok": "OK",
        "avg": avg, "min": min(times), "max": max(times),
        "jitter": jitter, "loss": (samples - successes) * 100 // samples,
    }


# ── Formatting ────────────────────────────────────────────────────────────

HDR_FMT  = "%-42s  %3s  %7s  %7s  %7s  %7s  %5s"
ROW_FMT  = "%-42s  %3s  %7s  %7s  %7s  %7s  %5s"
SEPARATOR = "-" * 88


def _fmt_row(r):
    if r["ok"] == "ERR":
        return ROW_FMT % (r["name"], "ERR", "-", "-", "-", "-", "100%")
    return ROW_FMT % (
        r["name"], r["ok"],
        f"{r['avg']:.0f}ms", f"{r['min']:.0f}ms",
        f"{r['max']:.0f}ms", f"{r['jitter']:.0f}ms",
        f"{r['loss']}%",
    )


def _fmt_pending(name):
    return ROW_FMT % (name, "...", "-", "-", "-", "-", "-")


def _fmt_counter(idx, total, nw):
    return f"[{idx + 1:{nw}d}/{total}]"


# ── TTY display state ─────────────────────────────────────────────────────

class Display:
    """Encapsulates all mutable display state."""

    def __init__(self, total, parallel):
        self.total    = total
        self.parallel = parallel
        self.nw       = len(str(total))
        self.prefix   = " " * (2 * self.nw + 4)  # indent matching [XX/YY] width
        self.lock     = threading.Lock()
        self.last_t   = 0.0
        self.throttle = 0.1
        self.mirror_idx: dict[str, int] = {}    # name → row index
        self.active_prog: dict[str, str] = {}   # name → progress string
        self._latencies: dict[str, list] = {}   # name → list of measured ms

    # -- counter / row helpers ---------------------------------------------

    def counter(self, idx):
        return _fmt_counter(idx, self.total, self.nw)

    # -- cursor math -------------------------------------------------------
    # Layout after initial print (cursor sits below last progress slot):
    #   header, separator, row[0]…row[N-1], blank, prog[0]…prog[P-1]
    # Rows from bottom for row i  = (N-i) + 1 + P
    # Rows from bottom for slot j = P - j

    def _up_for_row(self, row_idx):
        return (self.total - row_idx) + 1 + self.parallel

    def _up_for_slot(self, slot):
        return self.parallel - slot

    def _write_at_row(self, row_idx, text):
        up = self._up_for_row(row_idx)
        sys.stdout.write(f"\033[{up}A\r\033[2K{text}\033[{up}B\r")

    def _write_at_slot(self, slot, text):
        up = self._up_for_slot(slot)
        if up == 0:
            sys.stdout.write(f"\r\033[2K{text}")
        else:
            sys.stdout.write(f"\033[{up}A\r\033[2K{text}\033[{up}B\r")

    def _flush_slots(self):
        items = list(self.active_prog.items())
        for slot in range(self.parallel):
            if slot < len(items):
                name, prog = items[slot]
                host = name.split()[0]
                self._write_at_slot(slot, f"  {host:<36s}  {prog}")
            else:
                self._write_at_slot(slot, "")
        sys.stdout.flush()

    # -- public API --------------------------------------------------------

    def print_header(self, mirrors):
        """Print the static header + all pending rows + (TTY) progress slots."""
        print()
        print(self.prefix + HDR_FMT % ("Mirror [region]", "OK?", "Avg", "Min", "Max", "Jitter", "Loss"))
        print(self.prefix + SEPARATOR)
        for idx, m in enumerate(mirrors):
            print(f"{self.counter(idx)} {_fmt_pending(m['name'])}")
        sys.stdout.flush()
        if IS_TTY:
            print()              # blank spacer before progress block
            for _ in range(self.parallel):
                print()

    def on_progress(self, name, sample_idx, samples, latency_ms):
        """Called from worker thread after each sample (throttled)."""
        lats = self._latencies.setdefault(name, [])
        if latency_ms is not None:
            lats.append(latency_ms)
        if lats:
            avg = sum(lats) / len(lats)
            self.active_prog[name] = (
                f"[{sample_idx}/{samples}]  "
                + (f"cur={latency_ms:.0f}ms  " if latency_ms is not None else "timeout  ")
                + f"avg={avg:.0f}ms"
            )
        else:
            self.active_prog[name] = f"[{sample_idx}/{samples}]  timeout"

        now = time.monotonic()
        if now - self.last_t < self.throttle:
            return
        with self.lock:
            self._flush_slots()
            self.last_t = time.monotonic()

    def on_complete(self, result):
        """Called from main thread when a mirror finishes."""
        name = result["name"]
        idx  = self.mirror_idx[name]
        self.active_prog.pop(name, None)
        self._latencies.pop(name, None)
        with self.lock:
            self._write_at_row(idx, f"{self.counter(idx)} {_fmt_row(result)}")
            self._flush_slots()
            self.last_t = time.monotonic()

    def clear_progress_block(self):
        """Erase the blank spacer + all progress slots."""
        lines = 1 + self.parallel
        sys.stdout.write(f"\033[{lines}A")
        for _ in range(lines):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{lines}A")
        for _ in range(lines):
            sys.stdout.write("\033[2K\n")
        sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Connecting to {APT_BASE} ...")
    redirect, mirrors = fetch_apt_info()

    # ── Print combined info section ───────────────────────────────────────
    total = len(mirrors)
    nw    = len(str(total))

    print(f"\n=== apt.armbian.com  ({total} mirrors, suite={args.suite}) ===")

    location = redirect.get("Location", "N/A")
    distance = redirect.get("X-Geo-Distance")
    redir_host = location.split("//")[-1].split("/")[0] if location != "N/A" else "N/A"
    print(f"  Redirect: {redir_host}  ({location})")
    print(f"  Distance: {float(distance)/1000:.0f} km" if distance else "  Distance: N/A")
    print(f"  Server:   {redirect.get('Server', 'N/A')}")

    # ── Set up display ────────────────────────────────────────────────────
    disp = Display(total, args.parallel)
    for idx, m in enumerate(mirrors):
        disp.mirror_idx[m["name"]] = idx

    disp.print_header(mirrors)

    # ── Run tests ─────────────────────────────────────────────────────────
    file_path = f"dists/{args.suite}/InRelease"
    results   = []
    t0        = time.monotonic()

    def make_cb(mirror_name):
        def cb(sample_idx, latency_ms):
            disp.on_progress(mirror_name, sample_idx, args.samples, latency_ms)
        return cb

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(
                    test_mirror, m, file_path, args.samples, args.timeout,
                    make_cb(m["name"]) if IS_TTY else None
                ): m
                for m in mirrors
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if IS_TTY:
                    disp.on_complete(res)
                else:
                    idx = disp.mirror_idx[res["name"]]
                    rows_up = total - idx
                    sys.stdout.write(
                        f"\033[{rows_up}A\r\033[2K"
                        f"{_fmt_counter(idx, total, nw)} {_fmt_row(res)}"
                        f"\033[{rows_up}B\r"
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nInterrupted.\n")

    elapsed = time.monotonic() - t0

    if IS_TTY:
        disp.clear_progress_block()

    print("=" * 88)
    print(f"Test date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Elapsed:   {elapsed:.1f}s  ({args.samples} samples × {total} mirrors, {args.parallel} parallel)")


if __name__ == "__main__":
    main()
