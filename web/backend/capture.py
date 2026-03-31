"""
capture.py — Scapy-based flow tracker that extracts UNSW-NB15-compatible features.

NOTE on ct_* features:
  ct_srv_src, ct_dst_ltm, ct_src_ltm, ct_srv_dst, ct_src_dport_ltm,
  ct_dst_sport_ltm, ct_dst_src_ltm require counting recent connections
  within a sliding time window. We implement a ConnectionHistory class
  that approximates these using a 60-second rolling window.
  These are estimates, not exact Argus reproductions — acceptable for a demo.
"""

import threading
import time

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, AsyncSniffer

from web.backend.config import FLOW_TIMEOUT_SEC, EVENT_LOG_MAXLEN

# ── Service / protocol maps ───────────────────────────────────────────────────

_SERVICE_MAP = {
    80: "http", 8080: "http", 443: "https", 8443: "https",
    21: "ftp",  22: "ssh",    23: "telnet", 25: "smtp",
    53: "dns",  110: "pop3",  143: "imap",  3306: "mysql",
    5432: "postgres",
}

_PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}


# ── Connection history for ct_* features ─────────────────────────────────────

class ConnectionHistory:
    """Rolling 60-second window of completed flow records."""

    WINDOW = 60.0

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list = []

    def add(self, src_ip, dst_ip, sport, dport, service, proto, ts):
        with self._lock:
            self._records.append({
                "ts": ts, "src": src_ip, "dst": dst_ip,
                "sport": sport, "dport": dport,
                "service": service, "proto": proto,
            })
            cutoff = ts - self.WINDOW
            self._records = [r for r in self._records if r["ts"] >= cutoff]

    def _recent(self, ts):
        cutoff = ts - self.WINDOW
        return [r for r in self._records if r["ts"] >= cutoff]

    def ct_srv_src(self, src_ip, service, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["src"] == src_ip and r["service"] == service)

    def ct_dst_ltm(self, dst_ip, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["dst"] == dst_ip)

    def ct_src_ltm(self, src_ip, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["src"] == src_ip)

    def ct_srv_dst(self, dst_ip, service, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["dst"] == dst_ip and r["service"] == service)

    def ct_src_dport_ltm(self, src_ip, dport, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["src"] == src_ip and r["dport"] == dport)

    def ct_dst_sport_ltm(self, dst_ip, sport, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["dst"] == dst_ip and r["sport"] == sport)

    def ct_dst_src_ltm(self, src_ip, dst_ip, ts):
        with self._lock:
            return sum(1 for r in self._recent(ts) if r["src"] == src_ip and r["dst"] == dst_ip)


# ── Per-flow accumulator ──────────────────────────────────────────────────────

class Flow:
    def __init__(self, key: tuple, start: float):
        self.key = key          # (src_ip, dst_ip, sport, dport, proto_num)
        self.start = start
        self.last_seen = start

        self.spkts = 0;  self.dpkts = 0
        self.sbytes = 0; self.dbytes = 0
        self.sloss = 0;  self.dloss = 0

        self.sttl = None; self.dttl = None
        self.swin = 0;    self.dwin = 0
        self.stcpb = 0;   self.dtcpb = 0

        self.src_times: list = []
        self.dst_times: list = []
        self.src_sizes: list = []
        self.dst_sizes: list = []

        self.syn_time    = None
        self.synack_time = None
        self.ack_time    = None
        self.fin_seen    = False
        self.state       = "CON"

        self.http_methods      = 0
        self.response_body_len = 0

    def add_packet(self, pkt, direction: str):
        now = time.time()
        self.last_seen = now
        size = len(pkt)

        if direction == "src":
            self.spkts += 1
            self.sbytes += size
            self.src_times.append(now)
            self.src_sizes.append(size)
            if pkt.haslayer(IP) and self.sttl is None:
                self.sttl = pkt[IP].ttl
        else:
            self.dpkts += 1
            self.dbytes += size
            self.dst_times.append(now)
            self.dst_sizes.append(size)
            if pkt.haslayer(IP) and self.dttl is None:
                self.dttl = pkt[IP].ttl

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            flags = str(tcp.flags)

            if direction == "src":
                self.swin = tcp.window
                if self.stcpb == 0:
                    self.stcpb = tcp.seq
            else:
                self.dwin = tcp.window
                if self.dtcpb == 0:
                    self.dtcpb = tcp.seq

            if "S" in flags and "A" not in flags:
                self.syn_time = now
                self.state = "REQ"
            elif "S" in flags and "A" in flags:
                self.synack_time = now
                self.state = "EST"
            elif "A" in flags and self.synack_time and not self.ack_time:
                self.ack_time = now
            if "F" in flags or "R" in flags:
                self.fin_seen = True
                self.state = "FIN" if "F" in flags else "RST"

    @staticmethod
    def _mean_interarrival(times: list) -> float:
        if len(times) < 2:
            return 0.0
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        return float(np.mean(diffs)) * 1000.0

    @staticmethod
    def _jitter(times: list) -> float:
        if len(times) < 3:
            return 0.0
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        return float(np.std(diffs)) * 1000.0

    def extract_features(self, history: ConnectionHistory) -> dict:
        src_ip, dst_ip, sport, dport, proto_num = self.key
        dur = max(self.last_seen - self.start, 1e-6)
        ts  = self.last_seen

        proto   = _PROTO_MAP.get(proto_num, "other")
        service = _SERVICE_MAP.get(dport, _SERVICE_MAP.get(sport, "-"))

        sinpkt = self._mean_interarrival(self.src_times)
        dinpkt = self._mean_interarrival(self.dst_times)
        sjit   = self._jitter(self.src_times)
        djit   = self._jitter(self.dst_times)

        total_pkts = self.spkts + self.dpkts
        rate   = total_pkts / dur
        sload  = (self.sbytes * 8.0) / dur
        dload  = (self.dbytes * 8.0) / dur
        smean  = float(np.mean(self.src_sizes)) if self.src_sizes else 0.0
        dmean  = float(np.mean(self.dst_sizes)) if self.dst_sizes else 0.0

        synack = 0.0; ackdat = 0.0
        if self.syn_time and self.synack_time:
            synack = (self.synack_time - self.syn_time) * 1000.0
        if self.synack_time and self.ack_time:
            ackdat = (self.ack_time - self.synack_time) * 1000.0
        tcprtt = synack + ackdat

        history.add(src_ip, dst_ip, sport, dport, service, proto, ts)
        ct_srv_src       = history.ct_srv_src(src_ip, service, ts)
        ct_dst_ltm       = history.ct_dst_ltm(dst_ip, ts)
        ct_src_ltm       = history.ct_src_ltm(src_ip, ts)
        ct_srv_dst       = history.ct_srv_dst(dst_ip, service, ts)
        ct_src_dport_ltm = history.ct_src_dport_ltm(src_ip, dport, ts)
        ct_dst_sport_ltm = history.ct_dst_sport_ltm(dst_ip, sport, ts)
        ct_dst_src_ltm   = history.ct_dst_src_ltm(src_ip, dst_ip, ts)

        return {
            "dur": dur, "proto": proto, "service": service, "state": self.state,
            "spkts": self.spkts, "dpkts": self.dpkts,
            "sbytes": self.sbytes, "dbytes": self.dbytes,
            "rate": rate,
            "sttl": self.sttl or 64, "dttl": self.dttl or 64,
            "sload": sload, "dload": dload,
            "sloss": self.sloss, "dloss": self.dloss,
            "sinpkt": sinpkt, "dinpkt": dinpkt,
            "sjit": sjit, "djit": djit,
            "swin": self.swin, "stcpb": self.stcpb, "dtcpb": self.dtcpb, "dwin": self.dwin,
            "tcprtt": tcprtt, "synack": synack, "ackdat": ackdat,
            "smean": smean, "dmean": dmean,
            "trans_depth": 0, "response_body_len": self.response_body_len,
            "ct_srv_src": ct_srv_src, "ct_dst_ltm": ct_dst_ltm,
            "ct_src_dport_ltm": ct_src_dport_ltm, "ct_dst_sport_ltm": ct_dst_sport_ltm,
            "ct_dst_src_ltm": ct_dst_src_ltm,
            "is_ftp_login": 1 if (dport == 21 or sport == 21) else 0,
            "ct_ftp_cmd": 0,
            "ct_flw_http_mthd": self.http_methods,
            "ct_src_ltm": ct_src_ltm, "ct_srv_dst": ct_srv_dst,
            "is_sm_ips_ports": 1 if (src_ip == dst_ip or sport == dport) else 0,
        }


# ── Event log ─────────────────────────────────────────────────────────────────

class EventLog:
    def __init__(self, maxlen: int = EVENT_LOG_MAXLEN):
        self._lock = threading.Lock()
        self._events: list = []
        self._maxlen = maxlen

    def append(self, item: dict):
        with self._lock:
            self._events.append(item)
            if len(self._events) > self._maxlen:
                self._events = self._events[-self._maxlen:]

    def since(self, cursor: int) -> tuple:
        with self._lock:
            total = len(self._events)
            return self._events[cursor:], total


# ── Capture manager ───────────────────────────────────────────────────────────

class CaptureManager:
    def __init__(self, predictor, event_log: EventLog):
        self.predictor  = predictor
        self.event_log  = event_log
        self.history    = ConnectionHistory()
        self._flows: dict = {}
        self._lock      = threading.Lock()
        self._sniffer   = None
        self._cleanup   = None
        self.running    = False

    def start(self, iface=None):
        self.running = True
        kwargs = {"prn": self._on_packet, "store": False, "filter": "ip"}
        if iface:
            kwargs["iface"] = iface
        self._sniffer = AsyncSniffer(**kwargs)
        self._sniffer.start()
        self._cleanup = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup.start()

    def stop(self):
        self.running = False
        if self._sniffer:
            try:
                self._sniffer.stop()
            except Exception:
                pass

    def _flow_keys(self, pkt):
        if not pkt.haslayer(IP):
            return None, None
        ip = pkt[IP]
        sport = dport = 0
        if pkt.haslayer(TCP):
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        fwd = (ip.src, ip.dst, sport, dport, ip.proto)
        rev = (ip.dst, ip.src, dport, sport, ip.proto)
        return fwd, rev

    def _on_packet(self, pkt):
        if not pkt.haslayer(IP):
            return
        fwd, rev = self._flow_keys(pkt)
        now = time.time()

        with self._lock:
            if fwd in self._flows:
                flow = self._flows[fwd]
                flow.add_packet(pkt, "src")
                if flow.fin_seen:
                    self._finalize(fwd)
            elif rev in self._flows:
                flow = self._flows[rev]
                flow.add_packet(pkt, "dst")
                if flow.fin_seen:
                    self._finalize(rev)
            else:
                flow = Flow(fwd, now)
                flow.add_packet(pkt, "src")
                self._flows[fwd] = flow

    def _finalize(self, key: tuple):
        flow = self._flows.pop(key, None)
        if flow is None:
            return
        try:
            features = flow.extract_features(self.history)
            df = pd.DataFrame([features])
            results = self.predictor.predict(df)
            if not results:
                return
            pred = results[0]
            # Accept any of the label key variants inference.py might return
            label = (
                pred.get("predicted_class")
                or pred.get("label")
                or pred.get("class")
                or "Unknown"
            )
            confidence = float(pred.get("confidence", 0.0))
            src_ip, dst_ip, sport, dport, proto_num = flow.key
            self.event_log.append({
                "timestamp":  flow.last_seen,
                "src":        f"{src_ip}:{sport}",
                "dst":        f"{dst_ip}:{dport}",
                "proto":      _PROTO_MAP.get(proto_num, str(proto_num)),
                "label":      label,
                "confidence": round(confidence * 100, 1),
                "is_attack":  label != "Normal",
                "bytes":      flow.sbytes + flow.dbytes,
                "duration":   round(flow.last_seen - flow.start, 3),
            })
        except Exception as e:
            print(f"[capture] finalize error: {e}")

    def _cleanup_loop(self):
        while self.running:
            time.sleep(10)
            cutoff = time.time() - FLOW_TIMEOUT_SEC
            with self._lock:
                expired = [k for k, f in self._flows.items() if f.last_seen < cutoff]
                for k in expired:
                    self._finalize(k)
                    