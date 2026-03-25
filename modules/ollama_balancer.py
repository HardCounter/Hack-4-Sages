"""
Active load balancer for multiple Ollama instances.

Uses **session-scoped** routing: ``session_scope()`` reserves a host for
the entire duration of an agent chain so that every LLM call within that
chain hits the same GPU while ``_in_flight`` stays elevated.  A second
concurrent request from another device therefore lands on the *other* host
and both GPUs work truly in parallel.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_HOSTS: List[str] = [
    "http://192.168.0.104:11434",
    "http://localhost:11434",
]

_MONITOR_INTERVAL = 10.0

_thread_local = threading.local()


class OllamaBalancer:
    """Session-scoped least-loaded balancer with active health monitoring."""

    def __init__(self, hosts: Optional[List[str]] = None):
        self.hosts = list(hosts or OLLAMA_HOSTS)
        self._lock = threading.Lock()
        self._healthy: Dict[str, bool] = {h: True for h in self.hosts}
        self._in_flight: Dict[str, int] = {h: 0 for h in self.hosts}
        self._total: Dict[str, int] = {h: 0 for h in self.hosts}
        self._rr_index = 0

        self._stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None
        self._start_monitor()

    # ── health probing ────────────────────────────────────────────────

    def _probe(self, host: str) -> bool:
        try:
            import ollama
            client = ollama.Client(host=host)
            client.list()
            return True
        except Exception:
            return False

    def _monitor_loop(self) -> None:
        while not self._stop.wait(_MONITOR_INTERVAL):
            for host in self.hosts:
                alive = self._probe(host)
                with self._lock:
                    was = self._healthy[host]
                    self._healthy[host] = alive
                if alive and not was:
                    print(f"[LB HEALTH] {host} RECOVERED")
                    logger.info("Ollama host %s recovered", host)
                elif not alive and was:
                    print(f"[LB HEALTH] {host} DOWN")
                    logger.warning("Ollama host %s is down", host)

    def _start_monitor(self) -> None:
        self._monitor = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ollama-health",
        )
        self._monitor.start()

    # ── host selection ────────────────────────────────────────────────

    def _pick_fresh(self, pool: List[str]) -> str:
        """Least-loaded with round-robin tie-break."""
        min_flight = min(self._in_flight[h] for h in pool)
        tied = [h for h in pool if self._in_flight[h] == min_flight]
        host = tied[self._rr_index % len(tied)]
        self._rr_index += 1
        return host

    # ── session-scoped reservation ────────────────────────────────────

    @contextmanager
    def session_scope(self):
        """Reserve a host for the entire agent chain running in this thread.

        While the scope is active every ``next_host()`` call returns the
        reserved host and ``release()`` is a no-op — the single
        ``_in_flight`` increment made here stays until the scope exits.
        This guarantees a concurrent request from another thread sees the
        host as busy and gets routed to a different GPU.
        """
        with self._lock:
            pool = [h for h in self.hosts if self._healthy[h]]
            if not pool:
                pool = list(self.hosts)
            host = self._pick_fresh(pool)
            self._in_flight[host] += 1
            self._total[host] += 1
            _thread_local.reserved_host = host
            print(f"[LB] session_scope ACQUIRED {host}  "
                  f"in_flight={dict(self._in_flight)}")
        try:
            yield host
        finally:
            _thread_local.reserved_host = None
            with self._lock:
                self._in_flight[host] = max(0, self._in_flight[host] - 1)
                print(f"[LB] session_scope RELEASED {host}  "
                      f"in_flight={dict(self._in_flight)}")

    def next_host(self) -> str:
        """Return a host for this call.

        If the current thread owns a ``session_scope``, the reserved host
        is returned without touching ``_in_flight`` (it is already held).
        Otherwise falls back to least-loaded selection.
        """
        reserved: Optional[str] = getattr(_thread_local, "reserved_host", None)

        with self._lock:
            if reserved and reserved in self.hosts:
                if self._healthy.get(reserved, False):
                    self._total[reserved] += 1
                    print(f"[LB] -> {reserved}  (session-reserved)  "
                          f"in_flight={dict(self._in_flight)}")
                    return reserved

            pool = [h for h in self.hosts if self._healthy[h]]
            if not pool:
                pool = list(self.hosts)

            host = self._pick_fresh(pool)
            self._in_flight[host] += 1
            self._total[host] += 1
            print(f"[LB] -> {host}  (fresh pick)  "
                  f"in_flight={dict(self._in_flight)}")
            return host

    def release(self, host: str) -> None:
        """Decrement in-flight — skipped for the reserved host inside a ``session_scope``."""
        if getattr(_thread_local, "reserved_host", None) == host:
            return
        with self._lock:
            if host in self._in_flight:
                self._in_flight[host] = max(0, self._in_flight[host] - 1)

    @contextmanager
    def use_host(self):
        """One-shot context manager for callers outside a session scope."""
        host = self.next_host()
        try:
            yield host
        finally:
            self.release(host)

    # ── manual overrides ──────────────────────────────────────────────

    def mark_unhealthy(self, host: str) -> None:
        with self._lock:
            if host in self._healthy:
                self._healthy[host] = False
                print(f"[LB] MARKED UNHEALTHY: {host}")
                logger.warning("Marked Ollama host %s as unhealthy", host)

    def mark_healthy(self, host: str) -> None:
        with self._lock:
            if host in self._healthy:
                self._healthy[host] = True

    # ── introspection ─────────────────────────────────────────────────

    def get_healthy_hosts(self) -> List[str]:
        with self._lock:
            return [h for h, ok in self._healthy.items() if ok]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                host: {
                    "healthy": self._healthy[host],
                    "in_flight": self._in_flight[host],
                    "total_requests": self._total[host],
                }
                for host in self.hosts
            }

    def check_all(self) -> Dict[str, bool]:
        results = {}
        for host in self.hosts:
            alive = self._probe(host)
            with self._lock:
                self._healthy[host] = alive
            results[host] = alive
        return results


# ── module singleton ──────────────────────────────────────────────────

_balancer: Optional[OllamaBalancer] = None
_init_lock = threading.Lock()


def get_balancer() -> OllamaBalancer:
    global _balancer
    if _balancer is None:
        with _init_lock:
            if _balancer is None:
                _balancer = OllamaBalancer()
    return _balancer


def next_host() -> str:
    return get_balancer().next_host()


def get_reserved_host() -> Optional[str]:
    """Return the host reserved by the current thread's ``session_scope``, if any."""
    return getattr(_thread_local, "reserved_host", None)
