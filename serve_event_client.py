import os
import time
from typing import Optional

import requests


class ServeEventClient:
    """Minimal HTTP client to send ServeEvent(s) to backend.

    Expected payload shape (per backend's registerServe):
      { "timestamp": <epoch_ms>, "productCode": <str> }

    The backend assigns clientId internally (default or via context).
    """

    def __init__(self, base_url: Optional[str] = None, timeout_seconds: float = 5.0):
        # Base URL like: http://localhost:8080
        self.base_url = (base_url or os.getenv("SERVE_API_URL", "")).rstrip("/")
        self.timeout_seconds = timeout_seconds

        # Endpoint path; make configurable with env as well
        # Example default: /api/serve
        self.endpoint_path = os.getenv("SERVE_API_PATH", "/api/serve")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def _post(self, json_payload):
        url = f"{self.base_url}{self.endpoint_path}"
        resp = requests.post(url, json=json_payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp

    def send_single(self, product_code: str, timestamp_ms: Optional[int] = None) -> bool:
        if not self.is_configured():
            print("ServeEventClient not configured: SERVE_API_URL is empty; skipping send")
            return False

        payload = {
            "timestamp": int(timestamp_ms if timestamp_ms is not None else time.time() * 1000),
            "productCode": product_code,
        }

        try:
            r = self._post(payload)
            print(f"Sent ServeEvent: {payload} -> {r.status_code}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to send ServeEvent {payload}: {exc}")
            return False

    def send_many(self, events: list[dict]) -> int:
        """Send multiple events; falls back to per-event if bulk fails.

        events: list of {"timestamp": int(ms), "productCode": str}
        Returns number of successfully sent events.
        """
        if not self.is_configured():
            print("ServeEventClient not configured: SERVE_API_URL is empty; skipping send")
            return 0

        url = f"{self.base_url}{self.endpoint_path}"

        # Try bulk first if backend supports list payloads
        try:
            r = requests.post(url, json=events, timeout=self.timeout_seconds)
            r.raise_for_status()
            print(f"Sent {len(events)} ServeEvents in bulk -> {r.status_code}")
            return len(events)
        except Exception as exc:  # noqa: BLE001
            print(f"Bulk send failed ({exc}); falling back to single sends...")

        # Fallback to single sends
        sent = 0
        for ev in events:
            try:
                r = requests.post(url, json=ev, timeout=self.timeout_seconds)
                r.raise_for_status()
                sent += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to send ServeEvent {ev}: {exc}")
        print(f"Sent {sent}/{len(events)} ServeEvents via per-event fallback")
        return sent


