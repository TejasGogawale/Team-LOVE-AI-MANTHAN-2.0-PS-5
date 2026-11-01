"""Bitchat VLM API client for the AI Surveillance pipeline.

Sends surveillance alerts and VLM scene descriptions to Bitchat mesh network.
Your Android device (with Bitchat) must be on the same WiFi as this machine.

Setup:
  1. Open Bitchat on Android -> Settings -> VLM API Settings -> Enable toggle
  2. Note the WiFi IP shown (e.g. 192.168.1.105)
  3. Set BITCHAT_IP in configs/pipeline.yaml  OR  set env var BITCHAT_IP
  4. import and use BitchatAlertClient from this module

Sent to Bitchat mesh:
  - Keyframe image followed by its VLM description (per escalated VLM pass)
  - No detector event logs — only the image + what the VLM actually saw
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import os
import time
import threading
import datetime
from typing import Any

import cv2
import numpy as np
import requests


def _frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR numpy frame as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()


def _prepare_image(frame: np.ndarray, max_dim: int = 480, quality: int = 60,
                   max_bytes: int = 24576) -> bytes:
    """Downscale + JPEG-encode a frame for reliable mesh delivery.

    Bitchat transfers images as fragmented mesh packets; smaller images mean
    fewer fragments and a much higher chance of successful delivery.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    jpeg = buf.tobytes()
    while len(jpeg) > max_bytes and max(frame.shape[:2]) > 96:
        frame = cv2.resize(frame, (max(1, int(frame.shape[1] * 0.75)),
                                  max(1, int(frame.shape[0] * 0.75))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Failed to encode frame to JPEG")
        jpeg = buf.tobytes()
    if len(jpeg) > max_bytes:
        raise ValueError("Image exceeds configured mesh payload budget")
    return jpeg


class BitchatAlertClient:
    """HTTP client for Bitchat's VLM API.

    All send methods are fire-and-forget (run in a daemon thread) so they
    never block the main surveillance loop.

    Args:
        ip:      Android device WiFi IP (e.g. "192.168.1.105")
        port:    Bitchat VLM API port (default 8765)
        channel: Bitchat channel to broadcast to (default "surveillance")
        timeout: HTTP request timeout in seconds
        rate_limit_s: Minimum seconds between messages (matches Bitchat default 5s)
    """

    def __init__(
        self,
        ip: str,
        port: int = 8765,
        channel: str | None = None,
        timeout: float = 10.0,
        rate_limit_s: float = 5.5,
        connect_timeout: float = 2.0,
        max_queue: int = 8,
        max_age_s: float = 120.0,
        retry_backoff_s: float = 5.0,
        retry_max_s: float = 30.0,
        reconnect_interval_s: float = 0.0,
        image_max_dim: int = 320,
        image_quality: int = 50,
        image_max_bytes: int = 6144,
    ):
        self.image_max_dim = image_max_dim
        self.image_quality = image_quality
        self.image_max_bytes = image_max_bytes
        self.base_url    = f"http://{ip}:{port}"
        self.channel     = channel
        self.timeout     = timeout
        self.rate_limit_s = rate_limit_s
        self.connect_timeout = connect_timeout
        self.max_queue = max(2, max_queue)
        self.max_age_s = max_age_s
        self.retry_backoff_s = retry_backoff_s
        self.retry_max_s = retry_max_s
        self.reconnect_interval_s = max(0.0, reconnect_interval_s)
        self._connected: bool | None = None
        self._connection_reason: str | None = None
        self._connection_lock = threading.Lock()
        self._next_connection_check = time.perf_counter() + self.reconnect_interval_s
        self._stop = threading.Event()

        self._lock         = threading.RLock()
        self._last_sent_t  = 0.0
        self._queue = deque(maxlen=self.max_queue)
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="bitchat-sender"
        )
        self._worker_thread.start()

        channel_info = f"channel=#{channel}" if channel else "default_destination"
        print(
            f"[bitchat] client initialized  url={self.base_url}  "
            f"{channel_info}  rate_limit={rate_limit_s}s"
        )

    def check_connection(self) -> bool:
        """Blocking check. Returns True if Bitchat API is reachable and enabled."""
        with self._connection_lock:
            data = {}
            try:
                r = requests.get(f"{self.base_url}/status", timeout=(self.connect_timeout, 4))
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict) or data.get("status") != "ok":
                    reason = "invalid API status response"
                elif not data.get("api_enabled", False):
                    reason = "phone API is disabled"
                elif data.get("mesh_running") is False:
                    reason = "phone Bluetooth mesh is stopped"
                else:
                    reason = "connected"
            except (requests.RequestException, ValueError) as e:
                reason = f"phone API unreachable ({type(e).__name__})"
            connected = reason == "connected"
            if connected != self._connected or reason != self._connection_reason:
                if connected:
                    print(f"[bitchat] connected url={self.base_url} "
                          f"peers={data.get('peers_count', 0)} mesh={data.get('mesh_running')}")
                else:
                    recovery = (f"; rechecking every {self.reconnect_interval_s:g}s"
                                if self.reconnect_interval_s else "")
                    print(f"[bitchat] offline: {reason} at {self.base_url}{recovery}. "
                          "Check the exact address on the phone's VLM API screen.")
            self._connected = connected
            self._connection_reason = reason
            self._next_connection_check = time.perf_counter() + self.reconnect_interval_s
            return connected

    def send_scene(self, scene_text: str, frame: np.ndarray | None = None) -> None:
        """Send a keyframe image then its VLM description — nothing else.

        Order matters: the image goes first so the user sees what the camera
        captured, then the description text is sent after Bitchat's own 5s API
        rate window has passed (messages sent sooner are silently dropped).
        """
        with getattr(self, "_lock", nullcontext()):
            if getattr(self, "_stop", None) is not None and self._stop.is_set():
                return
            if frame is not None:
                # Reserve both slots together; sender cannot interleave scenes.
                if hasattr(self, "_queue"):
                    while len(self._queue) > self.max_queue - 2:
                        self._queue.popleft()
                self._enqueue("/send/image", {"caption": scene_text}, frame)
                self._enqueue("/send/text", {"text": scene_text}, None, delay_s=0.0)
            else:
                self._enqueue("/send/text", {"text": scene_text}, None)

    def send_alert(
        self,
        event_type: str,
        detail: str,
        frame: np.ndarray | None = None,
        priority: bool = False,
    ) -> None:
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        msg   = f"[{event_type.upper()}] {ts} - {detail}"
        
        if frame is not None:
            self._enqueue("/send/image", {"caption": msg}, frame, skip_rate=priority)
        else:
            self._enqueue("/send/text", {"text": msg}, None, skip_rate=priority)

    def _enqueue(
        self,
        endpoint: str,
        data: dict,
        frame: np.ndarray | None,
        skip_rate: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            frame_copy = frame.copy() if frame is not None else None
            self._queue.append((endpoint, data, frame_copy, skip_rate, delay_s,
                                time.perf_counter()))
            if endpoint == "/send/image":
                print(f"[bitchat] ENQUEUE IMAGE: queue size={len(self._queue)}, frame={'YES' if frame_copy is not None else 'NO'}")

    def _worker(self) -> None:
        """Background sender thread - respects rate limiting."""
        pending = None
        failures = 0
        while not self._stop.is_set():
            # Status recovery runs even with no VLM output or queued alert.
            # Keep frames queued until both HTTP API and mesh are ready.
            if self.reconnect_interval_s:
                now = time.perf_counter()
                if self._connected is None or now >= self._next_connection_check:
                    self.check_connection()
                if self._connected is False:
                    self._stop.wait(max(0.1, self._next_connection_check - time.perf_counter()))
                    continue
            item = None
            with self._lock:
                if pending is not None:
                    item, pending = pending, None
                elif self._queue:
                    item = self._queue.popleft()
            if item is None:
                self._stop.wait(0.1)
                continue

            endpoint, data, frame, skip_rate, delay_s, created_t = item
            if time.perf_counter() - created_t > self.max_age_s:
                continue

            print(f"[bitchat] WORKER: Processing {endpoint}, has_frame={'YES' if frame is not None else 'NO'}")

            # Rate limiting - Bitchat default is 5000ms between messages
            # The phone applies its rate limit even to priority alerts.
            elapsed = time.perf_counter() - self._last_sent_t
            if elapsed < self.rate_limit_s:
                if self._stop.wait(self.rate_limit_s - elapsed):
                    break

            # Delayed send — keeps the phone's own 5s API rate window free
            # (an image sent sooner is silently dropped by the phone).
            if delay_s > 0:
                if self._stop.wait(delay_s):
                    break
            if time.perf_counter() - created_t > self.max_age_s:
                continue

            try:
                self._send(endpoint, data, frame)
                self._last_sent_t = time.perf_counter()
                failures = 0
                print("[bitchat] API accepted message for mesh delivery")
            except (requests.ConnectionError, requests.Timeout) as e:
                failures += 1
                pending = item
                delay = min(self.retry_max_s,
                            self.retry_backoff_s * 2 ** min(failures - 1, 6))
                print(f"[bitchat] Phone API unreachable at {self.base_url}; "
                      f"retry in {delay:.0f}s. Check phone Wi-Fi IP and VLM API toggle. ({type(e).__name__})")
                if self.reconnect_interval_s:
                    self._connected = False
                    self._next_connection_check = time.perf_counter() + delay
                self._stop.wait(delay)
            except Exception as e:
                self._last_sent_t = time.perf_counter()
                print(f"[bitchat] send error: {e}")

    def close(self) -> None:
        """Stop retrying and release queued camera frames on pipeline exit."""
        self._stop.set()
        with self._lock:
            self._queue.clear()
        self._worker_thread.join(timeout=self.connect_timeout + self.timeout + 1)

    def _send(
        self,
        endpoint: str,
        data: dict,
        frame: np.ndarray | None,
    ) -> None:
        print(f"[bitchat] _send called: endpoint={endpoint}, frame={'YES' if frame is not None else 'NO'}")
        
        if endpoint == "/send/image" and frame is not None:
            url = self.base_url + "/send/image"
            jpeg = _prepare_image(
                frame,
                max_dim=getattr(self, "image_max_dim", 320),
                quality=getattr(self, "image_quality", 50),
                max_bytes=getattr(self, "image_max_bytes", 6144),
            )
            caption = data.get("caption") or data.get("description") or data.get("text", "")
            print(f"[bitchat] Sending image ({len(jpeg)} bytes) with caption: {caption[:50]}")
            files = {"image": ("surveillance_frame.jpg", jpeg, "image/jpeg")}
            form_data = {}
            if caption:
                form_data["caption"] = caption
            if self.channel:
                form_data["channel"] = self.channel
            # Propagate failures to the single retry worker without traceback spam.
            r = requests.post(url, files=files, data=form_data,
                              timeout=(getattr(self, "connect_timeout", 2.0), self.timeout))
        else:
            url = self.base_url + "/send/text"
            text_payload = data.get("text") or data.get("description", "")
            print(f"[bitchat] Sending text: {text_payload[:80]}")
            
            payload = {"text": text_payload}
            if self.channel:
                payload["channel"] = self.channel
                
            r = requests.post(url, json=payload,
                              timeout=(getattr(self, "connect_timeout", 2.0), self.timeout))

        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        else:
            resp = r.json()
            if resp.get("status") != "ok":
                raise RuntimeError(f"API rejected message: {resp}")


def build_client_from_config(cfg: dict | None = None) -> BitchatAlertClient | None:
    """Build a BitchatAlertClient from pipeline config or env vars.

    Config key: pipeline.yaml -> bitchat:
      ip:       "192.168.1.105"   # Android WiFi IP
      port:     8765
      channel:  "surveillance"
      enabled:  true

    Env override: BITCHAT_IP=192.168.1.105

    Returns None if bitchat is disabled or IP not set.
    """
    bc_cfg = (cfg or {}).get("bitchat", {})

    if not bc_cfg.get("enabled", False):
        return None

    cfg_ip = str(bc_cfg.get("ip", "") or "")
    env_ip = os.environ.get("BITCHAT_IP", "")
    ip = env_ip or cfg_ip
    if env_ip and env_ip != cfg_ip:
        print(
            f"[bitchat] WARN: env BITCHAT_IP={env_ip} overrides pipeline.yaml "
            f"bitchat.ip={cfg_ip or '(unset)'}. Clear it with "
            "`Remove-Item Env:BITCHAT_IP` if that is not intended."
        )
    if not ip:
        print(
            "[bitchat] WARN: bitchat.enabled=true but no IP configured. "
            "Set bitchat.ip in pipeline.yaml or BITCHAT_IP env var."
        )
        return None

    channel = bc_cfg.get("channel", None)
    if channel:
        channel = str(channel)
    
    client = BitchatAlertClient(
        ip=ip,
        port=int(bc_cfg.get("port", 8765)),
        channel=channel,
        rate_limit_s=float(bc_cfg.get("rate_limit_s", 5.5)),
        timeout=float(bc_cfg.get("timeout", 10.0)),
        connect_timeout=float(bc_cfg.get("connect_timeout", 2.0)),
        max_queue=int(bc_cfg.get("max_queue", 8)),
        max_age_s=float(bc_cfg.get("max_age_s", 120.0)),
        retry_backoff_s=float(bc_cfg.get("retry_backoff_s", 5.0)),
        retry_max_s=float(bc_cfg.get("retry_max_s", 30.0)),
        reconnect_interval_s=float(bc_cfg.get("reconnect_interval_s", 10.0)),
        image_max_dim=int(bc_cfg.get("image_max_dim", 320)),
        image_quality=int(bc_cfg.get("image_quality", 50)),
        image_max_bytes=int(bc_cfg.get("image_max_bytes", 6144)),
    )

    client.check_connection()

    return client
