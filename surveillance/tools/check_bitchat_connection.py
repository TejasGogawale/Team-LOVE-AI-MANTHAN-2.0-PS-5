"""Read-only phone API diagnostic. Run with the project's Python environment."""
import sys
import socket
from pathlib import Path

import requests
import yaml


def main():
    import os
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    bc = cfg.get("bitchat", {})
    ip = os.environ.get("BITCHAT_IP") or bc.get("ip")
    url = f"http://{ip}:{bc.get('port', 8765)}/status"
    print(f"Checking {url} (no alerts sent)")
    try:
        with socket.create_connection((ip, int(bc.get("port", 8765))), timeout=2):
            pass
    except OSError as exc:
        print(f"Cannot open phone API port: {type(exc).__name__}")
        print("If the phone responds to ping, its IP is reachable but the API port is blocked or unavailable.")
        print("On the phone, toggle BitChat's VLM API off and on, and keep BitChat in the foreground.")
        print("The API must accept Wi-Fi connections; a listener bound only to 127.0.0.1 cannot be accessed from this PC.")
        return 1
    try:
        response = requests.get(url, timeout=(2, 4))
        response.raise_for_status()
        status = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Phone API unreachable: {type(exc).__name__}")
        print("Open BitChat > VLM API Settings on the phone. Enable the API and check its Wi-Fi IP.")
        print("Phone and PC must be on a network allowing communication between devices.")
        print("Update bitchat.ip in configs/pipeline.yaml or set BITCHAT_IP if the phone IP changed.")
        return 1
    print(f"API enabled: {status.get('api_enabled')}; mesh running: {status.get('mesh_running')}; peers: {status.get('peers_count')}")
    if not status.get("api_enabled") or status.get("mesh_running") is False:
        print("Enable the phone API and Bluetooth mesh in BitChat settings.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
