# VLM performance and BitChat connectivity

Restart the running pipeline to load these changes:

```powershell
.\.venv\Scripts\python.exe -m pipeline.main_loop
```

The VLM shares the GPU with detection. Escalated inference now waits at least
10 seconds after completion (also after errors), with periodic descriptions
every 30 seconds when idle. Vision input is capped at 200704 pixels and each
prompt at 128 generated tokens. Smaller input reduces compute but can make tiny
objects harder to interpret. Both limits can be adjusted in `configs/vlm.yaml`.
Relation inference does not overlap VLM jobs and submits at most once every
5 seconds. Face processing loads detection and recognition modules only.

Low FPS warnings now include the three slowest measured stages and whether a
VLM job is active. Average stage times are per processed frame; they do not
include all unwrapped display/storage work. The 20 FPS threshold remains a
target, not a measured guarantee.

The PC connects to the phone's Wi-Fi HTTP API on port 8765. The phone forwards
messages to its Bluetooth mesh. An HTTP connection timeout means the phone API
cannot be reached; it does not establish a Bluetooth failure.

Check connectivity without sending any alerts:

```powershell
.\.venv\Scripts\python.exe tools/check_bitchat_connection.py
```

If this fails, verify the IP shown in BitChat's VLM API Settings, keep BitChat
open, and put the phone and PC on a network that permits communication between
devices. Update `bitchat.ip` in `configs/pipeline.yaml` or set `BITCHAT_IP` when
the IP changes. The API toggle alone does not guarantee network reachability.

The sender keeps at most eight queued messages, retries connection failures
with increasing delays from 5 to 30 seconds, and discards messages older than
120 seconds. Images are resized to fit a 24 KiB payload budget. All messages,
including priority alerts, respect the phone's rate window. An API acceptance
confirms submission to the phone, not receipt by a remote Bluetooth peer.

Phone status is checked every 10 seconds in the sender worker even when no VLM
scene is generated. When the API is disabled, unreachable, or reports a stopped
mesh, sends pause. When the configured endpoint becomes ready, queued alerts
resume automatically. Offline status is logged when its reason changes, rather
than printing repeated requests tracebacks. The configured address is retained;
the client does not redirect camera images to another discovered device.
