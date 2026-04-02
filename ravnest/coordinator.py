"""
Ravnest Cluster Coordinator

Lightweight HTTP service that manages dynamic node registration,
hardware profiling, and cluster lifecycle. Nodes register on startup,
send heartbeats, and get restarted when the cluster changes.

Usage:
    ravnest coordinator --port 8080
    ravnest worker --coordinator http://master:8080
"""

import json
import os
import time
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from .hardware import get_hardware_info, compute_proportions
except ImportError:
    from hardware import get_hardware_info, compute_proportions


class NodeRegistration(BaseModel):
    node_id: str  # unique identifier (hostname or tailscale IP)
    hardware: dict  # from get_hardware_info()


class ClusterState:
    def __init__(self, min_nodes=2, heartbeat_timeout=60):
        self.nodes: Dict[str, dict] = {}  # node_id -> {hardware, last_seen, rank}
        self.min_nodes = min_nodes
        self.heartbeat_timeout = heartbeat_timeout
        self.proportions: list = []
        self.cluster_version = 0  # increments on every cluster change
        self.lock = threading.Lock()
        self.model = None
        self.device = None
        # Barrier: tracks which nodes are ready for which version
        self.ready_nodes: Dict[int, set] = {}  # version -> set of node_ids
        # Event log for dashboard
        self.events: List[dict] = []
        self.start_time = time.time()

    def register(self, node_id: str, hardware: dict) -> dict:
        with self.lock:
            is_new = node_id not in self.nodes
            self.nodes[node_id] = {
                "hardware": hardware,
                "last_seen": time.time(),
                "rank": None,
            }
            if is_new:
                self.events.append({"time": time.time(), "type": "join", "node": node_id, "hw": f"{hardware['name']} {hardware['memory_gb']}GB"})
                self._recompute()
            return self._status()

    def heartbeat(self, node_id: str) -> dict:
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Node {node_id} not registered")
            self.nodes[node_id]["last_seen"] = time.time()
            return self._status()

    def remove(self, node_id: str) -> dict:
        with self.lock:
            if node_id in self.nodes:
                self.events.append({"time": time.time(), "type": "leave", "node": node_id})
                del self.nodes[node_id]
                self._recompute()
            return self._status()

    def check_heartbeats(self):
        """Remove nodes that haven't sent a heartbeat recently."""
        with self.lock:
            now = time.time()
            dead = [
                nid for nid, info in self.nodes.items()
                if now - info["last_seen"] > self.heartbeat_timeout
            ]
            if dead:
                for nid in dead:
                    print(f"[coordinator] Node {nid} timed out, removing")
                    self.events.append({"time": time.time(), "type": "timeout", "node": nid})
                    del self.nodes[nid]
                self._recompute()

    def signal_ready(self, node_id: str, version: int) -> dict:
        """Node signals it's ready to reconfigure to this version."""
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Node {node_id} not registered")
            if version not in self.ready_nodes:
                self.ready_nodes[version] = set()
            self.ready_nodes[version].add(node_id)

            all_ready = self._check_barrier(version)
            return {
                "version": version,
                "your_ready": True,
                "all_ready": all_ready,
                "ready_count": len(self.ready_nodes.get(version, set())),
                "total_needed": len(self.nodes),
            }

    def _check_barrier(self, version: int) -> bool:
        """Check if all current nodes are ready for this version."""
        if version not in self.ready_nodes:
            return False
        current_nodes = set(self.nodes.keys())
        ready = self.ready_nodes[version]
        return current_nodes.issubset(ready)

    def _recompute(self):
        """Recompute ranks and proportions based on current nodes."""
        # Clear old barriers
        self.ready_nodes = {}
        sorted_nodes = sorted(self.nodes.keys())
        hardware_list = []
        for i, nid in enumerate(sorted_nodes):
            self.nodes[nid]["rank"] = i
            hardware_list.append(self.nodes[nid]["hardware"])

        if len(hardware_list) >= 2:
            self.proportions = compute_proportions(hardware_list)
        else:
            self.proportions = [1.0] * len(hardware_list)

        self.cluster_version += 1
        print(f"[coordinator] Cluster v{self.cluster_version}: "
              f"{len(self.nodes)} nodes, proportions={self.proportions}")
        for nid in sorted_nodes:
            info = self.nodes[nid]
            hw = info["hardware"]
            print(f"  rank {info['rank']}: {nid} ({hw['name']} {hw['memory_gb']}GB {hw['type']})")

    def _status(self) -> dict:
        return {
            "cluster_version": self.cluster_version,
            "num_nodes": len(self.nodes),
            "min_nodes": self.min_nodes,
            "ready": len(self.nodes) >= self.min_nodes,
            "proportions": self.proportions,
            "nodes": {
                nid: {
                    "rank": info["rank"],
                    "hardware": info["hardware"],
                }
                for nid, info in self.nodes.items()
            },
        }


def create_coordinator_app(min_nodes=2, heartbeat_timeout=60, model=None, device=None):
    app = FastAPI(title="Ravnest Cluster Coordinator")
    state = ClusterState(min_nodes=min_nodes, heartbeat_timeout=heartbeat_timeout)
    state.model = model
    state.device = device

    # Background heartbeat checker (fast polling for quicker dead-node detection)
    def heartbeat_monitor():
        while True:
            time.sleep(5)
            state.check_heartbeats()

    monitor = threading.Thread(target=heartbeat_monitor, daemon=True)
    monitor.start()

    @app.get("/status")
    def status():
        with state.lock:
            return state._status()

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        with state.lock:
            s = state._status()
            uptime = int(time.time() - state.start_time)
            hours, rem = divmod(uptime, 3600)
            mins, secs = divmod(rem, 60)
            uptime_str = f"{hours}h {mins}m {secs}s"

            # Build node rows
            node_rows = ""
            sorted_nodes = sorted(state.nodes.keys())
            for nid in sorted_nodes:
                info = state.nodes[nid]
                hw = info["hardware"]
                ago = int(time.time() - info["last_seen"])
                health = "🟢" if ago < state.heartbeat_timeout else "🔴"
                prop = state.proportions[info["rank"]] if info["rank"] < len(state.proportions) else 0
                node_rows += f"""<tr>
                    <td>{health}</td>
                    <td>{info['rank']}</td>
                    <td><code>{nid}</code></td>
                    <td>{hw['name']}</td>
                    <td>{hw['type'].upper()}</td>
                    <td>{hw['memory_gb']} GB</td>
                    <td>{prop:.0%}</td>
                    <td>{ago}s ago</td>
                </tr>"""

            if not node_rows:
                node_rows = '<tr><td colspan="8" style="text-align:center;color:#888">No nodes connected</td></tr>'

            # Build event rows (last 20)
            event_rows = ""
            for ev in reversed(state.events[-20:]):
                ago = int(time.time() - ev["time"])
                icon = {"join": "➕", "leave": "👋", "timeout": "💀"}.get(ev["type"], "•")
                detail = ev.get("hw", "")
                event_rows += f"<tr><td>{icon} {ev['type']}</td><td><code>{ev['node'][:20]}</code></td><td>{detail}</td><td>{ago}s ago</td></tr>"

            if not event_rows:
                event_rows = '<tr><td colspan="4" style="text-align:center;color:#888">No events yet</td></tr>'

            # Proportions bar
            prop_bar = ""
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#CDDC39", "#795548"]
            for i, nid in enumerate(sorted_nodes):
                info = state.nodes[nid]
                prop = state.proportions[info["rank"]] if info["rank"] < len(state.proportions) else 0
                color = colors[i % len(colors)]
                prop_bar += f'<div style="width:{prop*100}%;background:{color};height:30px;display:inline-block;text-align:center;color:white;line-height:30px;font-size:12px" title="{nid}">R{info["rank"]} {prop:.0%}</div>'

            if not prop_bar:
                prop_bar = '<div style="width:100%;background:#eee;height:30px;text-align:center;line-height:30px;color:#888">No nodes</div>'

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ravnest Cluster Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat {{ text-align: center; padding: 10px 20px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
        th {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
        .ready {{ color: #4CAF50; font-weight: bold; }}
        .not-ready {{ color: #F44336; font-weight: bold; }}
        .prop-bar {{ border-radius: 4px; overflow: hidden; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Ravnest Cluster Dashboard</h1>

    <div class="card">
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{s['num_nodes']}</div>
                <div class="stat-label">Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value {'ready' if s['ready'] else 'not-ready'}">{'Ready' if s['ready'] else 'Waiting'}</div>
                <div class="stat-label">Status</div>
            </div>
            <div class="stat">
                <div class="stat-value">v{s['cluster_version']}</div>
                <div class="stat-label">Version</div>
            </div>
            <div class="stat">
                <div class="stat-value">{uptime_str}</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>Layer Distribution</h3>
        <div class="prop-bar">{prop_bar}</div>
    </div>

    <div class="card">
        <h3>Nodes</h3>
        <table>
            <tr><th></th><th>Rank</th><th>Node ID</th><th>Hardware</th><th>Type</th><th>Memory</th><th>Layers</th><th>Last Seen</th></tr>
            {node_rows}
        </table>
    </div>

    <div class="card">
        <h3>Recent Events</h3>
        <table>
            <tr><th>Event</th><th>Node</th><th>Details</th><th>When</th></tr>
            {event_rows}
        </table>
    </div>

    <p style="color:#888;font-size:12px;text-align:center">Auto-refreshes every 5 seconds. API: <code>/status</code> <code>/config/node_id</code></p>
</body>
</html>"""
        return html

    @app.post("/register")
    def register(reg: NodeRegistration):
        result = state.register(reg.node_id, reg.hardware)
        return result

    @app.post("/heartbeat/{node_id}")
    def heartbeat(node_id: str):
        try:
            return state.heartbeat(node_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")

    @app.post("/leave/{node_id}")
    def leave(node_id: str):
        return state.remove(node_id)

    @app.get("/config/{node_id}")
    def get_node_config(node_id: str):
        """Get the inference config for a specific node."""
        with state.lock:
            if node_id not in state.nodes:
                raise HTTPException(status_code=404, detail="Node not registered")
            if not state._status()["ready"]:
                raise HTTPException(status_code=503, detail="Cluster not ready, waiting for more nodes")

            info = state.nodes[node_id]
            sorted_nodes = sorted(state.nodes.keys())
            master_id = sorted_nodes[0]

            # Build peer map: rank -> node_id (IP)
            peers = {
                state.nodes[nid]["rank"]: nid
                for nid in sorted_nodes
            }

            return {
                "rank": info["rank"],
                "world_size": len(state.nodes),
                "master_addr": master_id,
                "proportions": state.proportions,
                "cluster_version": state.cluster_version,
                "model": state.model,
                "device": state.device,
                "peers": peers,
            }

    @app.post("/ready/{node_id}/{version}")
    def signal_ready(node_id: str, version: int):
        """Worker signals it's ready for a specific cluster version."""
        try:
            return state.signal_ready(node_id, version)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")

    @app.get("/barrier/{version}")
    def check_barrier(version: int):
        """Check if all nodes are ready for this version."""
        with state.lock:
            all_ready = state._check_barrier(version)
            ready_count = len(state.ready_nodes.get(version, set()))
            return {
                "version": version,
                "all_ready": all_ready,
                "ready_count": ready_count,
                "total_needed": len(state.nodes),
            }

    return app
