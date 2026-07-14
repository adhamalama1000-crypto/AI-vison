"""
Regression test for the 'capture freezes the app' bug.

A fake camera is fed frames into its in-memory buffer (no RTSP, no second
connection). While many CPU-bound capture/enrol requests run concurrently, a
stream of lightweight /health requests must stay responsive. If capture blocked
the event loop (the old behaviour), health latency would spike; with the work
offloaded to the threadpool it stays low.
"""

from __future__ import annotations

import asyncio
import time

import cv2
from skimage import data

from rtsp_backend.config import CameraConfig, Settings
from rtsp_backend.app import build_app


def _astro():
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


async def _scenario(tmp_path):
    import httpx

    settings = Settings(db_path=str(tmp_path / "t.db"), data_dir=str(tmp_path / "data"),
                        models_dir=str(tmp_path / "models"), cameras=[])
    app = build_app(settings)
    cam = app.state.manager.add_camera(
        CameraConfig(id="cam1", name="Fake", url="rtsp://example/stream"),
        start=False, emit=False)
    cam.buffer.put(_astro())               # a real frame with a detectable face
    app.state.ai.set_enabled("face", True)  # each capture now does real detection

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as client:
        eid = (await client.post("/api/employees", json={"full_name": "Astro"})).json()["id"]

        health_latencies: list[float] = []
        stop = asyncio.Event()

        async def ping_health():
            while not stop.is_set():
                t0 = time.monotonic()
                r = await client.get("/health")
                health_latencies.append(time.monotonic() - t0)
                assert r.status_code == 200
                await asyncio.sleep(0.02)

        async def do_captures(n=12):
            oks = 0
            # fire captures concurrently so several detections overlap
            async def one():
                nonlocal oks
                r = await client.post(f"/api/employees/{eid}/capture",
                                      json={"camera_id": "cam1"})
                if r.status_code == 200 and r.json().get("enrollment", {}).get("ok"):
                    oks += 1
            await asyncio.gather(*[one() for _ in range(n)])
            return oks

        pinger = asyncio.create_task(ping_health())
        await asyncio.sleep(0.2)
        oks = await do_captures()
        stop.set()
        await pinger

    return oks, health_latencies


def test_capture_does_not_block_event_loop(tmp_path):
    oks, latencies = asyncio.run(asyncio.wait_for(_scenario(tmp_path), timeout=60))
    # every capture enrolled a face
    assert oks >= 10, f"only {oks} captures enrolled"
    # the loop stayed responsive: health pings kept flowing and none stalled badly
    assert len(latencies) >= 10, f"health starved: {len(latencies)} pings"
    assert max(latencies) < 1.0, f"event loop blocked: max health latency {max(latencies):.2f}s"
