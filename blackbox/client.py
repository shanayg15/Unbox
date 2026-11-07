"""Lane A, task A1: HTTP client implementing contracts.types.BlackBox.

This is the only blackbox module the agent may import, so it stays free of
everything server-side.
"""
from __future__ import annotations

import httpx

from contracts.types import Episode, Obs, QueryBatch, SensorSpec


class HttpBlackBox:
    def __init__(self, bb_id: str, url: str, timeout: float = 120.0) -> None:
        self.bb_id = bb_id
        self._http = httpx.Client(base_url=url, timeout=timeout)

    def spec(self) -> SensorSpec:
        return SensorSpec(**self._get("/spec"))

    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch:
        body = self._post("/act", {"obs": obs_batch, "source": source, "rule_id": rule_id})
        return QueryBatch(
            query_id=body["query_id"],
            obs=obs_batch,
            actions=body["actions"],
            source=source,
            rule_id=rule_id,
            inference_us=body["inference_us"],
        )

    def episode(self, seed: int, render: bool = False, frame_every: int = 3) -> Episode:
        return Episode(**self._post("/episode", {"seed": seed, "render": render, "frame_every": frame_every}))

    def query_count(self) -> int:
        return self._get("/count")["queries"]

    def _get(self, path: str) -> dict:
        response = self._http.get(path, params={"bb": self.bb_id})
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, body: dict) -> dict:
        response = self._http.post(path, json={"bb": self.bb_id, **body})
        response.raise_for_status()
        return response.json()
