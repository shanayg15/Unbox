"""Lane A, task A1: HTTP client implementing contracts.types.BlackBox"""
from contracts.types import Episode, Obs, QueryBatch, SensorSpec


class HttpBlackBox:
    def __init__(self, bb_id: str, url: str) -> None:
        raise NotImplementedError

    def spec(self) -> SensorSpec:
        raise NotImplementedError

    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch:
        raise NotImplementedError

    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode:
        raise NotImplementedError

    def query_count(self) -> int:
        raise NotImplementedError
