from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Union

Obs = list[float]
Action = Union[int, list[float]]
Params = dict[str, float]
ProgramFn = Callable[[Obs, Params], Action]
ActionType = Literal["discrete", "continuous"]


@dataclass
class SensorSpec:
    obs_names: list[str]
    obs_low: list[float]           # empirical p1 if the space is unbounded
    obs_high: list[float]          # empirical p99 if the space is unbounded
    action_type: ActionType
    action_names: list[str]        # discrete: one name per action id; continuous: one per dim
    action_low: list[float] | None = None
    action_high: list[float] | None = None


@dataclass
class Episode:
    seed: int
    obs: list[Obs]
    actions: list[Action]
    rewards: list[float]
    total_reward: float
    terminated: bool
    truncated: bool
    frames: list[str] | None = None        # base64 JPEG, one every `frame_every` steps
    frame_every: int = 2


@dataclass
class QueryBatch:
    query_id: str                           # "q17"; assigned by the black box
    obs: list[Obs]
    actions: list[Action]
    source: Literal["rollout", "probe", "prober", "holdout"]
    rule_id: str | None = None
    inference_us: float | None = None       # mean model time per obs, measured server-side


class BlackBox(Protocol):
    def spec(self) -> SensorSpec: ...
    def act(self, obs_batch: list[Obs], source: str = "probe", rule_id: str | None = None) -> QueryBatch: ...
    def episode(self, seed: int, render: bool = False, frame_every: int = 2) -> Episode: ...
    def query_count(self) -> int: ...


@dataclass
class Program:
    version: int
    code: str
    params: Params
    agreement: float | None = None          # open-loop agreement on holdout after fitting


@dataclass
class Counterexample:
    obs: Obs
    bb_action: Action
    clone_action: Action
    query_id: str


@dataclass
class EvalReport:
    open_loop_agreement: float
    n_holdout_obs: int
    clone_mean_reward: float | None = None
    bb_mean_reward: float | None = None
    n_episodes: int = 0
    n_failures: int = 0
    failure_rate_bound_95: float | None = None   # rule of three: 3/n when n_failures == 0
    us_per_action_clone: float | None = None
    us_per_action_bb: float | None = None


LedgerStatus = Literal["observed", "inferred", "probing", "confirmed", "refuted"]


@dataclass
class LedgerEntry:
    rule_id: str                             # "r1", "r2", ... assigned by Python
    text: str                                # plain English, one line
    status: LedgerStatus
    confidence: float                        # 0..1
    evidence: list[str] = field(default_factory=list)   # query_ids
    program_version: int | None = None
