"""Lane B, task B3: agent loop"""


class RunResult:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError


def run(bb_id, twist_label_hidden, brain, budget, max_turns, live, emit, store) -> RunResult:
    raise NotImplementedError


def render_pair(fn, params, demo_seed):
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
