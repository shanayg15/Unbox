"""Lane B, task B2: brain clients"""


class Brain:
    def __init__(self, provider: str) -> None:
        raise NotImplementedError

    def turn(self, messages: list[dict]) -> dict:
        raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
