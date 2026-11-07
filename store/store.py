"""Lane B, task B4: store (JSON and Mongo backends, same interface)"""


class Store:
    def save_event(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_brain_call(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_program(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_ledger(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save_eval(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def list_runs(self) -> list:
        raise NotImplementedError

    def load_events(self, run_id: str) -> list:
        raise NotImplementedError


class JsonStore(Store):
    pass


class MongoStore(Store):
    pass
