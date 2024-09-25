from typing import Any, Protocol


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, key: int) -> Any: ...
