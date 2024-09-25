from dataclasses import dataclass, asdict
from typing import Callable, KeysView, Optional, Dict, List, Union, ValuesView


@dataclass
class PrepareDict:
    batch: Optional[Callable] = None
    outputs: Optional[Callable] = None
    step_outputs: Optional[Callable] = None

    def __getitem__(self, item) -> Optional[Callable]:
        return getattr(self, item)

    def get(self, item, default=None) -> Optional[Callable]:
        return getattr(self, item, default)

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()


@dataclass
class StepOutputsDict:
    batch: Union[None, str, List[str]] = None
    outputs: Union[None, str, List[str]] = None

    def __getitem__(self, item) -> Union[None, str, list[str]]:
        return getattr(self, item)

    def get(self, item, default=None) -> Union[None, str, list[str]]:
        return getattr(self, item, default)

    def keys(self) -> KeysView[str]:
        return asdict(self).keys()

    def values(self) -> ValuesView[Union[str, list[str]]]:
        return asdict(self).values()

    def items(self):
        return asdict(self).items()


@dataclass
class MetricDict:
    kwargs: Dict[str, str]
    metric: Callable
    compute_on: Optional[str] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def keys(self) -> KeysView[str]:
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()


@dataclass
class PreprocessingDict:
    method: Optional[Dict[str, Callable]] = None
    apply: Optional[Dict[str, Callable]] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()


@dataclass
class EvaluationDict:
    metrics: Dict[str, MetricDict]
    prepare: Optional[PrepareDict] = None
    step_outputs: Optional[StepOutputsDict] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()
