from typing import (
    Any,
    Generic,
    Iterator,
    Optional,
    Union,
    KeysView,
    ValuesView,
    ItemsView,
    TypeVar,
)


# Define a type variable for the values stored in the dictionary
T = TypeVar("T")


class DictList(Generic[T]):
    """
    A dictionary-like class that allows access to its items by key or index.


    Attributes:
        _data (Dict[str, Any]): The underlying dictionary to store items.
        _keys (List[str]): List of keys to maintain the order for index-based access.

    Methods:
        __getitem__(key: Union[str, int]) -> Any: Get an item by key or index.
        __setitem__(key: str, value: Any) -> None: Set an item by key.
        __delitem__(key: str) -> None: Delete an item by key.
        get(key: str, default: Optional[Any] = None) -> Any: Get an item with a default if the key is not present.
        __iter__() -> Iterator[str]: Get an iterator over the keys of the dictionary.
        keys() -> KeysView[str]: Get a view object of the dictionary's keys.
        values() -> ValuesView[Any]: Get a view object of the dictionary's values.
        items() -> ItemsView[str, Any]: Get a view object of the dictionary's items.
        __len__() -> int: Get the number of items in the dictionary.
        __str__() -> str: Get the string representation of the dictionary.
        __repr__() -> str: Get the official string representation of the object.
    """

    def __init__(self, data: Union[None, dict[str, T]] = None) -> None:
        """Initialize the dictList with an optional dictionary."""
        if data is None:
            data = {}
        self._data: dict[str, Any] = data
        self._keys: list[str] = list(data.keys())

    def __getitem__(self, key: Union[str, int]) -> T:
        """Get an item by key or index."""
        key_ = key if isinstance(key, str) else self._keys[key]
        return self._data[key_]

    def __setitem__(self, key: str, value: T) -> None:
        """Set an item by key."""
        self._data[key] = value
        if key not in self._keys:
            self._keys.append(key)

    def __delitem__(self, key: str) -> None:
        """Delete an item by key."""
        del self._data[key]
        self._keys.remove(key)

    def get(self, key: str, default: Optional[Any] = None) -> T:
        """Get an item with a default if the key is not present."""
        return self._data.get(key, default)

    def __iter__(self) -> Iterator[str]:
        """Get an iterator over the keys of the dictionary."""
        return iter(self._data)

    def keys(self) -> KeysView[str]:
        """Get a view object of the dictionary's keys."""
        return self._data.keys()

    def values(self) -> ValuesView[T]:
        """Get a view object of the dictionary's values."""
        return self._data.values()

    def items(self) -> ItemsView[str, T]:
        """Get a view object of the dictionary's items."""
        return self._data.items()

    def __len__(self) -> int:
        """Get the number of items in the dictionary."""
        return len(self._data)

    def __str__(self) -> str:
        """Get the string representation of the dictionary."""
        return str(self._data)

    def __repr__(self) -> str:
        """Get the official string representation of the object."""
        return f"{self.__class__.__name__}({self._data})"
