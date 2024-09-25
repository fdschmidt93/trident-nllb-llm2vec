from functools import reduce  # Python 3
import re


def deepgetitem(obj, item, default=None):
    """Steps through an item chain to get the ultimate value.

    If ultimate value or path to value does not exist, does not raise
    an exception and instead returns `fallback`.

    Does not work if keys comprise dots.

    Credits: https://stackoverflow.com/a/38623359

    >>> d = {'snl_final': {'about': {'_icsd': {'icsd_id': 1}}}}
    >>> deepgetitem(d, 'snl_final.about._icsd.icsd_id')
    1
    >>> deepgetitem(d, 'snl_final.about._sandbox.sbx_id')
    >>>
    """

    def getitem(obj, name):
        try:
            if isinstance(obj, dict):
                return obj[name]
            else:
                return getattr(obj, name)
        except (KeyError, TypeError, AttributeError):
            return default

    return reduce(getitem, item.split("."), obj)


def get_dict_by_glob_pattern(dictionary: dict, pattern: str):
    # Convert the glob pattern to a regular expression
    # Replace '*' with '.*' to match any characters (non-greedy)
    regex = re.compile(re.escape(pattern).replace("\\*", ".*?"))

    # Search for matching keys and create a new dictionary with them
    return {key: value for key, value in dictionary.items() if regex.match(key)}
