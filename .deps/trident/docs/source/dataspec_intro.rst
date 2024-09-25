A :class:`~trident.core.dataspec.TridentDataspec` class encapsulates the configuration for data handling in a machine learning workflow. It manages various aspects of data processing including dataset instantiation, preprocessing, dataloading, and evaluation.

**Configuration Keys**

- ``dataset``: Specifies how the dataset should be instantiated.
- ``dataloader``: Defines the instantiation of the ``DataLoader``.
- ``preprocessing`` (optional): Details the methods or function calls for dataset preprocessing.
- ``evaluation`` (optional): Outlines any post-processing steps and metrics for dataset evaluation.
- ``misc`` (optional): Reserved for miscellaneous settings that do not fit under other keys.
