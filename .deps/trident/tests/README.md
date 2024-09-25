# Tests

This directory contains the structure and configuration files used for testing with pytest. The structure has been designed for compatibility with hydra experiment configuration.

The `configs` directory comprises `hydra` configuration for the corresponding test in `test_experiments.py`. Each config is a single file without dependencies to other configuration files.
Utility functions are in the `helpers` directory.

## Executing Tests

Run `pytest` in the root directory of the original project to execute all tests.

## Contribution

When adding new tests or modifying existing ones:

1. Ensure that any new configuration files are added to the configs directory with a descriptive name.
2. If leveraging helper functions, place them inside the helpers directory and ensure they're imported correctly in the main test files.
3. Always run the tests after making changes to ensure everything works as expected.
