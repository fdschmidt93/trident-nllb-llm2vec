.. include globals explicitly for pandoc
.. include:: globals.rst

###############
Using |project|
###############

|project| is a deep learning framework designed to train and evaluate models efficiently. It serves as a convenience layer atop several robust libraries:

* Lightning_
* hydra_
* transformers_
* datasets_

|project|'s goal is to minimize boilerplate through a modular design:

* Data and preprocessing pipelines are powered by datasets_ and LightningDataModule_, requiring minimal user setup.
* Training and evaluation are simplified with high-level wrappers around Lightning_.

Familiarity with hydra_ is essential, as it forms the core framework for configuring and composing experiments in |project|.

Quick Start
===========

.. include:: installation.rst

Usage
-----

Typical usage of trident follows the below schema:

1. Clone the repo
2. Write a configuration for your model (see also :ref:`walkthrough`)
3. Train on an existing experiment with `python -m trident.run experiment=mnli module=my_model`

You can find existing pipelines at :repo:`experiments configs <configs/experiment/>`. A full experiment (incl. `module`) is defined in the :repo:`MNLI-TinyBert config <configs/experiment/mnli_tinybert.yaml>`.


Contributing
------------

Please see :ref:`Contributing <contributing>`!


Credits
-------

* This project is was largely inspired by and is based on https://github.com/ashleve/lightning-hydra-template
* A related project is: https://github.com/PyTorchLightning/lightning-transformers

Author
------

| Name: Fabian David Schmidt
| Mail: fabian.schmidt@uni-wuerzburg.de
| Affiliation: University of Würzburg
