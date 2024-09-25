###############
TridentDataspec
###############

.. _dataspec_configuration:

*************
Configuration
*************

.. include:: dataspec_intro.rst

dataset
=======

.. include:: dataspec_dataset.rst

.. _preprocessing:

preprocessing
=============

.. include:: dataspec_preprocessing.rst

dataloader
==========

.. include:: dataspec_dataloader.rst

evaluation
==========

.. include:: dataspec_evaluation.rst

***
API
***

Properties
==========

cfg
"""

The ``cfg: omegaconf.DictConfig`` holds the dataspec configuration.

.. seealso:: :ref:`TridentDatapsec.Configuration <dataspec_configuration>`

dataset
"""""""

The dataset as declared in  ``cfg.dataset`` and preprocssed by ``cfg.preprocessing``.

evaluation
""""""""""

The evaluation as declared in  ``cfg.evaluation``.

Methods
=======

get_dataloader
""""""""""""""

.. automethod:: trident.core.dataspec.TridentDataspec.get_dataloader
    :noindex:
