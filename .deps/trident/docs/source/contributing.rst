.. _contributing:

Contributing
=============

Many thanks for taking the time to contribute to |project| if you are reading this!

Writing Documentation
---------------------

Each Pull Request is required to document the contribution adequately in markup or docstrings. 

* |project|\'s documentation is written in reStructuredText (`rst`) and can be found in the :repo:`docs/source </docs/source>` folder of the repo.
* The Python docstrings follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_ like projects such as Huggingface transformers. 
* The `docgen <https://github.com/fdschmidt93/trident/blob/main/.github/workflows/docgen.yaml>`_ action will then auto-generate the documentation pages with `Sphinx <https://www.sphinx-doc.org/en/master/>`_ (also a great reference on how to write idiomatic docs)

Once you have written the relevant documentation you can preview your changes locally as follows.

.. code-block:: bash

    # in project folder 
    conda env create -f .docgen-env.yaml
    conda activate sphinx
    cd docs
    sphinx-apidoc -o ./source ../ -f
    sphinx-build ./source/ ./build/

At last, you can open either :obj:`docs/build/` the entire documentation at once or :obj:`docs/build/readme.html` or the landing page with your preferred browser, e.g.

.. code-block:: bash

    # in docs
    chromium ./build
    chromium ./build/readme.html

Opening a PR
------------

1. Fork the Project
2. Create your Feature Branch (:code:`git checkout -b my_contribution`)
3. Make your changes
4. Stage and commit your Changes (:code:`git add -u && git commit -m 'Add my contribution'`)
5. Push to the Branch (:code:`git push origin my_contribution`)
6. Open a Pull Request by going to the project webpage; it'll then suggest to you to open a PR on |project|
