
Morpheus-Framework
==================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://www.python.org/downloads/release/python-360/





Installation
============

Requirements:

- ``astropy``
- ``numpy``
- ``tqdm``


.. code-block:: bash

    pip install morpheus-astro-framework


Usuage
======

Setup
-----

To use ``morpheus_framework`` to apply your model to astronomical images you
need to provide ``morpheus_framework`` with your model in the form of a
``Callable`` function and the inputs arrays in the form of a list of ``numpy``
arrays or a list of strings that are the file locations of the ``fits`` files
that are inputs.

.. code-block:: python

    from morpheus_framework import morpheus_framework

    n_classes = 5             # number of classes that are output from the model
    batch_size = 16           # number of samples to extract per batch
    window_shape = (100, 100) # (height, width) of each sample

    output_hduls, output_arrays = morpheus_framework.predict(
        model,        # your model in a callable from
        model_inputs, # list of numpy arrays or strings that point to fits files
        n_classes,
        batch_size,
        window_shape
    )

Output Format
-------------


Parallelization
---------------

``morpheus_framework`` supports the parallel classification of large images
by splitting the input along the first dimension (height typically), classifying
each piece in parallel, and then combining the resulting classifications into
a single classified image.

GPU
***

CPU
***


Citation
========

If you use this package in your research please cite the original paper:

.. code-block::

    @ARTICLE{2020ApJS..248...20H,
        author = {{Hausen}, Ryan and {Robertson}, Brant E.},
        title = "{Morpheus: A Deep Learning Framework for the Pixel-level Analysis of Astronomical Image Data}",
        journal = {\apjs},
        keywords = {Galaxy classification systems, Galaxies, Extragalactic astronomy, Convolutional neural networks, Computational methods, GPU computing, Astrophysics - Astrophysics of Galaxies, Computer Science - Machine Learning},
        year = 2020,
        month = may,
        volume = {248},
        number = {1},
        eid = {20},
        pages = {20},
        doi = {10.3847/1538-4365/ab8868},
        archivePrefix = {arXiv},
        eprint = {1906.11248},
        primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJS..248...20H},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }






