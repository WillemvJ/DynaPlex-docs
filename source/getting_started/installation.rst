Installation
============

Requirements
------------

- **Python 3.13 only**
- **Platform support:**

  - Windows: AMD64
  - Linux: x86_64 (glibc 2.28 or newer, ``manylinux_2_28``)
  - macOS: Apple Silicon (arm64), macOS 14.0 or newer; no Intel hardware supported
  
- **PyTorch:** Algorithms require a compatible PyTorch installation. Use the `PyTorch selector <https://pytorch.org/get-started/locally/>`_ to install the appropriate version for your system.

.. note::
   Building from source is presently not supported.

Installation
------------

Install DynaPlex using pip:

.. code-block:: bash

   pip install dynaplex

Or install with all optional dependencies:

.. code-block:: bash

   pip install dynaplex[complete]

.. note::
   Even with a complete install, separate installation of PyTorch is still required for the RL algorithms.

Next Steps
----------

Once you have DynaPlex installed, you could start with the :doc:`introduction to MDPs <introduction_to_mdp>`, or dive right into the :doc:`tutorials <tutorial/airplane_mdp>`. 
