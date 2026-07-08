.. figure:: assets/images/logo.png
   :alt: DynaPlex logo
   :figwidth: 100%

DynaPlex is a Python library for solving Markov Decision Problems and similar models (POMDP, HMM). It supports
deep reinforcement learning, approximate dynamic programming, classical parameterized policies, and exact methods based on policy and value iteration. Models in DynaPlex are written in Python, and exposed via a generic and vectorized interface.

DynaPlex focuses on solving problems arising in Operations Management: Supply Chain, Transportation and Logistics, Manufacturing, etc.

.. note::

    If you are new to MDPs, you might benefit from first reading the :doc:`introduction to MDPs <getting_started/introduction_to_mdp>` and going through the step-by-step tutorial, starting with the :doc:`MDP formulation <tutorial/airplane_mdp>` pages.

Contents
--------

.. toctree::
   :maxdepth: 0
   :caption: Getting started

   getting_started/installation
   getting_started/verifying_installation
   getting_started/introduction_to_mdp

.. toctree::
   :maxdepth: 0
   :caption: Tutorial

   tutorial/airplane_mdp
   tutorial/airplane_mdp_python_code
   tutorial/binpacking_mdp
   tutorial/binpacking_mdp_python_code

.. toctree::
   :maxdepth: 0
   :caption: Reference

   reference/language_reference

.. toctree::
   :maxdepth: 0
   :caption: Getting help and Contributing

   community/getting_help