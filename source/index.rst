.. figure:: assets/images/logo.png
   :alt: DynaPlex 2 logo
   :figwidth: 100%

DynaPlex 2 is a Python library for solving Markov Decision Problems and similar models (POMDP, HMM). It supports 
deep reinforcement learning, approximate dynamic programming, classical parameterized policies, and exact methods based on policy and value iteration. Models in DynaPlex 2 are written in Python, and exposed via a generic and vectorized interface. 

DynaPlex 2 focuses on solving problems arising in Operations Management: Supply Chain, Transportation and Logistics, Manufacturing, etc. 

.. note::

    If you are new to MDPs, you might benefit from first reading the :doc:`introduction to MDPs <getting_started/introduction_to_mdp>` and going thorugh the step-by-step tutorial, starting with the :doc:`MDP formulation <tutorial/airplane_mdp>` pages.
    If you just want to know how to install, setup, and add a model, see the docs under "Getting started"

Contents
--------

.. toctree::
   :maxdepth: 0
   :caption: Getting started

   getting_started/introduction_to_mdp
   getting_started/installation
   getting_started/conda
   getting_started/adding_model
   getting_started/testing

.. toctree::
   :maxdepth: 0
   :caption: Tutorial

   tutorial/airplane_mdp
   tutorial/setup
   tutorial/adding_mdp
   tutorial/policy
   tutorial/testing_running

.. toctree::
   :maxdepth: 0
   :caption: Getting help and Contributing

   community/contributing
   community/getting_help