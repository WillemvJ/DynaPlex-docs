.. _label_binpacking:

Bin Packing MDP formulation
===========================

The bin packing problem is an example of an infinite horizon MDP. Weights arrive sequentially and must be assigned to one of several bins. Each bin has a maximum capacity (``max_bin_size``). When a bin's total weight exceeds this capacity, the overflow weight incurs a cost, and the bin is emptied. The goal is to minimize the total overflow cost over an infinite horizon.

In this problem, weights are revealed one by one according to a probability distribution. Each weight must be immediately assigned to one of the available bins. When a bin exceeds its maximum capacity, any overflow weight incurs a cost proportional to the overflow amount, and the bin is reset to empty. The process continues indefinitely.

.. note::

   This is an **infinite horizon** MDP, meaning the process continues indefinitely rather than terminating after a fixed number of steps. This contrasts with finite horizon MDPs like the :doc:`airplane MDP <airplane_mdp>`, which has a fixed termination time (day 26).

   In infinite horizon MDPs, it is especially important to explicitly track time progression. DynaPlex requires you to manually update ``context.time_elapsed`` to track the passage of time. In this bin packing problem, time progresses by one unit each time a new random weight arrives (in the ``modify_state_with_event`` function).

The Components of the MDP
-------------------------

1. **States (S):** The state consists of:
   - ``weight_vector``: A NumPy array representing the current weight in each bin
   - ``upcoming_weight``: The weight that has just arrived and must be assigned
   - ``category``: The state category (``AWAIT_EVENT`` or ``AWAIT_ACTION``)

2. **Actions (A):** The action is to select which bin (0 to ``number_of_bins-1``) to assign the upcoming weight to. All bins are always valid actions.

3. **Transitions:** 
   - When a new weight arrives (event), it is sampled from the weight distribution and stored in ``upcoming_weight``. The state transitions to ``AWAIT_ACTION``, and ``context.time_elapsed`` is incremented.
   - When an action is taken, the weight is added to the selected bin. If the bin overflows, the overflow cost is added to ``context.cumulative_cost``, and the bin is reset to empty. The state transitions back to ``AWAIT_EVENT``.

4. **Costs (C):** The cost is the overflow amount when a bin exceeds its capacity.

.. note::

   **Time Tracking in Infinite Horizon MDPs:**
   
   Unlike finite horizon MDPs where time is naturally bounded, infinite horizon MDPs require explicit time tracking. In DynaPlex, you must manually increment ``context.time_elapsed`` to track the progression of time. This is essential for:
   
   - Algorithms that need to know how many steps have elapsed
   - Stopping criteria for training or evaluation
   
   In this bin packing problem, time advances by one unit each time a new weight arrives, which happens in the ``modify_state_with_event`` function: ``context.time_elapsed += 1``.

Policy
------

This MDP example includes two simple heuristic policies:

1. **LowestWeightPolicy:** Always assigns the incoming weight to the bin with the lowest current weight. This is a greedy strategy that tries to balance the load across bins.

2. **FirstFitPolicy:** Assigns the weight to the first bin that can accommodate it without overflow. If no bin can accommodate it without overflow, it assigns to the first bin. This is a common heuristic in bin packing problems.
