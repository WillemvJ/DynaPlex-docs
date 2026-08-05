# DynaPlex

<div class="dp-hero" markdown>
![DynaPlex logo](assets/images/logo.png){ width="420" }
</div>

DynaPlex is a Python library for solving Markov Decision Problems and similar
models (POMDP, HMM). It supports deep reinforcement learning, approximate
dynamic programming, classical parameterized policies, and exact methods based
on policy and value iteration. Models in DynaPlex are written in Python, and
executed by a fast multi-threaded engine with a bundled LLVM JIT.

DynaPlex focuses on solving problems arising in Operations Management: Supply
Chain, Transportation and Logistics, Manufacturing, and related fields.

```bash
pip install dynaplex
```

!!! tip "New to MDPs?"
    Start with the [introduction to MDPs](getting-started/introduction-to-mdps.md),
    then work through the step-by-step tutorial, beginning with the
    [airplane ticket selling MDP](tutorials/airplane-mdp.md).

## Where to go next

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Getting started**

    ---

    Install the package and verify that it works on your platform, JIT
    included.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Two worked examples — a finite-horizon ticket selling problem and an
    infinite-horizon bin packing problem — with complete, runnable code.

    [:octicons-arrow-right-24: Airplane MDP](tutorials/airplane-mdp.md)

-   :material-book-open-variant:{ .lg .middle } **Language reference**

    ---

    The complete reference for DynaML, the Python subset in which DynaPlex
    models are written — including all built-in functions.

    [:octicons-arrow-right-24: Language reference](reference/language-reference.md)

-   :material-account-group:{ .lg .middle } **Community**

    ---

    Questions, bug reports, and discussion.

    [:octicons-arrow-right-24: Getting help](community/getting-help.md)

</div>
