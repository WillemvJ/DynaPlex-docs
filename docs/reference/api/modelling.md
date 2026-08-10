# Modelling MDPs

Everything you import when writing an MDP, a State class, or a policy.
These names live in `dynaplex.modelling`. Event distributions and random
number generation are on the
[distributions & randomness page](randomness.md); the language rules for
compiled code are in the [language reference](../language-reference.md).

## The MDP contract

There is no universal state class in DynaPlex. Each MDP defines its own
State: a plain dataclass holding exactly the quantities that describe the
system between decisions, plus one required field, `category`, which tells
the engine whether the state next expects an event or an action. For the
[airplane ticket MDP](../../tutorials/airplane-mdp.md), that looks like:

```python
@dataclass(slots=True)
class State:
    remaining_days: int
    remaining_seats: int
    price_offered_per_seat: int
    category: StateCategory = StateCategory.AWAIT_EVENT
```

In the signatures below, `StateType` is a type variable standing for
whatever State class *your* MDP defines — `MDPProtocol` is generic in it.
An MDP, then, is a class whose methods operate on that state: it creates
one (`get_initial_state`) and modifies it in place
(`modify_state_with_event`, `modify_state_with_action`,
`write_action_validity`). A policy operates on the same State class, and
this shared type is what ties them together: `assert_policy_for_mdp`
lets the type checker verify that a policy's `StateType` matches its
MDP's.

!!! note "What is a protocol, and why the `assert_*` functions?"
    A [protocol](https://typing.python.org/en/latest/spec/protocol.html) is
    Python's way of describing an interface by *shape* rather than by
    inheritance: any class that has the listed attributes and methods
    satisfies `MDPProtocol` — you never subclass it or import it into your
    model. That leaves one gap: nothing tells you when your class *doesn't*
    match, e.g. a misspelled method or a wrong signature. That is what
    `assert_mdp` and friends are for. They do nothing at runtime, but a call
    like `assert_mdp(mdp)` gives a type checker such as
    [pyright](https://microsoft.github.io/pyright/) a place to verify
    conformance, so `pyright my_model.py` reports exactly which member is
    missing or mistyped — before anything is run or compiled.

::: dynaplex.modelling.MDPProtocol

::: dynaplex.modelling.PolicyProtocol

## States and trajectories

::: dynaplex.modelling.StateCategory

::: dynaplex.modelling.HorizonType

::: dynaplex.modelling.TrajectoryContext

::: dynaplex.modelling.Validity

## Const classes and annotations

Compiled DynaML code relies on a few concepts that standard Python has no
syntax for — most importantly **constness**: MDP and policy objects are
read-only during a trajectory, and marking their classes with
`@const_dataclass` lets the compiler enforce and exploit that. Similarly,
the compiler needs some information that ordinary annotations do not
carry, such as the rank of a NumPy array field; the `Rank`, `Array*D` and
`ConstArray*D` annotations supply it. The names below are the API for
declaring all of this; what constness *means* inside compiled methods,
and the rules for array dtypes and ranks, are explained in the language
reference under
[const classes and read-only containers](../language-reference.md#const-classes-and-read-only-containers)
and [NumPy arrays](../language-reference.md#numpy-arrays).

::: dynaplex.modelling.const_dataclass

::: dynaplex.modelling.Const

::: dynaplex.modelling.Rank

::: dynaplex.modelling.Array1D

::: dynaplex.modelling.Array2D

::: dynaplex.modelling.Array3D

::: dynaplex.modelling.ConstList

::: dynaplex.modelling.ConstArray1D

::: dynaplex.modelling.ConstArray2D

::: dynaplex.modelling.ConstArray3D

## Copying state

::: dynaplex.modelling.clone

## Validating your model

::: dynaplex.modelling.assert_mdp

::: dynaplex.modelling.assert_policy_for_mdp

::: dynaplex.modelling.assert_featurizer_for_mdp

## Errors

::: dynaplex.DynaPlexError
