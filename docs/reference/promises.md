# What DynaPlex expects from an MDP and a policy

DynaPlex's harnesses — the policy comparer, DCL, PPO, the exact solver — rely on
a handful of assumptions about what an MDP's methods and a policy's `get_action`
do. None of them is visible in a signature, and a model that breaks one still
compiles and runs; it just gives wrong answers, quietly. This page states those
expectations in prose, set by set: what we expect from an **MDP**, from a
**policy**, from a **featurizer**, and — for the exact solver — from an MDP with
**discrete identifiable events**. One promise, [state-read-only](#state-read-only),
is shared by all three roles and has its own section. Each expectation is a named *promise*; a model
class carries its promises (`@mdp`, `@policy`, `@featurizer`), the compiler checks
each one whenever the promised method compiles, and a broken promise is reported
as an *MDP model validation* (or *policy model validation*) failure naming the
promise, the line and the rule, with a link back to its section here.

## How promises are checked

- **Statically, at compile time.** The compiler records, for every compiled
  method, what happens to each parameter: which fields are read, assigned,
  updated in place, handed to another function, or stored somewhere. A promise
  is a rule over those records. Nothing is executed to check it.
- **Transitively.** Passing the state to a helper that modifies it counts as
  modifying the state; `state.pipeline.pop_front()` counts, because
  `FifoQueue.pop_front` modifies the queue. Passing the rng to a sampler counts
  as using the rng.
- **Per method, per parameter.** A promise never attaches to a field or a class
  as such: the same context field is accumulate-only in the event function and
  freely readable in a policy.
- **By type where that matters.** Rules about random streams apply to *every*
  generator reachable from the context — `rng`, `policy_rng`, and any stream
  your own context class adds — not to a fixed list of names.
- **Only when the method compiles.** A role method that no harness calls (a
  state-only policy never triggers `write_action_validity`) is not compiled
  and therefore not checked in that run. Plain CPython execution of the same
  classes is never checked — as for every other DynaML rule.
- **Never silently bypassed.** With the default `backend="engine"`, a broken
  promise raises. `backend="auto"` is an explicit opt-in to falling back to
  the Python kernel, and it warns when it does.

An undecorated class carries no promises; nothing is checked for it.

## MDP

What we expect from every MDP class (`@mdp` above `@const_dataclass`): its
transition methods treat the context as the harness's bookkeeping and its query
methods leave the state alone. In detail:

### cost-accumulate-only

*Methods:* `modify_state_with_event`, `modify_state_with_action`.
*Rule:* `context.cumulative_cost` appears only as the target of `+=` or `-=`.

**Assumption.** The cost accumulator is an *output channel* owned by the
algorithm. The comparer resets it to zero at the end of warm-up; DCL zeroes it
per rollout; PPO reads and rebases it per step. A model that reads it makes its
dynamics depend on the harness's bookkeeping — the process is no longer an MDP
in its state — and a model that assigns it discards cost the harness has
already accounted.

**Breaks it:** `if context.cumulative_cost > 100:`, `context.cumulative_cost =
...`, `context.cumulative_cost *= ...`, passing the value to a helper.
**Fix:** if a decision or transition depends on cost so far, make that quantity
part of your State and maintain it there.

### time-accumulate-only

*Methods:* `modify_state_with_event`, `modify_state_with_action`.
*Rule:* `context.time_elapsed` appears only as the target of `+=`.

**Assumption.** Elapsed time is harness bookkeeping too: the comparer's warm-up
window and horizon are counted on it, and it may be rebased. Time-dependent
dynamics belong in the State (the airplane model keeps `remaining_days`).

**Breaks it:** reading it in a condition, assigning it, `-=`.
**Fix:** carry your own clock in the State.

### rng-bind-once

*Methods:* `modify_state_with_event`, `modify_state_with_action`.
*Rule:* no generator under `context` is re-bound, re-seeded, or stored into a
field or a container; the only thing a model does to a stream is draw from it.

**Assumption.** The harness owns the identity of every random stream: it
reseeds them in place per trajectory and, in DCL, recycles them across
rollouts. A generator aliased into long-lived storage would keep drawing from a
stream the harness believes it has replaced, and common-random-number
comparisons across policies would silently break.

**Breaks it:** `reseed(context.rng, 42)` (`reseed` is imported from
`dynaplex.runtime`), `state.saved_rng = context.rng`,
`some_list.append(context.rng)`. (Re-seeding inside a model would restart the
event stream every step — the costs it produces are meaningless.)
Re-binding the stream itself (`context.rng = default_rng(42)`) is refused one
layer earlier, as an assignment to a `Final` field — every context declares its
streams `Final[Generator]` — so you will see a compile error for that, not a
promise failure; the promise adds the aliasing rule that `Final` cannot express.
**Fix:** draw from the stream where you need it; never keep a reference.

### event-stream-only

*Method:* `modify_state_with_event`.
*Rule:* `context.policy_rng` is not used — not directly, not through a
sampler, not by handing it to a helper that draws.

**Assumption.** The policy stream belongs to the policy. The event step draws
from `context.rng` (or from a stream your own context adds, such as a separate
lead-time stream); a randomized policy draws from `context.policy_rng`. Keeping
the two apart is what makes the event sequence a function of the seed alone:
swap the policy, or make it draw more often, and the demands, arrivals and
failures stay the same. An event step that draws from the policy stream ties
the events to how often the policy happened to draw.

**Breaks it:** `context.policy_rng.random()`, `self.sampler.sample(context.policy_rng)`,
`self.helper.draw(context.policy_rng)` inside the event step.
**Fix:** use `context.rng`, or add a stream to your context for a second
independent source.

### action-deterministic

*Method:* `modify_state_with_action`.
*Rule:* no generator under `context` is used at all — not `rng`, not
`policy_rng`, not a stream your context adds; not directly, not through a
sampler.

**Assumption.** The transition on an action is a deterministic function of
(state, action). All randomness lives in `modify_state_with_event`, so that
two policies evaluated on the same seeds see exactly the same event stream —
this is what makes the comparer's paired comparisons and DCL's sample
generation valid. A random action transition would also make the exact
solver's enumeration wrong.

**Breaks it:** `context.rng.random()` or `self.sampler.sample(context.rng)`
inside the action step.
**Fix:** move the draw into the event step and, if needed, carry its outcome
in the State until the action step needs it.

An MDP also promises **state-read-only** for `write_action_validity`. That
promise is shared with policies and featurizers and stated once, in
[The state is read-only](#state-read-only).

## Policy

What we expect from every policy class (`@policy`): `get_action` is a pure
query of the state that, if randomized, draws from its own stream and never
looks at the harness's cost accounting.

A policy also promises **state-read-only** for `get_action` — see
[The state is read-only](#state-read-only).

### policy-stream-only

*Method:* `get_action(state, context)`.
*Rule:* of the generators under `context`, only `context.policy_rng` may be
used.

**Assumption.** A randomized policy draws from its own stream so that its
draws never perturb the *event* stream. Otherwise evaluating a randomized
policy would change the demands, arrivals or failures the model sees, and the
comparison against another policy on "the same" seeds would not be paired at
all.

**Breaks it:** `context.rng.random()`, `sampler.sample(context.rng)`, or a
user-added stream in `get_action`.
**Fix:** use `context.policy_rng`.

### policy-no-cost

*Method:* `get_action(state, context)`.
*Rule:* `context.cumulative_cost` is not read.

**Assumption.** The same as `cost-accumulate-only`: the accumulator is rebased
by the harness, so a decision that depends on it depends on bookkeeping.
`context.time_elapsed` *is* readable in a policy (a policy may want the time).

## Featurizer

What we expect from a featurizer (`@featurizer` attaches this by itself):
**state-read-only** for `write_features` — the featurizer describes the state,
it never changes it. See [The state is read-only](#state-read-only).

## The state is read-only

One promise is made by all three roles. The transition methods
(`modify_state_with_event`, `modify_state_with_action`) are the *only* methods
that change a state; everything else that receives a state is a **query** and
must leave it exactly as it found it.

### state-read-only

*Methods:*

| Role | Method | Attached by |
| --- | --- | --- |
| MDP | `write_action_validity(state, ...)` | `@mdp` |
| policy | `get_action(state)` / `get_action(state, context)` | `@policy` |
| featurizer | `write_features(state, ...)` | `@featurizer` |
| exact solver | `get_event_dist(state)` | `DISCRETE_IDENTIFIABLE_EVENTS` (as [event-dist-pure](#event-dist-pure)) |

*Rule:* the `state` parameter is only read — no field store, no in-place
container update, no mutating method call, no passing it to a helper that
modifies it, no storing it into another object.

**Assumption.** These methods are *queries*. The harnesses call them any
number of times per decision and in any order — the validity mask, the
features and the policy are all asked about the *same* state, and the exact
solver enumerates from states it visits again. All of that assumes the state
is unchanged afterwards. A mutation in a query changes the trajectory in ways
no harness can account for, and in ways that depend on which harness is
running, so the same model would behave differently under the comparer than
under DCL.

**Breaks it:** `state.x = 0`, `state.items.append(...)`, `state.queue.pop_front()`,
`self.helper.advance(state)`, `s = state; s.x += 1`.
**Fix:** compute what you need into locals; if a query needs derived
quantities, cache them in the State during the transition methods, where
mutation belongs.

The failure message names the role whose promise was broken (*"breaks the MDP
promise 'state-read-only' in LostSales.write_action_validity"*, *"the policy
promise ..."*), so you can tell which method to fix; the rule is the same for
all of them.

## Discrete identifiable events

What the exact solver expects on top of the MDP promises.

The exact solver derives the event *distribution* from `modify_state_with_event`
and enumerates it. That needs more than the base contract. The set is
`dynaplex.validation.DISCRETE_IDENTIFIABLE_EVENTS`: its name says exactly what
static analysis can establish — the event step is one identifiable draw from a
discrete distribution the state selects. It is *not* a promise of solvability:
whether the reachable state space is finite and small enough is something the
solver discovers at run time, and no static check can vouch for it. The solver
checks this set when constructed; a model can declare it up front with
`@mdp(promises=DISCRETE_IDENTIFIABLE_EVENTS)`.

### single-draw

*Method:* `modify_state_with_event`.
*Rule:* exactly one call receives a generator, it is the method's first
statement (unconditional), it is a `.sample` / `.conditional_sample` on a
`DiscreteDist`, `AliasSampler` or `CdfSampler`, and the sampler is a field or
list element reachable from `self` or `state` — `self.demand_sampler`,
`self.dists[state.regime]` — not the result of a call.

**Assumption.** The solver lifts the sampler expression verbatim into
`get_event_dist(self, state)` and replaces the draw by an enumerated value.
That is only exact if the draw is the one source of randomness, it always
happens, and the distribution is a materialized pmf that the state can select
but not compute.

**Breaks it:** two draws, a draw under `if`, `context.rng.random()`,
`DiscreteDist.poisson(rate).sample(...)` (a call in the receiver),
`self.make_dist(state).sample(...)`.
**Fix:** materialize every distribution once in `__init__` (a list indexed by
the state if it varies), draw first, then apply.

### deterministic-initial-state

*Method:* `get_initial_state`.
*Rule:* no generator under `context` is used.

**Assumption.** Enumeration starts from one state. Random initial states would
need a distribution over roots; not supported.

### event-dist-pure

*Method:* `get_event_dist(self, state)` (generated by the solver, or hand-written).
*Rule:* `state` is only read — the same rule as [state-read-only](#state-read-only).

## Writing your own promises

The vocabulary (`dynaplex.validation`): a **selector** — `Param("state")`,
`Path("context", "stats.counts")`, `OfType("context", Generator)`, combinable
with `-` to exclude a path — and a **constraint** — `NoUse()`, `ReadOnly()`,
`Accumulate({"+=", "-="})`, `BindOnce()`, `SingleCall(callees, at="entry")` —
make a `Promise(selector, constraint, message, name=...)`. A
`RoleContract({method_name: [promises]})` groups them per method; contracts add
with `+`.

```python
from dynaplex import mdp, const_dataclass
from dynaplex.validation import Accumulate, Path, Promise, RoleContract, validate

MY_STATS = RoleContract({
    "modify_state_with_event": [
        Promise(Path("context", "stockouts"), Accumulate({"+="}),
                "stockouts is a statistic: count, never read", name="stockouts-count-only"),
    ],
})

@mdp(promises=MY_STATS)
@const_dataclass(init=False, slots=True)
class MyMDP:
    ...

verdict = validate(engine, MY_STATS, cls=MyMDP)   # query mode: no exception
print(verdict.format())
```

## Reading a validation failure

```
dynaplex.validation.PromiseViolation: MDP model validation failed:
In function 'LostSales.modify_state_with_action@k1$State$TrajectoryContext$Int' (my_model.py), line 3:
context.rng is used (load) — breaks the MDP promise 'action-deterministic' (NoUse on every Generator under context).
  modify_state_with_action must be deterministic given (state, action): randomness belongs in
  modify_state_with_event, so every policy sees the same event stream (common random numbers).
  Line 3:     state.on_hand -= int(context.rng.random() * 2)
MDP model validation failed: the model breaks the MDP promise 'action-deterministic' in
LostSales.modify_state_with_action; what DynaPlex
expects from an MDP and a policy is stated at https://dynaplex.github.io/DynaPlex/reference/promises/#action-deterministic
```

The function name carries the compiler's variant suffix (`@k1`, `$State...`):
one source method may compile several times. Line numbers are relative to the
method (`def` is line 1), and the offending source line is printed.
