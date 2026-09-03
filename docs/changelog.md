# What's new

Changes per release, newest first. Breaking changes are marked **breaking**.

## Unreleased (next: 1.14.0)

### Runtime checks: `assert` follows Python, `dynaplex.fail`, fast mode and rehearsal

- **`assert` is a check, as in Python.** Compiled code now runs in one of two modes:
  *checked*, where every `assert`, `if __debug__:` block and list bounds check is live,
  and *fast*, where they are compiled out (Python `-O` semantics; an index out of range
  is then undefined behaviour, the C release-build trade). Fast mode makes the
  featurization and policy loops branch-free: +32 % on the vehicle-routing feature
  row, +11–17 % on the other featurized rows, bit-identical results.
- **`dynaplex.fail(msg)`** is the way to fail on purpose: it raises `DynaPlexError(msg)`
  in every mode (and in CPython) and never returns, so a method may end on it. Use it
  for the branch that must not be reached; keep `assert` for invariants.
- **`__debug__`** is the compile-time checks flag; `if __debug__:` compiles only the
  taken branch.
- **`checks=`** on `Engine`/`Program` (`True`/`False`, default checked) and on
  `PolicyComparer`, `DCL` and `VectorEnv` (`True`/`False`/`"rehearsal"`). The algorithms
  default to **`"rehearsal"`**: a short seeded pass through their own kernels compiled
  checked, then the fast compile. **`DYNAPLEX_CHECKS=0|1`** overrides any default from
  the environment (rerun a misbehaving script fully checked, no code change).
- **`dynaplex.rehearse(mdp, policy, features=..., trajectories=...)`** — the checked pass
  as a standalone model checker, plus a bytewise CPython-vs-compiled parity comparison.
- The algorithms' own invariants (root-pool exhaustion, `max_steps`, an infinite-horizon
  MDP reaching FINAL, a state with no valid action) now use `dynaplex.fail` and fire in
  every mode.

### DCL on GPUs and cluster nodes: machine plans

- **`DCL(machine=...)`** selects an execution plan — processes (one per GPU), engines,
  workers, slots, the device the network runs on — engineered per machine: the Snellius
  H100 and A100 nodes in quarter/half/whole units, a whole Genoa node, and `laptop`, which
  is derived from what the machine has (Metal on Apple silicon, cuda elsewhere, else cpu).
  `machine="auto"` recognizes the allocation among those plans and refuses anything else;
  every plan is verified against the allocation before a collection starts. The default
  (`machine=None`) is unchanged. See *Training and evaluation → Machines and execution plans*.
- **`dynaplex.machine`**: hardware-vs-allocation detection (affinity mask, cgroup limits,
  visible GPUs, SLURM variables); `python -m dynaplex.machine` prints it.
- **Generation ≥ 1 collection** now alternates engines so the network's forward for one
  engine overlaps the other engine's simulation, with asynchronous device bindings (pinned
  buffers; on cuda one CUDA-graph replay per round) and `processes=P` spawning one
  controller group per GPU. Results are byte-identical to the previous single-loop
  collection on the same device. Measured on the lot-sizing example (K=5, 150 000 samples
  per generation): 23 minutes per generation on one H100 node, where the previous loop
  needed hours.

### Model validation: classes carry promises

- **`@mdp` and `@policy`** (importable from `dynaplex`; apply above `@const_dataclass`):
  the class promises the MDP / policy contract for its methods — the cost accumulator is
  `+=`/`-=`-only, `time_elapsed` is `+=`-only, no generator is used in
  `modify_state_with_action`, generators are never re-bound or stored, `write_action_validity`
  and `get_action` only read the state, a policy draws from `policy_rng` only and never reads
  the cost. `@featurizer` promises that `write_features` only reads the state. The compiler
  checks every promise whenever the methods compile; a violation is a compile error naming
  the model's line and explaining the rule. Undecorated classes are unchanged. See
  *Language reference → Promises*.
- **`@mdp(promises=...)`** appends an algorithm's set; `dynaplex.validation` holds the
  vocabulary (`Param`/`Path`/`OfType` selectors, `NoUse`/`ReadOnly`/`Accumulate`/`BindOnce`/
  `SingleCall` constraints, `RoleContract`), `validate(engine, contract)` for a non-raising
  verdict, and `DISCRETE_IDENTIFIABLE_EVENTS`, the set the upcoming exact solver requires.
- The shipped models (`dynaplex.models`) and the examples now carry `@mdp` / `@policy`.
- **breaking** — `PolicyComparer(backend=...)` and `gym.VectorEnv(backend=...)` default to
  `"engine"`: a model that fails to compile, or violates a promise, raises instead of
  silently running on the Python kernel. `"auto"` remains available as the explicit
  opt-in to that fallback and now warns when it falls back. Violations raise
  `dynaplex.validation.PromiseViolation` (a `ValueError`).

### Compile-time configuration: `static()` and value folding

- **`static(expr)`** (`dynaplex.modelling.static`): a compile-time branch decision.
  `if static(self.mdp.has_lead_time):` selects the branch from the configuration the
  engine was constructed with; the untaken branch is discarded *before* type-checking,
  so one model body can serve a whole MDP family whose variants have different fields.
  `expr` must be statically known (a chain of `Final`/`@const_dataclass` hops from the
  engine root, or from a world's known constructor arguments); otherwise compilation
  fails with a message naming the hop or the call/spawn site that broke the chain.
  See *Advanced → Compile-time configuration with `static()`*.
- **Specialization.** Methods called on a known configuration object, and
  `@separate_world_class` worlds created with known constructor arguments, are compiled
  once per distinct configuration. Known configuration values (`self.mdp.scale`,
  `self.horizon` in such a world) compile to constants, so the JIT can fold compares,
  dead branches and constant loop bounds. Results are bit-identical to unspecialized code.
- `DYNAPLEX_DISABLE_FOLDING=1` disables all of the above (models using `static()` then
  fail to compile, deliberately).

### Performance: faster compiled kernels

- The LLVM backend now runs a target-aware pass pipeline with an aggressive inline
  threshold, marks error exits cold, gives the kernel context precise pointer attributes,
  lowers 2-arg `min`/`max`, `heapq.heappush`/`heappop` (int, float and orderable-object
  lists), the exact unary math functions (`sqrt`, `floor`, `ceil`, `round`, `abs`,
  `isnan`, `isfinite`), list membership (`in`) and list equality to inline IR, tells LLVM
  which builtin calls cannot touch list/array memory, and allocates objects (`ClassName(...)`) inline on a freelist fast
  path instead of through a runtime call. Cached slab pointers now live in registers across
  loops and calls (refreshed only where an allocation can actually move a buffer).
- List element storage is an arena with an explicit fill mark: appends and list creation no
  longer zero-fill freshly reserved segments (a list's contents are exactly its `len`
  elements; anything beyond is unspecified and never read).
- Results are bit-identical to the interpreter and to CPython. Single-worker throughput on
  the reference workloads: lost-sales +16%, bin-packing +22%, object-heavy
  event-simulation models +19–21%.
- A process that JIT-compiled at least one kernel could crash with SIGSEGV at interpreter
  exit in Debug builds (static-destruction order between the backend's LLJIT and an LLVM
  global); the JIT is now deliberately left alive at exit.

### Fixes

- The documentation links in compile-error and validation messages now point at the docs
  for the **installed** version (`.../DynaPlex/1.14/...`); a development build links to
  `latest`. Previously every such link was unversioned and 404ed on the versioned site.
- Enum-typed dataclass fields with a default (`act: Action = Action.UP`) can now be
  omitted in constructor calls inside compiled code; previously the default was rejected
  with "expected Enum(...), got Int".
- The C++ instruction dump (`--cpp` in inspection tooling) decoded `CALL` operands
  incorrectly after value-returning calls; fixed (development tooling only).

## 1.13.0

**Platform**

- **Python 3.11 and 3.12 supported** (previously 3.13+). Wheels ship for
  CPython 3.11–3.14. One documented semantic note: generated `__eq__` on
  model classes follows CPython 3.13 dataclass semantics (per-field `==`,
  no identity shortcut for a shared NaN or non-reflexive field object) on
  every Python version.

**Modelling**

- **`Featurizer` base class.** Featurizers now subclass `dynaplex.Featurizer`
  (`@featurizer @dataclass(slots=True) class Mine(Featurizer): ...`), mirroring
  `TrajectoryContext`. The base carries no fields and no behaviour; it exists so
  type checkers accept a featurizer where `features=` is expected
  (`DCL`, `PPO`, `VectorEnv`) and see the synthesized `install`/`reset`/
  `finish`/`spec`/`Holder`. Existing featurizers keep working at runtime
  without the base — only the static type changes. *Breaking for type
  checkers only.*

- **Trajectory-context statistics.** An MDP can declare its own trajectory
  context — a `@trajectory_context` dataclass carrying per-trajectory
  statistics (scalars and fixed-shape arrays) — and construct it in
  `make_context(self)`. `PolicyComparer` returns them as
  `assessment.stats`, one `[n, *shape]` array per statistic, rows aligned
  across policies under common random numbers. The holder is a dataclass
  you declare (`class MyStats(TrajectoryStats)`, named on the context as
  `Stats = MyStats`), so `PolicyComparer(mdp)` is a `PolicyComparer[MyStats]`
  and `assessment.stats` is typed in the editor; leave `Stats` out and a
  holder is derived (untyped).
  See the [tutorial](advanced/airplane-statistics.md) and the
  [language reference](reference/language-reference.md#trajectory-contexts-and-statistics).
- **Context scratch.** `Scratch[...]` marks a context scalar or array as
  per-trajectory scratch — carried by clones, but neither collected into
  `Stats` nor touched by `reset()`: it belongs to the context-driven policy
  that uses it (a plan computed once per period), which decides from the
  state when it is stale. Policy bookkeeping without putting it in the state.
- **Dataclass inheritance** (one base, one level): fields and methods are
  inherited as in CPython, overrides may call the base version
  (`super().m(...)` / `Base.m(self, ...)`), and a hand-written subclass
  `__init__` calls the base's. Custom trajectory contexts now inherit from
  `TrajectoryContext` instead of repeating its five members; `slots=True` is
  no longer required on a context. `@trajectory_context` rejects, at class
  definition, a subclass whose `@dataclass` generated an `__init__` over the
  base's hand-written one (the mix the compiler also rejects; in plain
  CPython it silently yields a positional-over-all-fields constructor).
  [Reference](reference/language-reference.md#inheritance).
- **Custom `__init__` and `__post_init__`** on DynaML dataclasses (states,
  contexts, helper objects): construct with your own signature, fields are
  checked for definite assignment at compile time.
  [Reference](reference/language-reference.md#custom-__init__-and-__post_init__).
- **For-loop targets are scoped to the loop**: using the loop variable after
  the loop (or shadowing an outer name) is rejected, matching what compiled
  code actually did.
- `GlobalStateWriter(h.v)` — the writer sizes itself from its array;
  hand-written `install` bodies no longer pass the width.

**Evaluation**

- `PolicyAssessment.stats` (see above). For the vanilla context this holds
  `cumulative_cost` and `time_elapsed` per trajectory.
- Infinite-horizon evaluation resets the context (not the streams) at the
  warmup boundary; window values restart from zero.

**Tooling**

- `dynaplex.set_jit_opt_level(level)` selects the LLVM optimisation level
  for compiled kernels (default 2; 0 compiles ~5x faster with bit-identical
  results — the test suite runs at 0).

**Breaking**

- `make_context(mdp, seed)` and `reseed_context(...)` are removed. Construct a
  seeded context with `dynaplex.modelling.new_context(mdp, seed)` — the MDP's
  own context when it declares `make_context()`, else a `TrajectoryContext`
  (`context.reseed(seed)` re-seeds an existing one).
- `TrajectoryContext` is no longer constructed positionally
  (`TrajectoryContext(rng, rng, 0.0, 0, valid)`); use `TrajectoryContext(mdp)`.
- A context-driven policy must annotate `context` with the MDP's context
  class (`TrajectoryContext` for MDPs without a custom context).

**Fixes**

- The airplane tutorial MDP now compiles on the engine (it silently ran on
  the Python backend before) and draws customer types from a
  `DiscreteDist` / `AliasSampler`.
- CPython `clone()` of an object whose class has a hand-written `__init__`.

## 1.12.3

- First public release on PyPI.
