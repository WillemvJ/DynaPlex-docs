# Machines: running DCL on a laptop or a cluster node

Deep Controlled Learning collects samples by simulating rollouts on many CPU cores while a
neural network decides the actions. From generation 1 on, that network runs on whatever
accelerator the machine has. How many worker threads, how many engines, how many
processes, and where the network lives are *execution* choices: they change how fast a
collection runs, not what it computes. DynaPlex packages those choices as a **machine
plan**, and `DCL(machine=...)` selects one.

!!! note "Since 1.14"
    Machine plans, `dynaplex.machine`, and the `processes` / `policy_binding` arguments of
    `collect_samples` are new in 1.14. Earlier versions only had the explicit knobs
    (`controllers`, `workers`, `slots`), which still work unchanged.

## Three ways to say where a run happens

```python
dcl = dynaplex.DCL(mdp, heuristic, features=MyFeaturizer, n=150_000, m=1000, h=30,
                   machine="auto")        # recognize this allocation (the default)
```

| `machine=` | what happens |
|---|---|
| `"auto"` (default) | DynaPlex detects the allocation and **recognizes** which plan it is: `laptop` on macOS and Windows, one of the cluster presets on Linux. It never invents one: an allocation that matches no plan raises `MachineMismatchError` rather than run on a shape nobody tuned and look slow; see [When no plan matches](#when-no-plan-matches). |
| a plan name | That plan's numbers fill every knob you left unset; explicit knobs override. The plan is **verified** against the allocation when a collection starts. |
| `None` | No detection, no plan. The explicit `controllers`, `workers`, `slots` are used as given; defaults 1, 1, 512 on the cpu. |
| a `dynaplex.machine.DclPlan` | Your own plan object, verified like a named one. |

A plan is part of the experiment identity: `controllers` in the run record is the total number
of engines and `slots_per_controller` is `workers * slots`, so two different plans are two
different experiments.

### When no plan matches

`machine="auto"` on an allocation DynaPlex does not recognize stops before any work is
done, with a `MachineMismatchError` that names what it detected (host, allowed CPUs, cuda
devices, SLURM partition) and the plans that exist. You then have three ways forward, in
order of preference:

1. **A Linux desktop or workstation:** pass `machine="laptop"`. It is derived from the
   cores and accelerator present and fits any single machine; `auto` only refuses to
   guess it on Linux because that is where cluster nodes live.
2. **A cluster allocation of a different shape:** request one of the allocations a
   preset was engineered for (the `sbatch` lines are in the plan notes above), or write a
   `dynaplex.machine.DclPlan` of your own for the shape you have. The rules the presets
   follow are in
   [Choosing a plan for a machine without a preset](#choosing-a-plan-for-a-machine-without-a-preset).
3. **Just run:** `machine=None` switches detection off and uses the explicit
   `controllers`, `workers`, `slots` (defaults 1, 1, 512 on the cpu) as given. Fine for a
   smoke test; expect it to be slow at scale.

And in every case, please [open an issue](https://github.com/DynaPlex/DynaPlex/issues)
with the output of `python -m dynaplex.machine --json`. A machine that people run DCL on
should have a tuned plan, and the detection record is what is needed to write one.

## The plans

A plan fixes: **processes** (one per GPU), **engines** per process, **workers** per engine,
**slots** per worker, the **device** the network runs on, and the **policy binding** (how
actions get from the network to the engines; see below).

### Cluster units (fixed, engineered)

These match the allocation units SLURM hands out on Snellius, and they exist in quarter,
half and whole variants with the same per-GPU shape:

| plan | allocation | processes × engines × workers | device |
|---|---|---|---|
| `gpu_h100_quarter` | 1 H100, 16 cores | 1 × 2 × 7 | cuda |
| `gpu_h100_half` | 2 H100, 32 cores | 2 × 2 × 7 | cuda |
| `gpu_h100` | 4 H100, 64 cores (whole node) | 4 × 2 × 7 | cuda |
| `gpu_a100_quarter` | 1 A100, 18 cores | 1 × 2 × 8 | cuda |
| `gpu_a100_half` | 2 A100, 36 cores | 2 × 2 × 8 | cuda |
| `gpu_a100` | 4 A100, 72 cores (whole node) | 4 × 2 × 8 | cuda |
| `genoa` | 192-core Genoa node | 32 × 1 × 4 | cpu |

The matching `sbatch` line is in each plan's note (`dynaplex.machine.plan("gpu_a100").describe()`),
for example `--partition=gpu_a100 --gpus=4 --cpus-per-task=72 --exclusive` for the whole
A100 node. A quarter node is the smallest unit the scheduler gives out and usually starts
within minutes; whole nodes wait for a node to drain.

All GPU units use 1024 rollout slots per worker (the measured optimum: the barrier per
engine step amortizes, and larger slabs gain nothing further on the cluster).
Throughput is linear in the total number of workers, and one GPU keeps up with far more
workers than one unit has, so pick the unit by how many cores you can get, not by GPU count.
Measured on a stochastic lot-sizing example (K=5, 150 000 samples, m=1000, h=30): 23 minutes per
generation on a whole H100 node, 25 on a whole A100 node, 45 on half an H100 node.

`genoa` is the CPU unit, and its shape follows from the network running on the same cores as
the workers: **32 processes of one engine with 4 workers each**, 512 slots per worker, and 2
torch threads per process. The rule the sweeps taught is `processes × (workers + torch
threads) ≤ cores` — workers spin at the barrier, so the forward's threads need cores of their
own rather than borrowing idle ones. Two torch threads beat every larger number once there
are 16 or more processes; fewer, fatter processes make the per-process forward the
bottleneck. Measured on 2000 generation-1 samples (m=1000, h=30, `[128, 128]`): 15 s for lost
sales and 38 s for lot sizing K=5, against 271 s and 1002 s for the single engine with 190
workers that this plan used before. Generation 0 has no forward to place and runs the same
32 processes of one engine each, sharing the identity with generation 1 on.

### `laptop` (derived from what the machine has)

`laptop` is the one plan that is computed rather than fixed, so it fits any Mac, Windows or
Linux machine you run it on:

1. Detect an accelerator: cuda through torch (a CPU-only torch build counts as **no** GPU),
   or Metal (`mps`) on Apple silicon.
2. With an accelerator and at least 6 cores: one process, **two engines** on the accelerator,
   `cores - 2` workers split over them. Two engines let the network run on one engine's
   batch while the other engine's workers simulate; a single engine on an accelerator is
   slower than the CPU because of launch overhead.
3. Otherwise: one engine on the cpu with `cores - 2` workers.

With an accelerator the plan uses 4096 slots per worker: the launch cost per round on Metal
is fixed, so larger slabs keep paying (measured on an M4 Pro: 512 to 4096 slots is a 3x
throughput difference). Cores are Apple performance cores where known, else the logical
CPUs. The plan's note says
what was chosen and why, e.g. `derived on mymac (Darwin): 10 performance cores; using mps,
2 engines` or `...; cpu only — torch 2.13.0+cpu is a CPU-only build (install the CUDA wheel
to use a GPU)`. `macbook`, `ms-laptop`, `mac` and `local` are aliases of `laptop`.

`machine="auto"` resolves to `laptop` on macOS and Windows. On Linux it only recognizes the
cluster units above; pass `"laptop"` explicitly on a Linux desktop.

## Seeing what is detected

```bash
python -m dynaplex.machine          # human summary; --json for the record; --no-torch to skip torch
```

```
machine: gcn159 (Linux; AMD EPYC 9334 32-Core Processor)
  hardware:   64 logical CPUs / 64 physical cores; 756 GiB RAM; 2 NUMA nodes; 1 GPU(s) on node
  allocation: 16 logical CPUs / 16 physical cores = PART of the node; 1 NUMA node(s); memory limit 180 GiB
              CPU ids 16-31
  gpu:        cuda:0 NVIDIA H100, 93 GiB
  scheduler:  SLURM_JOB_ID=..., SLURM_CPUS_PER_TASK=16, SLURM_GPUS=1, CUDA_VISIBLE_DEVICES=0, ...
```

Detection separates the **hardware** (what the node has) from the **allocation** (what this
process may use: the CPU affinity mask the scheduler enforces, the cgroup memory limit, the
GPUs torch can see). Plans are verified against the allocation, so a plan engineered for a
whole node refuses to run on a quarter of one instead of silently oversubscribing it.
`dynaplex.machine.detect()` returns the same information as a `Machine` record.

## What runs where

Under a plan, each stage of a DCL generation has a fixed home:

| stage | where | notes |
|---|---|---|
| generation-0 rollouts (heuristic policy) | CPU workers, the plan's processes and engines | the policy is compiled into the engine; no accelerator involved |
| generation ≥ 1 rollouts: simulation, featurization, action masks | CPU workers | the bulk of the work; throughput is linear in workers |
| generation ≥ 1 rollouts: the network's forward + masked argmax | the plan's device | cuda: one CUDA-graph replay per engine per round; mps: asynchronous device binding; cpu: blocking |
| training the next network | the plan's device | data moves to the device once; the minibatch order comes from the seeded CPU generator, so it is the same on every device; `train=dict(device=...)` overrides |
| evaluation (`PolicyComparer`) | CPU workers, single engine; the agent's device for the forward | no plan needed; move the agent with `agent.to("cuda")` if you want |

On a cluster GPU plan everything that can use the GPU does. On `laptop` the same holds
for whatever accelerator was found. Note that for small networks at the default batch
size of 64, training on an accelerator is not necessarily faster than on the cpu; the
per-step launch overhead dominates a 24k-weight MLP. It is still the plan's device by
default so that a run's artifacts come from one place; override with `train=`.

## What happens inside a collection

**Processes.** With `processes = P`, DynaPlex spawns P worker processes, each owning a
contiguous group of engines, a contiguous slice of the allowed CPUs, and one GPU
(`cuda:0`, `cuda:1`, ...). Each process collects its own share of the samples; the shares are
concatenated at the end. The result is byte-for-byte the same as one process running all
the engines, so the process count is a pure speed knob.

**Engines and the policy binding.** Within a process the engines alternate: while engine A's
workers simulate, the network computes engine B's actions. The *binding* is how that call is
made, chosen automatically from the device:

| device | binding | mechanism |
|---|---|---|
| cpu | blocking | one `agent.act` call per engine |
| mps | device | asynchronous forward with reused buffers, an event per engine |
| cuda | graph | the whole copy-forward-argmax-copy sequence captured once as a CUDA graph and replayed per round |

All three produce identical actions on a given device. `collect_samples(policy_binding=...)`
overrides the choice.

**Precision of the policy call.** For attention-sized networks the forward is arithmetic, not
overhead, and `policy_precision` (`"fp32"` default, `"tf32"`, `"fp16"`, `"bf16"`; on DCL, a plan
or `collect_samples`) trades rounding for speed: tf32 uses the tensor cores for every matmul,
fp16/bf16 run the forward under autocast with the masked argmax still in fp32. Measured on an
A100 for a 69-token transformer policy over 8192 states: fp32 40 ms, tf32 23 ms, fp16 18 ms,
with 100% (tf32, fp16) and 99.95% (bf16) of the chosen actions unchanged. Features and stored
samples stay fp32; the cast happens on the device. Lower precision can move near-tie actions,
as a different device can.

**Generation 0** (heuristic rollouts) runs the rollout policy inside the engines, on the cpu,
in the plan's processes; the plan's GPU settings only matter from generation 1.

**Checks and the rehearsal.** `DCL(checks=...)` applies to the generation-0 collection only.
The default, `"rehearsal"`, first runs that collection once with every runtime check on at a
tiny budget, so an out-of-range index or a failing `assert` in the model or featurizer is
reported before the fast compile. The rehearsal is not a model checker: run
[`check_mdp`](../reference/api/evaluation.md) for that. Two model faults that used to stall
a collection fail with a named cause in every mode instead: a decision state with no valid
action, and an MDP whose decision states never offer more than one valid action (nothing
to label).
Generations 1 and up always collect fast: a fault the generation-0 rehearsal does not
surface is unlikely to surface under a later policy, and checking every generation would
cost speed. A resumed run that skips generation 0 does not rehearse.

## Choosing a plan for a machine without a preset

When no preset matches your machine or your allocation, pass `machine=None` and the knobs
to `DCL(...)` (or `collect_samples`, which takes no plan) directly: `processes`,
`controllers` (engines per process), `workers` (per engine), `slots` (per worker) and, on
the cpu, `torch_threads`. The rules below are the
ones the cluster presets were tuned by.

**One engine per process, two to drive a GPU.** Engines inside one process step in turn,
so a third engine adds no parallelism, only build time and memory; `controllers` above 2 is
refused. Use one engine per process on the cpu. Use two engines per process on an
accelerator: while the network computes engine A's actions, engine B's workers simulate,
and a single engine on an accelerator idles the device between launches. Parallelism comes
from `processes`: each process builds its own engine, in parallel, and the result is
byte-identical to any other split with the same total number of engines and the same
`workers * slots`.

**Cores.** Workers spin at the round barrier, so give them cores of their own and count
the network's threads next to them:

```
processes × (workers + torch_threads) ≤ cores        (cpu network)
processes = number of GPUs; workers ≈ cores per GPU / 2 per engine   (accelerator)
```

On the cpu, 2 torch threads per process is enough once there are 16 or more processes;
fewer, fatter processes make each process's forward the bottleneck. Start from the `genoa`
shape scaled to your core count: with `c` cores, `processes = c / 6`, one engine, 4
workers, 2 torch threads. On a GPU, one process per GPU with two engines and half the
process's cores per engine; giving each engine all of the cores oversubscribes them (measured:
almost five times slower on a quarter H100 node).

**Slots.** 512 per worker on the cpu; 1024 on a cluster GPU; up to 4096 on Metal, where the
per-round launch cost is fixed and larger slabs keep paying. Slots trade memory for a longer
round between barriers; past the optimum nothing is gained.

**Samples per controller.** `n` must reach the total number of engines, and each engine
collects `n / engines` samples along a few on-policy trajectories. On a finite-horizon MDP,
keep that share well above the number of decisions in an episode, or the trajectories never
reach the end and the collection warns about possible data skew (the count per engine is in
the sample set's `manifest["execution"]["episodes_completed"]`).

**Record what you chose.** The execution block of every sample set records the split,
device, binding, precision and host, so a run can be reproduced on the same kind of machine.
Once a shape has proven itself, wrap it as a `DclPlan` and pass it as `machine=`; the plan is
then verified against the allocation on every run.

## Reproducibility

A collection is bit-reproducible for a fixed plan. Different plans draw the same
*distribution* of samples but not the same samples, in the same way that a different worker
count does. Across devices, near-tie action choices can differ between a cpu and a cuda
forward, so resume a run on the kind of machine it started on when byte-identity matters;
statistically the trained agents agree (on a stochastic lot-sizing example, four different plans gave
generation-2 agents within 0.6% of each other).
