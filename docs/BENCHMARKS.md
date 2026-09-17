# Benchmarks

All numbers measured on this machine, 2026-09-16 to 09-18. Anything not measured is not stated.

**Hardware / software.** NVIDIA GeForce RTX 5080 (Blackwell, `sm_120`, 16 GB), driver 595.71 /
CUDA 13.2, WSL2 on Linux 6.6, Python 3.12.3, torch 2.14.0+cu130.

These measurements drove four design decisions, so they are recorded rather than summarised:
`num_envs` exists, the replay buffer is device-resident, gradient updates are few-and-large rather
than many-and-small and `play` defaults to CPU.

---

## 1. Inference latency vs batch size

Median µs per forward pass, `eval` mode under `no_grad`, for a 2x256 MLP of this shape (the shipped default is 17→128→128→3, which is smaller and faster still).

| batch | CPU (µs) | CUDA (µs) | winner |
|------:|---------:|----------:|--------|
| 1     | 15.8 | 72.7 | **CPU 4.6x** |
| 8     | 29.8 | 73.7 | CPU 2.5x |
| 64    | 60.5 | 90.8 | CPU 1.5x |
| 256   | 186.2 | 95.2 | CUDA 2.0x |
| 1024  | 453.2 | 90.7 | CUDA 5.0x |
| 4096  | 1491.6 | 99.6 | **CUDA 15x** |

The CUDA column is **flat from batch 1 to 4096** (73 → 100 µs). This network is kernel-launch
bound, not compute bound: at batch 1 you pay ~73 µs of launch overhead to do arithmetic the CPU
finishes in 16. The crossover is around batch 128-256.

**Consequence:** `num_envs` (default 256 on CUDA) exists purely to keep every policy query on the
right side of that crossover.

## 2. The synchronisation tax

Reading the chosen action back from the GPU forces `cudaStreamSynchronize`:

| operation | µs |
|---|---:|
| CUDA forward, batch 1, no readback | 74.8 |
| CUDA forward + `.argmax().item()` | **197.6** (+122.8 for the sync alone) |
| host→device copy of one observation | 25.1 |
| CPU forward + `.item()` | **17.6** |

The sync costs more than the forward pass it waits on. For single-observation action selection
CUDA is ~11x slower than CPU, which is why `snakeai/play.py` defaults to `--device cpu`.

## 3. Training step cost

Forward + backward + optimizer step, median µs.

| batch | CPU (µs) | CUDA (µs) | speedup |
|------:|---------:|----------:|--------:|
| 256   | 1122.5 | 789.9 | 1.4x |
| 1000  | 1940.3 | 797.8 | 2.4x |
| 4096  | 8921.5 | 786.6 | 11.3x |
| 16384 | 39074.9 | 810.3 | **48.2x** |

CUDA is again flat: **a batch-16384 update costs the same wall-clock as a batch-256 update.** On
GPU, batch size is free and update *count* is what costs.

**Consequence:** the training loop runs `updates_per_iter` gradient steps of `batch_size` each
after every round of moves. Few large updates cost far less than many small ones. Getting this wrong was measurable: an early version did 64
separate batch-128 updates per iteration and ran at **2,508 steps/s**; batching them into 2 ×
batch-4096 took the identical workload to **26,758 steps/s**, a 10.7x gain.

## 4. The environment is not the bottleneck

Pure-Python environment, rendering and plotting stripped:

| operation | rate |
|---|---:|
| `Game.update()` (v1) | 742,288 /s |
| `Game.get_state()` (v1) | 1,408,493 /s |
| one full env step (update + 2 × get_state) | ~2.8 µs → **~361,000 /s** |

v1 ran its training loop under `clock.tick(120)` with a blocking `plt.pause(0.1)` on every death.
**The frame-rate cap alone throttled training by roughly 3,000x.** Removing it was worth more than
any device choice, which is why `snakeai/train.py` contains no frame rate at all.

## 5. End-to-end training throughput

v2, `small` preset (8×8 board), measured over full runs:

| config | env steps/s | vs v1 (120/s) |
|---|---:|---:|
| cuda × 1024 | 75,250 | 627x |
| cuda × 256 | 39,399 | 328x |
| cpu × 64 | 15,542 | 130x |
| cpu × 8 | 3,925 | 33x |

## 6. What actually matters: learning per wall-clock second

Throughput is not the objective. Each config below got ~45 s of wall clock, then 50 greedy
evaluation episodes on identical seeds.

| config | steps | wall (s) | steps/s | gradient updates | **eval score** |
|---|---:|---:|---:|---:|---:|
| **cuda × 256** | 1,200,128 | 31.5 | 39,399 | 9,346 | **17.60** |
| cuda × 1024 | 3,300,352 | 43.9 | 75,250 | 6,416 | 17.20 |
| cpu × 64 | 690,048 | 44.4 | 15,542 | 21,534 | 16.80 |
| cpu × 8 | 160,000 | 40.8 | 3,925 | 39,876 | 15.24 |

Two things worth noting. **CUDA × 256 wins and it won in 31.5 s against CPU's 44.4 s.** And
`cpu × 8` ran **four times as many gradient updates** as the winner and still scored worst. More
updates is not better, which is what licenses the large-batch trade in §3.

`cuda × 1024` is the visible edge of that trade: highest throughput, fewest updates, slightly worse
score. Very large `num_envs` buys env steps you no longer have the gradient steps to learn from.
256 is the default because it sits at the top of this table, not because it is a round number.

## 7. Replay buffer

A 200k-transition deque of Python tuples, sampled with `random.sample`, cost 1.08 ms per
batch-1000 draw versus 0.19 ms for a list, 6x slower, because `deque` indexing walks the block
chain. Real, but modest: it permitted ~3,700 env steps/s, well above v1's 120/s cap, so it was
never the binding constraint. It is gone anyway: v2 preallocates device tensors and samples with
an on-device `randint` + gather, with no host transfer at all.

Footprint is negligible. 100k transitions of a 17-wide observation is **13.6 MB**, so the
buffer lives in VRAM without thought. The shipped 1M capacity is ~136 MB of 16 GB.

## 8. Compiling the region labelling

The environment works out how much room each move leads into and whether the tail is still
reachable. Both come from splitting the empty cells into connected regions. In pure Python that
was 57 ms per 256 games and 73% of a training iteration.

Measured on 256 real boards with snakes averaging 40 cells, every variant correct:

| labelling of the empty cells | ms per 256 games |
| --- | ---: |
| pure Python, two separate floods | 52.8 |
| compiled, two separate floods | 2.25 |
| compiled, one labelling pass | 2.56 |
| compiled, one merged flood | 2.55 |

Once compiled the algorithm barely matters, so the labelling pass was chosen for being the
simplest: no budget inside the walk, no early exit, no reuse shortcut and no state carried
between calls. Both features become lookups against the same labels, so they cannot disagree.

Threads were measured too. The labelling itself scales nearly linearly, 1.29 ms on one thread to
0.12 ms on 24, but about 0.9 ms of surrounding Python does not parallelise and dominates almost
immediately. Ten threads bought roughly 8% on the whole iteration, so it was not worth the
complexity.

End to end on the 40x30 board: **3,235 to 14,708 environment steps per second, about 4.5x.**
With the floods compiled the remaining costs are the gradient updates and the snake dynamics,
roughly evenly split. The observation is no longer the bottleneck.

## 9. Where the gradient update time actually goes

§3 said batch size is free on GPU. Measured again on the shipped 17-wide network, it is flat over
a 32x range:

| batch | ms per update |
|------:|--------------:|
| 256   | 1.397 |
| 1024  | 1.313 |
| 2048  | 1.269 |
| 8192  | 1.309 |

One update launches about **72 CUDA kernels at roughly 20 µs each**, which is the entire 1.3 ms.
The GPU is idle waiting on launches, so the cost is the launch count and nothing else.

That makes the fix the update *count*, not the arithmetic:

| approach | ms per update | gain |
|---|---:|---|
| baseline | 1.476 | |
| fused Adam | 1.356 | 1.09x |
| `torch.compile(mode="reduce-overhead")` | 1.272 | 1.16x |
| both | 1.216 | 1.21x |
| TF32 | 1.360 | **none, 0.99x** |

TF32 and `torch.compile` both disappoint for the same reason: the network is far too small to be
matmul bound, so making the arithmetic cheaper changes nothing. Compiling only puts the two
forward passes into CUDA graphs and leaves the backward, the optimiser and the buffer indexing as
loose launches.

Halving the update count is worth more than all of it. Moving from `8 x 2048` to `2 x 8192`
trains on the same 16,384 samples per iteration for a quarter of the launches, measured at
**14,800 to 22,600 env steps/s, 1.53x** on the 40x30 board. It is not free in learning terms,
since 2 fat gradient steps are not equivalent to 8 thin ones, so it is a trade rather than a win.

## 10. Evaluation is nearly free to make less noisy

Eval runs `eval_episodes` games in parallel, so the cost is the number of loop iterations rather
than the number of games. Cost of one iteration against how many games run at once:

| games | ms per iteration | vs 20 games |
|------:|-----------------:|---|
| 20    | 1.524 | 1.00x |
| 64    | 1.970 | **1.29x** |
| 128   | 4.871 | 3.20x |
| 256   | 7.872 | 5.17x |

Going from 20 games to 64 costs 29% more wall clock for 3.2x the episodes, because a batch of 20
is pure launch overhead anyway. That matters because eval noise is what decides whether two runs
can be told apart. At 20 episodes it was 38% of the observed variance between checkpoints, at 64
it is about 12%.

The thing to watch is the interval, not the cost. A good 40x30 game runs ~9,500 sequential moves,
so one eval takes about 14 s however many games it runs. At 22,600 steps/s an `eval_every` of
25,000 fires roughly every second of training, which spends more wall clock evaluating than
training. The shipped default is 100,000.

---

## Reproducing

```bash
python -m pytest -q                       # 66 tests, CPU only
python -m snakeai.train --preset small    # full learning curve in a couple of minutes
```
