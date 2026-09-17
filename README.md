# SnakeAI

A Snake-playing agent trained with Double DQN in PyTorch, with a pygame viewer.

Training is headless and vectorised: it runs hundreds of games in parallel with batched GPU
inference and a replay buffer resident in VRAM. Watching the agent play is a separate process that
hot-reloads the checkpoint, so it costs the trainer nothing.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch   # or .../whl/cpu
pip install -r requirements.txt

python -m snakeai.train --preset small          # a full learning curve in a couple of minutes
python -m snakeai.play --checkpoint runs/dev/best.pt
```

## Commands

| command | what it does |
| --- | --- |
| `python -m snakeai.train` | headless training; writes to `runs/<name>/` |
| `python -m snakeai.play --checkpoint PATH` | watch a trained agent |
| `python -m snakeai.play --checkpoint runs/dev/ckpt.pt --follow` | watch a **live** run; reloads as it improves |
| `python -m snakeai.play --human` | play it yourself (arrows / WASD) |
| `python -m snakeai.evaluate --checkpoint PATH` | greedy evaluation on fixed seeds |
| `python -m snakeai.plot runs/dev` | plot a run from its CSV |
| `python -m pytest` | 50 tests, CPU only |

Presets: `--preset small` (8×8, minutes), `default` (20×15), `big` (40×30, the original board).

Any config key can be overridden and a typo is an error rather than a silent no-op:

```bash
python -m snakeai.train --set agent.lr=3e-4 --set agent.hidden=256,256 --run-dir runs/wider
python -m snakeai.train --print-config          # resolved config, then exit
```

## Training runs

Activate the venv first: `source .venv/bin/activate`

```bash
# quick sanity check, a full learning curve in a couple of minutes on the 8x8 board
python -m snakeai.train --preset small --run-dir runs/small

# the default 20x15 board
python -m snakeai.train --run-dir runs/dev --set train.total_steps=3000000

# the ORIGINAL 40x30 board, to compare against v1's best of 94
python -m snakeai.train --preset big --run-dir runs/big \
    --set train.total_steps=6000000 \
    --set train.save_every=500000

# watch it play while it trains, in a second terminal. Costs the trainer nothing.
python -m snakeai.play --checkpoint runs/big/ckpt.pt --follow

# when it is done
python -m snakeai.evaluate --checkpoint runs/big/best.pt --episodes 100
python -m snakeai.plot runs/big --save docs/big_run.png

# carry on a run that was stopped
python -m snakeai.train --preset big --run-dir runs/big --resume runs/big/ckpt.pt \
    --set train.total_steps=12000000
```

On a 5080 the 40x30 board runs at roughly 2,700 environment steps per second, so 6M steps is
about 40 minutes. Throughput falls as the snake grows, because the flood fills that build the
observation have more board to cross, so treat that as a lower bound rather than a promise.

Useful knobs: `--set agent.updates_per_iter=8` and `--set agent.batch_size=2048` control how
much it learns per round of moves, `--set train.num_envs=64` runs fewer games in parallel, plus
`--set agent.lr=3e-4` and `--set agent.hidden=256,256`.

## Layout

```text
snakeai/
  config.py    every tunable, as frozen dataclasses with validated cross-field invariants
  env.py       the MDP: dynamics, 11-dim observation, reward. No torch, no pygame.
  vecenv.py    N independent envs stepped together, with a safe auto-reset contract
  agent.py     Q-network, device-resident replay, the Double DQN update
  train.py     headless training loop. Imports no GUI library; a test enforces that.
  evaluate.py  greedy evaluation on fixed seeds
  play.py      the only module that imports pygame
  plot.py      offline plotting from metrics.csv
  device.py    device selection and a guard for missing GPU kernels
```

## Observation and reward

17 float32 values, all in `[-1, 1]`, all **relative to the snake's heading**, so the same
situation rotated four ways produces an identical vector and the network learns one case instead
of four:

| slots | what they tell the snake |
| --- | --- |
| `danger_straight/left/right` | is this move instantly fatal (looks one cell ahead) |
| `food_ahead/behind/left/right` | which side the food is on |
| `food_forward`, `food_lateral` | how far away it is, signed |
| `length_frac`, `hunger_frac` | how big I am, how close to starving |
| `free_straight/left/right` | how much room each move leads into (1.0 = fits my whole body) |
| `tail_straight/left/right` | could I still reach my own tail after that move |

The last two groups are what stop the snake trapping itself. `free_*` floods outward from each
move's landing cell and counts reachable empty cells; `tail_*` asks whether that region still
connects to the tail. Two moves can both lead into plenty of room while only one stays connected.
`free_*` alone cannot tell those apart.

Reward is bounded and **independent of the snake's length**:

| event | reward |
| --- | ---: |
| every step | −0.01 |
| eat food | +1.00 |
| fill the board (win) | +2.00 |
| hit a wall or itself | −1.00 |
| starve (hunger clock) | 0.00 and the episode is **truncated, not terminated** |

Plus potential-based shaping, `γ·Φ(s′) − Φ(s)` with `Φ = −0.1 · normalised_distance`. That form is
policy-invariant (Ng, Harada & Russell 1999): it speeds learning up without changing which policy
is optimal.

## GPU notes

`--device auto` uses CUDA for training and CPU for playing. Both are correct and the reason is
measured rather than assumed. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md):

- This network is **kernel-launch bound**. A CUDA forward costs ~the same at batch 1 as at batch
  4096, so batch-1 inference wastes the GPU; CPU is 4.6x faster there and 11x faster once the
  `.item()` sync is counted. `num_envs` (default 256) keeps every forward past the crossover.
- On GPU a batch-16384 update costs the same as a batch-256 one, so the loop spends its gradient
  budget on **few large updates** rather than many small ones. That one change was worth 10.7x.
- Blackwell (`sm_120`) needs a cu128+ wheel. An older wheel still reports `cuda.is_available()`,
  then either fails at the first matmul or silently JITs and runs slow. `device.py` checks the
  arch list at startup and says so.

## v1

The original agent (flat scripts, vanilla DQN with BatchNorm) reached a best score of 94 on the
40×30 board. Its checkpoint and score curve are kept in
`model/Best_DQN_Model_64_512_94_776k/` for reference. It does not load into v2. The observation
vector, the network and the checkpoint format all changed, so use `--preset big` to train a
comparable agent.

That version had a number of defects that v2 is structured to make unrepresentable rather than
merely fixed.
