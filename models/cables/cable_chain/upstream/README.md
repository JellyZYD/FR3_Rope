# Cable Insertion Reinforcement Learning Baselines

Reinforcement learning simulation baselines for autonomous robotic cable insertion in MuJoCo.

This repository implements procedural cable insertion physics environments, baseline policy training with Proximal Policy Optimization (PPO), evaluation harnesses, and robotic manipulator integration. It provides the simulation substrate referenced in Poisson Labs' research on visual representation robustness and closed-loop robotic policy evaluation, including [*Where V-JEPA 2.1's Dense Features Hold Up (and Where They Don't)*](https://poissonlabs.ai/research/vjepa-2-1-robustness/).

---

## What Is Modeled and Measured

1. **Procedural Cable Insertion Environment (`CableInsertionEnv`)**
   - **Physics Substrate:** High-frequency MuJoCo simulation (`1000 Hz` physics timestep, `10 Hz` agent control frequency) modeling a 4-segment passive cable chain with ball joints, torsional damping, and a cylindrical terminal connector.
   - **Kinematics:** Floating continuous 3-DOF action control commanding cartesian displacements with coordinate transforms relative to the gripper origin.
   - **Reward Formulation:** Decomposed dense distance shaping ($-10 \times \text{distance}$) coupled with graduated one-time milestone bonuses: $+10.0$ at $5\text{ cm}$, $+25.0$ at $3\text{ cm}$, and $+100.0$ at $2\text{ cm}$ (successful terminal insertion).
   - **Observation Modes:** Low-dimensional continuous state vector (gripper position, connector position, target position, vector offsets) and optional visual rendering modes.

2. **Competition-Scale UR5e Robotic Arm Environment (`CableInsertionUR5eEnv`)**
   - **Manipulator Kinematics:** Composed 7-DOF Universal Robots UR5e arm with a Robotiq 2F-85 parallel-jaw gripper, assembled programmatically via MuJoCo's `MjSpec` API.
   - **Sensory Surface:** Multi-camera observation pipelines including wrist-mounted cameras (looking down along the cable) and fixed overhead/side cameras.
   - **Action Space:** 7-dimensional continuous joint velocity control bounded by joint velocity limits (`MAX_JOINT_VEL`).

---

## Baseline Performance

- **Pretrained Checkpoint:** The repository includes a trained PPO policy checkpoint ([`cable_ppo.zip`](cable_ppo.zip), $150\text{ KB}$, $155,494\text{ bytes}$) and associated normalization state ([`cable_ppo_vecnormalize.pkl`](cable_ppo_vecnormalize.pkl), $1.7\text{ KB}$).
- **Validation Convergence:** Evaluated across a 50-episode validation harness (fixed seed 42), the policy achieves **$100\%$ task success rate** (50/50 successful insertions within the $2\text{ cm}$ target threshold).
- **Episode Efficiency:** Average episode length of $5.32\text{ steps}$ ($\sigma = 2.68$), with mean terminal distance of $0.016\text{ m}$ to the target receptacle socket.
- **Learning Dynamics:** Empirical training reveals a steep convergence transition between $2\text{M}$ and $3\text{M}$ environment steps when trained from scratch using MLP policies without curriculum learning.

---

## How to Reproduce

### 1. Environment Setup

Clone the repository and create a Python 3.11+ virtual environment:

```bash
git clone https://github.com/poisson-labs/cable-insertion.git
cd cable-insertion

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Run Test Suite

Run the full verification unit test suite:

```bash
pytest
```

To run code formatting and lint verification:

```bash
ruff check .
ruff format --check .
```

### 3. Evaluate Pretrained Policy

Run the evaluation loop using the tracked PPO checkpoint:

```bash
python -m eval.eval
```

### 4. Train From Scratch

To launch PPO policy training on the procedural cable environment:

```bash
python -m training.train
```

For remote training on GPU instances with Weights & Biases telemetry, refer to the bringup script:

```bash
bash scripts/runpod_setup.sh
```

---

## Repository Index & Artifacts

- **Standard & Hygiene:** [`docs/repo-hygiene.md`](docs/repo-hygiene.md), [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), [`pyproject.toml`](pyproject.toml)
- **Environments:**
  - Procedural floating-gripper cable simulation: [`envs/cable_env.py`](envs/cable_env.py)
  - MuJoCo XML model: [`envs/assets/cable_scene.xml`](envs/assets/cable_scene.xml)
  - Composed UR5e + Robotiq 2F-85 arm environment: [`envs/cable_env_ur5e.py`](envs/cable_env_ur5e.py)
- **Training & Evaluation:**
  - Training entry points: [`training/train.py`](training/train.py), [`training/train_ur5e_state.py`](training/train_ur5e_state.py), [`training/train_ur5e_vision.py`](training/train_ur5e_vision.py)
  - Evaluation runner: [`eval/eval.py`](eval/eval.py)
  - Synchronization tooling: [`scripts/sync_checkpoints.sh`](scripts/sync_checkpoints.sh), [`scripts/runpod_setup.sh`](scripts/runpod_setup.sh)
- **Tracked Model Weights:**
  - Pretrained PPO model: [`cable_ppo.zip`](cable_ppo.zip) ($150\text{ KB}$)
  - VecNormalize statistics: [`cable_ppo_vecnormalize.pkl`](cable_ppo_vecnormalize.pkl) ($1.7\text{ KB}$)

---

## Citation

```bibtex
@misc{kolasinski2026cableinsertion,
  author = {Kolasinski, Taylor},
  title  = {Cable Insertion Reinforcement Learning Baselines},
  year   = {2026},
  month  = {September},
  howpublished = {\url{https://poissonlabs.ai/research/vjepa-2-1-robustness/}},
  note   = {Poisson Labs research repository; source code and checkpoints at \url{https://github.com/poisson-labs/cable-insertion}}
}
```

## License

MIT. See [`LICENSE`](LICENSE). Third-party dependencies and upstream license notices are documented in [`NOTICE`](NOTICE).
