---
name: sglang-jax-skypilot-dev
description: Use when running tests, benchmarks, or validation for the sgl-jax project on a remote TPU cluster via SkyPilot. Covers cluster provisioning, test execution, and result retrieval for sglang-jax development workflows.
argument-hint: "[test-path-or-command] [args...]"
---

# sglang-jax SkyPilot Dev Skill

This skill handles running tests and development tasks for the sgl-jax project on remote TPU clusters via SkyPilot.

> **Note:** This project requires TPU for all JAX tests. Never run JAX/TPU tests locally.

## 1. Prerequisites

- The TPU cluster must already be provisioned. Check that `.cluster_name_tpu` exists and is non-empty in the project root.
- If the file does not exist or is empty, provision a cluster first (see Section 3).

**Execution Instructions:**
Before running the launch script, find its absolute path in the `scripts/` directory alongside this skill definition. Use file search tools (e.g., `glob`) to locate `launch_tpu.sh` before executing it.

## 2. Project Layout

| Component | Path |
|-----------|------|
| Main source | `python/sgl_jax/` |
| SRT module | `python/sgl_jax/srt/` |
| Tests | `test/` |
| Test suite runner | `test/srt/run_suite.py` |

## 3. Cluster Management

### Provisioning

```bash
# Common TPU types for this project: tpu-v6e-4, tpu-v6e-8, tpu-v4-8
bash <absolute_path_to_launch_tpu.sh> <accelerator_type> <experiment_name>

# Example:
bash <absolute_path_to_launch_tpu.sh> tpu-v6e-4 dp-test
```

The launch script automatically writes the cluster name to `.cluster_name_tpu`.

### Teardown

```bash
sky down $(cat .cluster_name_tpu) -y
```

## 4. Running Tests

All test execution uses **SSH + tmux** for reliable process management and log inspection. Never use `sky exec` for running tests.

### General Workflow

1. SSH into the cluster
2. Update code on the remote via git
3. Start test tasks in named tmux sessions
4. Monitor and retrieve results

---

### Step 1: SSH into the cluster

```bash
# Get the cluster IP
CLUSTER_IP=$(sky status --ip $(cat .cluster_name_tpu))

# SSH directly to the remote
ssh -i ~/.ssh/sky-key gcpuser@$CLUSTER_IP
```

---

### Step 2: Update code on the remote via git

On the **remote machine**, use git to pull the latest changes:

```bash
cd ~/sky_workdir

# Pull latest changes from the remote repository
git fetch origin
git pull origin <branch-name>

# Or if you need to switch branches
git checkout <branch-name>
git pull origin <branch-name>

# Install/update dependencies after pulling
uv sync --extra tpu
```

**Important:** Always commit and push your local changes before running tests on the remote. The remote should pull from the git repository, not sync from your local filesystem.

---

### Step 3: Run tests in tmux sessions

#### Mode A: Test File Testing

Run test files directly under the `test/` directory. Use this for unit tests, integration tests, and regression tests that don't require a live server.

**Run the full test suite:**
```bash
# On the remote machine:
tmux new-session -d -s test-suite -c ~/sky_workdir
tmux send-keys -t test-suite \
  "uv run --extra tpu python test/srt/run_suite.py" Enter

# Follow the output
tmux attach -t test-suite
```

**Run a single test file:**
```bash
# On the remote machine:
tmux new-session -d -s test -c ~/sky_workdir
tmux send-keys -t test \
  "uv run --extra tpu python -m pytest test/srt/<test_file.py> -v" Enter

# Follow the output
tmux attach -t test
```

**Run tests matching a pattern:**
```bash
# On the remote machine:
tmux new-session -d -s test -c ~/sky_workdir
tmux send-keys -t test \
  "uv run --extra tpu python -m pytest test/srt/ -k <pattern> -v" Enter

# Follow the output
tmux attach -t test
```

**Common pytest flags:**

| Flag | Purpose |
|------|---------|
| `-v` | Verbose output |
| `-k <pattern>` | Run tests matching pattern |
| `-x` | Stop on first failure |
| `--tb=short` | Short traceback |
| `-s` | Show stdout (useful for JAX logs) |

---

#### Mode B: Service Debug Testing

Use this when you need to start a server process, wait for it to be ready, then run a second process (debug script, benchmark, or accuracy test) against the live server.

#### Step 1: Start the server in a named tmux session

```bash
# On the remote machine:
tmux new-session -d -s server -c ~/sky_workdir
tmux send-keys -t server \
  "uv run --extra tpu python <SERVER_COMMAND> [ARGS]" Enter
```

#### Step 2: Wait for the server to be ready

```bash
# On the remote machine — poll until the service is ready:
until <READINESS_CHECK>; do
  echo "Waiting for server..."; sleep 5
done
echo "Server is ready."
```

Common readiness checks:

| Method | Command |
|--------|---------|
| HTTP health endpoint | `curl -sf http://localhost:<PORT>/health` |
| Log line appeared | `tmux capture-pane -pt server \| grep -q "server started"` |
| Port is open | `nc -z localhost <PORT>` |

#### Step 3: Run the client in a separate tmux session

```bash
# On the remote machine:
tmux new-session -d -s client -c ~/sky_workdir
tmux send-keys -t client \
  "uv run --extra tpu python <CLIENT_COMMAND> [ARGS]" Enter

# Follow client output
tmux attach -t client
```

#### Step 4: Inspect and clean up

```bash
# List all sessions
tmux ls

# View server logs (without attaching)
tmux capture-pane -pt server -S -1000

# Kill a session when done
tmux kill-session -t server
tmux kill-session -t client
```

**Tmux session naming convention:**

| Session name | Purpose |
|---|---|
| `server` | The main inference/serving process |
| `client` | Debug / benchmark / accuracy test runner |
| `<feature>-server` | Use feature-prefixed names when running multiple experiments simultaneously |

<!-- Specific server commands, ports, and readiness checks for each feature area go in Section 6 -->

## 5. Operational Notes

- **SSH Access**: Use direct SSH with `ssh -i ~/.ssh/sky-key gcpuser@$CLUSTER_IP` to connect to the cluster
- **Code Updates**: Always use git pull on the remote to update code; commit and push local changes first
- **Tmux Sessions**: All test execution happens in named tmux sessions for reliability and log inspection
- **Workdir**: The remote workdir is at `~/sky_workdir/` on the TPU VM
- **Logs**: View logs using `tmux capture-pane` or by attaching to the session
- **Interruption**: Detach from tmux with `Ctrl+B D`; sessions persist after disconnection
- **agent.md**: Check the project's `agent.md` for the active cluster name when working on parallel development tasks

## 6. Functional-Point-Specific Testing

<!-- This section will be expanded with specific test commands per feature area -->

*(To be populated with data-parallelism feature-specific test commands and validation steps.)*
