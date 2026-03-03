---
name: agent-worktree
description: Create git worktrees with metadata tracking for PR workflows. Use when the user wants to create an isolated worktree for feature work that will be merged back via PR.
metadata:
  short-description: Create tracked git worktrees
---

# agent-worktree

Create a git worktree based on a source branch, with a target branch name, and record metadata for future PR creation.

## Parameters

- `target` (required) — new branch name, also used as worktree subdirectory (with `/` replaced by `-`)
- `source` (optional) — base branch, defaults to current branch
- `description` (optional) — short description of the work

## Workflow

### Step 1: Resolve parameters

- Get current git branch as default source
- Get project directory name: `dir_name=$(basename "$PWD")`
- Compute worktree parent: `wt_root="../${dir_name}_worktree"`
- Compute subdirectory: replace `/` with `-` in target → `wt_subdir`
- Full worktree path: `${wt_root}/${wt_subdir}`

### Step 2: Create worktree

```bash
mkdir -p "${wt_root}"
git worktree add -b "${target}" "${wt_root}/${wt_subdir}" "${source}"
```

### Step 3: Append metadata to `cc_worktree.toml`

Append a new `[[worktrees]]` entry to `${wt_root}/cc_worktree.toml`:

```toml
[[worktrees]]
source = "<source>"
target = "<target>"
path = "<wt_subdir>"
description = "<description>"
created_at = "<YYYY-MM-DD>"
```

### Step 4: Confirm

Print the worktree path and remind the user that future PRs should merge `target` → `source`.
