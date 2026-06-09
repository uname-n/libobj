# CLAUDE.md — libobj

`libobj` is an embedded document database, built as a Rust workspace of four crates. This file governs two things every session: (a) **how you work here** — you are the *orchestrator* who decomposes work into pit-tracked issues, delegates each to a subagent running in an isolated git worktree, reviews the result, and integrates it into `main`; and (b) the **safety rules** every change must satisfy before merge — clippy, `cargo deny`, and tests must all be clean.

## Project

| Crate | Role | Notes |
|-------|------|-------|
| `obj-rs` | Public crate (`obj`) — typed `Db` / `Collection<T>` API | The frozen public surface; `#![forbid(unsafe_code)]` |
| `obj-core` | Storage engine (pager, WAL, B+tree, codec, catalog, txn) | **Unstable** — implementation detail of `obj-rs`, no SemVer guarantee |
| `obj-derive` | Proc-macro for `#[derive(obj::Document)]` | Unstable; consume via `obj-rs`; `#![forbid(unsafe_code)]` |
| `libobj` | C ABI / FFI boundary | The `unsafe` boundary; every entry point is `unsafe extern "C"` |

The ten safety rules ("Power of Ten for Rust") are defined and enforced in **Safety rules** below; user-facing docs live in `README.md`.

## Orchestration

You **plan, track, delegate, review, and integrate**. You do **not** write implementation code directly — subagents do, each inside its own worktree.

### pit setup

pit is an MCP-backed issue tracker. Issues live in `.pit/db.sqlite`. Both `.pit/` and `.claude/worktrees/` are gitignored.

`.mcp.json` (project root):

```json
{
  "mcpServers": {
    "pit": {
      "command": "pit"
    }
  }
}
```

Install pit if not present:

```bash
curl -fsSL https://raw.githubusercontent.com/uname-n/pit/master/install.sh | sh
```

### Directory layout

```
.
├── .claude/worktrees/      # one worktree per active issue (gitignored)
│   ├── issue-3/
│   └── issue-7/
├── .pit/db.sqlite          # pit issue database (gitignored)
├── .mcp.json
├── CLAUDE.md
└── crates/                 # obj-rs, obj-core, obj-derive, libobj
```

### The loop

1. **Plan → create issues.** Decompose the task into small, well-scoped units. For each, `create_issue` with an action-oriented title, a body (context + acceptance criteria + constraints/file pointers), labels, and status `open`. Split any issue with more than ~3 acceptance criteria.

2. **Mark it started.** `update_issue(id=<ID>, status="in-progress")`. The worktree itself is created by `claude --worktree` in the next step — no manual `git worktree add`.

3. **Delegate to a subagent** in a fresh worktree (parallelize independent issues only when they share no files). `claude --worktree <name>` creates `.claude/worktrees/<name>/` on branch `worktree-<name>`; run it headless with `-p`:

   ```bash
   claude --worktree issue-<ID> -p \
     "You are working on issue #<ID>. Read it with get_issue(<ID>), then implement \
      the acceptance criteria. Work only within this worktree. Commit with \
      'closes #<ID>: <short description>'. Do not merge or switch branches."
   ```

   This makes `.claude/worktrees/issue-<ID>/` on branch `worktree-issue-<ID>` (confirm with `git worktree list`).

   **NEVER pass `--dangerously-skip-permissions` (or any flag that disables the permission gate) to a subagent.** Headless subagents inherit this project's `.claude/settings.json` `permissions.allow` list, which already grants everything the workflow needs — `Edit`/`Write`, all `cargo` commands (`test`, `clippy`, `build`, `deny`, `audit`, `fmt`, `bench`, `llvm-cov`), the git operations (`add`, `commit`, `diff`, `log`, `status`, `worktree`, `branch`), and the `pit` MCP tools. A `-p` run does not re-prompt for allowlisted actions, so it will not hang. If a subagent genuinely needs an action that isn't allowlisted, **add a scoped rule to `.claude/settings.json`** — do not disable the gate. Never reach for a skip-permissions workaround.

4. **Review** when the subagent reports done:
   - Inspect the diff: `git diff main..worktree-issue-<ID>` (or `git log main..worktree-issue-<ID> --stat`).
   - Run safety checks in the worktree: `cd .claude/worktrees/issue-<ID> && <Commands below>`.
   - Re-read acceptance criteria via `get_issue(<ID>)`.
   - If revision needed: `add_comment(id=<ID>, body="Needs revision: <feedback>")`, then re-delegate.

5. **Integrate** once it passes:

   ```bash
   git checkout main
   git merge --no-ff worktree-issue-<ID> -m "merge: closes #<ID> — <title>"
   git worktree remove .claude/worktrees/issue-<ID>   # -p runs don't auto-clean
   git branch -d worktree-issue-<ID>
   ```

   Then `update_issue(id=<ID>, status="closed", close_reason="implemented and merged")`.

### Subagent instruction template

Always include this when spawning a subagent:

```
You are a subagent working on issue #<ID> in the worktree at .claude/worktrees/issue-<ID>.

1. Call get_issue(<ID>) to read the full issue and acceptance criteria.
2. Implement the required changes. Work ONLY within this worktree directory.
3. Do not modify files outside your worktree.
4. Do not create or switch branches.
5. When complete, stage and commit all changes:
     git add -A && git commit -m "closes #<ID>: <short description>"
6. Report back: summarize what you did and flag anything uncertain or incomplete.
```

### pit tool reference

| Tool | When to use |
|------|-------------|
| `create_issue` | Planning phase — log every unit of work before starting |
| `list_issues` | Status overview; filter by `open`, `in-progress`, `closed` |
| `get_issue` | Read full issue + comments before delegating or reviewing |
| `update_issue` | Change status, update body, or set close reason |
| `add_comment` | Record review feedback, blocker notes, or decisions |
| `search_issues` | Find related issues before creating duplicates |
| `list_labels` | See what label conventions are in use |
| `delete_issue` | Remove issues no longer valid (use sparingly) |

Issue lifecycle: `open` → `in-progress` → `closed`.

### Orchestrator rules

- Always create the issue **before** delegating — its ID names the worktree (`issue-<ID>`) and branch (`worktree-issue-<ID>`).
- One issue per worktree; never bundle multiple issues into one branch.
- Never implement code in the main worktree — the main checkout is for orchestration only.
- Always review (diff + safety checks) before merging.
- Keep issues updated — status and comments are the source of truth.
- Prefer `--no-ff` merges to preserve the issue branch in history.
- Check for blockers: run `list_issues` at the start of each session.

### Suggested labels

| Label | Use for |
|-------|---------|
| `feature` | New functionality |
| `bug` | Something broken |
| `refactor` | Internal code-quality changes |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `blocked` | Waiting on another issue |
| `review` | Subagent done, awaiting orchestrator review |

## Safety rules (Power of Ten)

Every change must satisfy all ten rules. Most are enforced mechanically; the rest are review-only.

Mechanically enforced (a lint or config fails the build):

| Rule | Enforced by | Where it lives |
|------|-------------|----------------|
| R1 No unbounded recursion | `unconditional_recursion = "deny"` (rustc) — only catches no-base-case recursion; bounded recursion is review + `MAX_*_DEPTH` convention | `Cargo.toml [workspace.lints.rust]` |
| R4 Functions ≤ 60 lines | `clippy::too_many_lines = "deny"` (pinned out of `pedantic`) with threshold 60 | `Cargo.toml [workspace.lints.clippy]` + `clippy.toml` (`too-many-lines-threshold = 60`) |
| R6 Narrowest scope / no `static mut` | `static_mut_refs = "forbid"` (rustc, un-overridable) + `clippy::needless_late_init = "deny"` | `Cargo.toml [workspace.lints.rust]` and `[workspace.lints.clippy]` |
| R7 Handle every Result/Option; no unwrap/panic in prod | `#![cfg_attr(not(test), deny(clippy::unwrap_used, expect_used, panic, unwrap_in_result, get_unwrap, unreachable, todo, unimplemented))]`; discarded `#[must_use]` results are caught by rustc `unused_must_use` via R10's `warnings = deny` | each `crates/*/src/lib.rs` crate root |
| R8 Minimal, documented `unsafe` | `clippy::undocumented_unsafe_blocks` in the same `cfg_attr(not(test), deny(...))` — **obj-core & libobj only**; obj-rs & obj-derive are `#![forbid(unsafe_code)]` so they omit it. `clippy.toml` accepts the `// SAFETY:` comment above the enclosing statement/attribute. libobj also `#![deny(unsafe_op_in_unsafe_fn)]`. | each `crates/*/src/lib.rs`; `clippy.toml` (`accept-comment-above-statement`, `accept-comment-above-attributes`) |
| R10 Zero warnings, full static analysis | `warnings = { level = "deny", priority = -1 }` (rustc); `clippy::all` + `clippy::pedantic` = `"deny"` (promoted warn→deny); `rustflags = ["-D","warnings"]` belt-and-suspenders; supply-chain/advisory tooling | `Cargo.toml` lint tables; `.cargo/config.toml`; `deny.toml`, `.cargo/audit.toml`, `rust-toolchain.toml` |

Review-only (no stable lint captures the rule's real content — degenerate cases like `never_loop` are still caught by `clippy::all`):

| Rule | Why review-only |
|------|-----------------|
| R2 Bounded loops | Needs an explicit bounded counter / `.take(MAX)`; no lint proves it |
| R3 No alloc on hot paths | No stable lint for allocation on hot paths |
| R5 Type-system invariants + `debug_assert!` | Presence of `debug_assert!` / newtype invariants is not lintable |
| R9 Prefer concrete types over `dyn`/macros | No lint flags `dyn` on hot paths |

### When you edit

- **No `.unwrap()`/`.expect()`/`panic!`/`todo!`/`unreachable!`/`unimplemented!` in production** — use `?`, `Result`, `Option`, `unwrap_or_else`, or an explicit `match`. The deny is gated on `not(test)`, so **tests use them freely — do NOT add per-test `#[allow]`s.**
- **Every `unsafe { }` block needs a `// SAFETY:` comment.** It may sit on the line above the enclosing `let`/`match`/attribute, so don't hoist the expression out just to document it. cfg-gated `unsafe` (e.g. `platform/lock.rs` `errno` branches) needs one on **every** branch — host clippy only checks one target.
- **Every `#[allow(...)]` / `#![allow(...)]` needs a one-line `// allow: WHY` comment directly above it.**
- Satisfy R1/R2/R6 by construction: a `MAX_*_DEPTH` counter that returns `Err` past the bound (not unbounded recursion), an explicit bound on every loop, and `Mutex`/`RwLock`/atomics (never `static mut`). Keep functions ≤ 60 lines (comments/blanks don't count).

### Deliberately NOT enforced — do not "fix" these

Do not enable these lints or add guard tests; they were considered and rejected:

- `clippy::indexing_slicing` — storage-engine page-buffer indexing is intentional.
- `clippy::multiple_unsafe_ops_per_block` — would force unnatural FFI block splits.
- `clippy::allow_attributes_without_reason` — the rule wants a `reason =` field; we use a `// allow:` comment instead.
- grep / alloc-counting guard tests — too fragile.

## Commands

Subagents run these in their worktree before reporting; the orchestrator re-runs them at review. CI-equivalent:

```sh
cargo clippy --workspace --all-targets        # also: --all-features
cargo test --workspace
cargo deny check                              # licenses, advisories, sources, AND duplicate dep versions
cargo audit                                   # advisory database
```

`cargo deny` denies duplicate dependency versions (R10). If a new dep introduces one, either unify the version or add a documented entry to `skip` in `deny.toml` (mirroring the `// allow: WHY` convention).

## Coverage

Coverage is **not** part of the mandatory safety checks. Run it explicitly as a separate step.

Install once: `cargo install cargo-llvm-cov --locked`

```sh
cargo llvm-cov --workspace --all-features --summary-only --fail-under-lines 90
```

### Ratchet plan

| Gate | Action |
|------|--------|
| 85% (done) | Initial gate — passed. |
| 90% (current) | `--fail-under-lines` raised to `90` after coverage reached 91.40%. |
| 95% | Raise to `95`; use `#[cfg(not(coverage))]` exclusions **only** for proven-unreachable or generated code, with a comment explaining why. |

Never exclude code merely to hit the number without justification.
