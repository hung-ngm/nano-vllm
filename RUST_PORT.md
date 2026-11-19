# Rust port setup

Setup only, so a later pass can rewrite this Python library in Rust with [rust-anything](https://github.com/erkamyaman/rust-anything). No Rust crate and no changes to the Python sources are in this branch.

## Install the CLI

The published crate is `rust-anything` 1.5.1 and needs Rust 1.97. With that toolchain:

```sh
cargo install rust-anything
rust-anything --version
```

This environment had Rust 1.83, so the same 1.5.1 release was installed from the Linux x86_64 binary on the [v1.5.1 GitHub release](https://github.com/erkamyaman/rust-anything/releases/tag/v1.5.1) (checksum checked against `SHA256SUMS`) into `/usr/local/cargo/bin/rust-anything`.

`record`, `diff`, and `app record` run the original code on many generated inputs, and `app record` runs each case twice. Code that writes files, talks to a database, or calls the network does that once per case. Run those commands on a copy of the project, with a test database and no production credentials in the environment.

## What was added

`rust-anything setup` on this machine found Cursor only and wrote `~/.cursor/mcp.json`. The server command there is the absolute path of the installed binary, with args `["mcp"]`. Cursor's setup does not install the workflow skill. Claude Code and Codex get the skill in their own skill directories when those agents are present.

Committed so the next agent has the same pieces inside the repo:

- `.mcp.json` and `.cursor/mcp.json`: MCP server `rust-anything`, command `rust-anything`, args `["mcp"]`.
- The workflow skill, copied unchanged from `erkamyaman/rust-anything` commit `3875a0332dd0008c359c98e6175f52fc1133914b` (`skills/rust-anything/SKILL.md` and `skills/rust-anything/whole-app.md`) into both `.cursor/skills/rust-anything/` and `.claude/skills/rust-anything/`. That commit is `main` at the time of the copy. The v1.5.1 tag is `9795e7aa3bcbdde69df423d8772a857b45b99478`; its skill files differ, and these copies follow `main`.
- Upstream license: MIT, Copyright (c) 2026 Erkam Yaman (erkamyaman), from that repository's `LICENSE`. The two skill files were copied byte for byte. They contain no license header.

No `rust-anything.json` at the repo root. The docs create that file inside the generated crate when `bridge` runs, and under `rust/app-cases/` when `app record` runs. They do not require a project config before `record`.

## First commands

On a copy of this repo, with no production credentials:

```sh
rust-anything analyze . --whole-app -o plan.md
```

Stop there until the plan and the first unit are accepted. Then, for that unit (this repo is a Python library; `record` needs the crate `bridge` just wrote):

```sh
rust-anything bridge . <unit-id>
rust-anything record rust/<crate>
```

Follow `.cursor/skills/rust-anything/whole-app.md` for the rest: decisions, the front door, one unit at a time through `build`, `diff`, and `gate`, then `app wire` and `app lift`. `example.py` loads model weights and runs generation; do not point `record` or `app record` at it until that copy has a test model path and no production credentials.
