## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

### Runtime setup

Provision the runtime environment with `scripts/setup_runtime_env.sh`, start services with `scripts/run_local.sh`; full playbook and known pitfalls in `docs/SETUP_RUNTIME.md`.
