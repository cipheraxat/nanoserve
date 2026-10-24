# nanoserve

A small vLLM-style LLM inference engine in PyTorch. Project memory lives in [`memory-bank/`](memory-bank/).

## At the start of every session

Read these, in this order, before changing anything:

1. `memory-bank/activeContext.md`: what is being worked on now, next steps, blockers.
2. `memory-bank/projectbrief.md`: what the project is for and what counts as done.
3. `memory-bank/systemPatterns.md`: architecture, invariants and coding standards. Follow them.
4. `memory-bank/progress.md`: what is built and what is left.
5. `memory-bank/runbook.md`: the exact commands to set up, test, lint, run, benchmark and push. Don't guess commands.

Check `memory-bank/decisions.md` before reversing a design choice, and `memory-bank/lessons.md` before working in an
area it lists.

## While working

- Major choice (a library, a data structure, an API shape, a scheduling or caching policy): add an ADR to
  `decisions.md` with the why, the trade-offs and the outcome.
- A bug or edge case that took more than one attempt: add it to `lessons.md` with the symptom, cause and fix.
- Keep `activeContext.md` current: focus, next steps, blockers. It should be enough to resume after a crash.

## After each milestone

- Update `progress.md` (done, in progress, to do) and `activeContext.md`.
- Run `uv run ruff check . && uv run pytest -q` before committing.
- If benchmark numbers change, update the README tables and `benchmarks/results/` together.
