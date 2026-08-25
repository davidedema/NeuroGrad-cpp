# docs/notes

Design rationale and a running learning log for this project, kept in git
so your teammate sees the same context you do (as opposed to living only
in one person's chat history).

- `implementation-roadmap.md` — the staged plan (mirrors what's tracked in
  GitHub Issues; read this before starting a new stage).
- `forward-vs-reverse-mode-ad.md` — why the project starts with
  forward-mode dual numbers and what changes if/when a reverse-mode engine
  is added later. Useful material for the final report.
- `log.md` — one entry per stage: what was implemented, what the
  finite-difference checks caught (if anything), what was non-obvious.
  Add to it as you go, not retroactively.
- `template-style-reference.hh` — style-only reference (not compiled, not
  part of the build) for how new templated classes should be structured;
  see `CLAUDE.md` for when it applies.
