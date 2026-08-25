# Working together on this

## Branches and PRs

- `main` is protected: no direct pushes. Every change goes through a
  feature branch and a Pull Request.
- Branch naming: `feature/<short-name>`, one per roadmap stage or
  sub-piece of one, e.g. `feature/dual-arithmetic`,
  `feature/eigen-integration`, `feature/layer-forward-pass`.
- Before merging: CI (`.github/workflows/ci.yml`) must be green on all
  three OSes, and the other person reviews the PR — even a short review.
  The review is as much a learning tool as a correctness gate: if you
  can't explain why a derivative formula is right to your teammate, that's
  worth catching before it's merged, not after.
- Keep PRs scoped to one stage/class where possible (e.g. "implement
  Dual arithmetic operators" as one PR, "implement elementary functions"
  as a second) — easier to review, easier to bisect if something breaks
  later.

## Splitting work

The roadmap in `docs/notes/implementation-roadmap.md` is staged
specifically so it can be split: e.g. one person owns `Dual.hh` +
finite-difference tests while the other starts sketching the `Layer` /
`NeuralNetwork` interface against the (stubbed) `autodiff::Scalar` type in
parallel, then swap for review once Stage 1 lands. Track stages as GitHub
Issues (one per stage, or split further as needed) and use a Project board
(To do / In progress / Review / Done) to see who's on what.

## Commit messages

Short, imperative, and specific enough that `git log --oneline` is useful
on its own:

```
implement Dual compound-assignment operators
add finite-difference tests for exp/log/sqrt
fix operator/= overwriting value_ before computing derivative_
```

## Learning log

Add one entry per stage to `docs/notes/log.md` (template in that file):
what you implemented, what you got wrong on the first try (if anything),
and what the finite-difference check caught, if it caught anything. This
doubles as raw material for the final report/relazione — write it while
it's fresh rather than trying to reconstruct it later.
