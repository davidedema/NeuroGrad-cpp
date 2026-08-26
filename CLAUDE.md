# Instructions for Claude on this project

This file is the standing instruction set for every interaction on
NeuroGrad-cpp. Read it before starting work; it complements (does not
replace) `README.md`, `CONTRIBUTING.md`, and `docs/notes/`.

## How to work

- Go step by step. On each request, touch only what was asked for — do not
  rewrite or "improve" the rest of the codebase as a side effect. Stay
  focused on the specific class/function/stage in question.
- Prefer several small, reviewable changes over one large one, mirroring
  the stage-by-stage structure of `docs/notes/implementation-roadmap.md`.

## Code style

- Clean code = self-explanatory names (types, functions, variables) first,
  comments second. Add a comment where it earns its place: on a
  non-obvious step, a subtle math/ordering constraint (e.g. computing a
  derivative before overwriting the value it depends on), or a deliberate
  convention choice (e.g. the subgradient of `relu` at 0). Don't restate
  what the code already says.
- Single-responsibility functions: one function does one operation. If a
  function is doing more than one thing, split it into named
  sub-functions rather than growing it in place.
- Structure new code as templates, following the conventions in
  `docs/notes/template-style-reference.hh`:
  - class/function templated on the scalar type (`template <typename T>`),
    not hardcoded to `double`;
  - Doxygen comment (`@brief`, plus `@tparam`/`@param`/`@return` as
    applicable) on every public member;
  - a `detail` namespace for internal helpers that aren't public API;
  - for a value type: compound-assignment operators (`+=`, `-=`, `*=`,
    `/=`) implemented first, with binary operators built on top of them
    when that's the clearer path;
  - elementary math functions as free `template <typename T>` functions
    that pull in the right overload via `using std::foo;` (ADL), so the
    same template works for `float`, `double`, or a nested autodiff type.
  - This applies to `autodiff::forward::Dual` too: it was retrofitted from
    its original non-templated scaffolding to `template <typename T> class
    Dual` following this style (see `docs/notes/implementation-roadmap.md`
    Stage 1/3 for the history). Future templated classes (Layer,
    NeuralNetwork, a future reverse-mode `Var`, ...) should follow the same
    conventions from the start.

## Tests

- Whenever a class or major piece of functionality is implemented, write
  its tests in the same change — don't leave verification for later.
- Follow the existing convention (see `README.md` "Testing convention"):
  one test file per class, named `test_<ClassName>.cc`, tagged with the
  class name and `[finite-diff]` where relevant; any derivative check
  against a numerical reference goes through `tests/utils/finite_diff.hh`
  rather than reimplementing finite differences locally.

## Docs to keep in sync

- `docs/notes/implementation-roadmap.md`: update the status of the
  relevant stage as work lands, and edit the plan itself if a structural
  decision changes (e.g. a different namespace layout, a stage split in
  two, a new dependency). Keep it reflecting where the project actually
  is, not where it was planned to be.
- `README.md`: update before making a commit whenever the change affects
  what's described there (project layout, build/test instructions,
  testing convention, CI).
- `CONTRIBUTING.md`: describes the branch/PR/commit-message workflow for
  this project — follow it (feature branches, scoped PRs, imperative
  commit messages) rather than improvising a different one.
- `docs/notes/log.md`: not this file's responsibility to write
  automatically, but flag to the user when a stage lands so they can add
  their learning-log entry per `CONTRIBUTING.md`.
