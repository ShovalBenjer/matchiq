# Contributing to matchiq

This document describes the contributing guidelines for the `matchiq` repository
under the `ShovalBenjer` organization. All contributors are expected to follow
these guidelines when opening pull requests or making changes.

## Table of Contents

- [PR Process](#pr-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Conventions](#commit-conventions)
- [Branch Naming](#branch-naming)
- [CI](#ci)

## PR Process

1. **Create a feature branch** from `main` using the branch naming convention
   (see [Branch Naming](#branch-naming)).
2. **Make your changes** with clear, focused commits.
3. **Open a pull request** against the `main` branch.
4. **Claude Code** automatically reviews every PR via the
   [`claude-code-review.yml`](../.github/workflows/claude-code-review.yml)
   workflow. Address any feedback it raises.
5. **Request review** from the code owners defined in
   [CODEOWNERS](../CODEOWNERS) (`@claude-setup/owners`).
6. **Merge** once approved. Squash merges are the default.

> Draft PRs are skipped by the automated review workflow. Mark your PR as
> ready for review when you want it reviewed.

## Code Style

- **Directories**: use `kebab-case` (e.g., `src/utils`, `docs/api`).
- **Files**: use `kebab-case` (e.g., `claude-code-review.yml`).
- **Variables/Functions**: follow the language's idiomatic convention
  (`snake_case` for Python/Rust, `camelCase` for JavaScript/TypeScript).
- Run linters and formatters before committing. Use pre-commit hooks if
  available.
- Keep changes focused — one logical change per PR.

## Testing

- Write tests for new functionality before or alongside the implementation.
- Ensure all existing tests pass before opening a PR.
- If the project has a test runner configured, run it locally:

```bash
# Python
pytest

# Rust
cargo test

# JavaScript/TypeScript
npm test
```

- Add or update test coverage as needed for the scope of your change.

## Commit Conventions

This project uses [conventional commits](https://www.conventionalcommits.org/).

| Prefix      | Purpose                                      |
|-------------|----------------------------------------------|
| `feat:`     | A new feature                                |
| `fix:`      | A bug fix                                    |
| `chore:`    | Maintenance tasks (deps, tooling, config)    |
| `docs:`     | Documentation changes                        |
| `ci:`       | CI/CD pipeline changes                       |
| `refactor:` | Code restructuring without behavior change   |
| `test:`     | Adding or modifying tests                    |

### Examples

```
feat: add user authentication endpoint
fix: resolve null pointer in config loader
chore: update GitHub Actions checkout to v4
docs: add contributing guidelines
ci: add lint step to workflow
refactor: extract shared utility module
test: add unit tests for auth service
```

- Each PR should contain a single logical commit when possible, or a small set
  of focused commits.
- Commit messages should be written in the imperative mood
  ("Add feature" not "Added feature").

## Branch Naming

Branches follow the Gastown orchestration naming convention:

```
gt/<agent>/<bead-id>
```

Examples:

- `gt/toast/b8782256`
- `gt/pike/89ec22f8`

The `main` branch is the default and must always be production-ready. Do not
push directly to `main`.

## CI

- All PRs trigger the automated Claude Code review workflow.
- The workflow uses `actions/checkout@v4` and `anthropics/claude-code-action@v1`.
- Review results are posted as PR comments.
- Ensure CI passes before merging.

## Questions?

Open an issue or reach out to the code owners (`@claude-setup/owners`) for
guidance.