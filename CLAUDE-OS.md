# CLAUDE-OS

> **Operating System** — the single source of truth for `matchiq`'s workflow, conventions, and tooling configuration.

## Project Overview

`matchiq` is a repository under the `ShovalBenjer` GitHub organization. It follows the
claude-setup SOTA directory structure standard.

## Branch Naming Convention

- **Feature branches**: `gt/<agent>/<bead-id>` (e.g., `gt/toast/b8782256`)
- **Main branch**: `main` — the default branch, always production-ready.
- Branches are managed by the Gastown orchestration system and should not be
  manually switched unless resolving a PR conflict or fixup.

## Commit Message Convention

Commits use the conventional commits format:

| Prefix      | Purpose                                      |
|-------------|----------------------------------------------|
| `feat:`     | A new feature                                |
| `fix:`      | A bug fix                                    |
| `chore:`    | Maintenance tasks (deps, tooling, config)    |
| `docs:`     | Documentation changes                        |
| `ci:`       | CI/CD pipeline changes                       |
| `refactor:` | Code restructuring without behavior change   |
| `test:`     | Adding or modifying tests                    |

**Squash merges** are used by default. Each PR should contain a single logical commit
when possible, or a small set of focused commits.

## CI / CD

### GitHub Actions

- **`.github/workflows/claude-code-review.yml`** — Automatically triggers a fresh
  Claude Code review on every PR (opened, synchronize, ready_for_review, reopened).
  Uses `actions/checkout@v4` and `anthropics/claude-code-action@v1`.

### Secrets

| Secret               | Purpose                                    |
|----------------------|--------------------------------------------|
| `ANTHROPIC_API_KEY`  | API key for Claude Code review action      |

> The `CLAUDE_CODE_OAUTH_TOKEN` secret is **not** used — the repo is not on the
> Claude Code trusted allowlist. Use `ANTHROPIC_API_KEY` with `anthropic_api_key` input.

## Code Ownership

All code is owned by `@claude-setup/owners` by default. See `CODEOWNERS` for details.

## Linting & Formatting

- Run linters before committing.
- Use pre-commit hooks if available.

## Contributing

See `README.md` for contributing guidelines. All contributions are reviewed by
Claude Code via the automated PR review workflow.

## Naming Conventions

- **Directories**: `kebab-case` (e.g., `src/utils`, `docs/api`)
- **Files**: `kebab-case` (e.g., `claude-code-review.yml`)
- **Variables/Functions**: Follow the language's idiomatic convention
  (`snake_case` for Python/Rust, `camelCase` for JavaScript/TypeScript)
