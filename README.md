# matchiq

A repository for the matchiq project under the `ShovalBenjer` organization,
following the claude-setup SOTA directory structure standard.

## Overview

`matchiq` is a project managed by the Gastown orchestration system. It uses
automated CI/CD with Claude Code for pull request reviews and follows
standardized repository conventions.

## Repository Structure

```
matchiq/
├── .github/
│   └── workflows/
│       └── claude-code-review.yml   # Automated PR review via Claude Code
├── .gitignore                        # Comprehensive ignore rules
├── CLAUDE-OS.md                      # Operating system / workflow config
├── CODEOWNERS                        # Code ownership rules
├── LICENSE                           # MIT License
├── README.md                         # This file
└── TODO.md                           # Task tracking
```

## Usage

Clone the repository:

```bash
git clone https://github.com/ShovalBenjer/matchiq.git
cd matchiq
```

## Contributing

1. Create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Open a pull request — Claude Code will automatically review it.
4. Address any feedback and merge once approved.

See `CLAUDE-OS.md` for the full workflow and conventions documentation.

## License

MIT — see [LICENSE](LICENSE).
