# CLAUDE.md — claude_riskmanager

> This file provides context for AI assistants (including Claude) working in this repository.
> Last updated: 2026-02-17

## Project Overview

**claude_riskmanager** is a risk management application. The project is currently in its initial setup phase with no source code committed yet.

This document should be updated as the project evolves to reflect the actual codebase structure, conventions, and workflows.

## Repository Status

This repository is newly initialized. As the project takes shape, update the sections below with accurate details.

## Directory Structure

```
claude_riskmanager/
├── CLAUDE.md          # AI assistant guide (this file)
└── .git/              # Git repository metadata
```

<!-- As the project grows, update this tree to reflect the actual layout. Example:
├── src/               # Application source code
├── tests/             # Test suite
├── docs/              # Documentation
├── config/            # Configuration files
├── scripts/           # Build/deploy scripts
└── ...
-->

## Getting Started

<!-- Fill in once the project has a build system, dependencies, and setup instructions. -->

```bash
# Clone the repository
git clone <repository-url>
cd claude_riskmanager

# Install dependencies
# TODO: Add dependency installation commands

# Run the application
# TODO: Add run commands

# Run tests
# TODO: Add test commands
```

## Development Workflow

### Branching Strategy

- Feature branches follow the pattern: `claude/<description>-<session-id>`
- Develop on feature branches and merge via pull requests

### Commit Conventions

- Write clear, descriptive commit messages
- Focus on the "why" rather than the "what"

### Code Style & Linting

<!-- Update with actual linting/formatting tools and commands once configured. -->

- TODO: Document code style standards
- TODO: Document linting commands

### Testing

<!-- Update with actual test framework and commands once configured. -->

- TODO: Document test framework
- TODO: Document how to run tests
- TODO: Document test coverage requirements

## Architecture

<!-- Update this section as the application architecture is defined. -->

### Key Components

- TODO: List major modules/components and their responsibilities

### Data Layer

- TODO: Document database or data storage approach

### API / Interface

- TODO: Document API structure or user interface

## Configuration

<!-- Document environment variables, config files, and secrets management. -->

- TODO: List required environment variables
- TODO: Document configuration files

## CI/CD

<!-- Document continuous integration and deployment pipelines once set up. -->

- TODO: Document CI/CD pipeline
- TODO: Document deployment process

## Key Conventions for AI Assistants

When working in this repository, follow these guidelines:

1. **Read before writing** — Always read existing files before modifying them
2. **Keep changes minimal** — Only make changes directly related to the task at hand
3. **Follow existing patterns** — Match the style and conventions already present in the codebase
4. **Run tests** — Verify changes don't break existing functionality before committing
5. **Update this file** — When adding significant new structure, dependencies, or workflows, update CLAUDE.md to reflect the changes
6. **No over-engineering** — Prefer simple, direct solutions over abstractions for hypothetical future needs
7. **Security first** — Never commit secrets, credentials, or sensitive data; validate inputs at system boundaries
