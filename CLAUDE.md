# CLAUDE.md — claude_riskmanager

> This file provides context for AI assistants (including Claude) working in this repository.
> Last updated: 2026-02-17

## Project Overview

**claude_riskmanager** is a collection of multiple mini projects, each containing its own Python code to perform financial analysis and risk management tasks. Each mini project is self-contained with its own scripts, data, and dependencies.

## Repository Structure

This is a **multi-project monorepo**. Each top-level directory (other than shared config files) represents an independent mini project focused on a specific financial analysis topic.

```
claude_riskmanager/
├── CLAUDE.md              # AI assistant guide (this file)
├── <mini-project-A>/      # Self-contained financial analysis project
│   ├── *.py               # Python scripts for this analysis
│   ├── data/              # Input data (if any)
│   └── README.md          # Project-specific documentation (if any)
├── <mini-project-B>/      # Another self-contained project
│   └── ...
└── ...
```

> **When adding a new mini project**, create a new top-level directory with a descriptive name and keep all related code, data, and documentation within it.

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd claude_riskmanager

# Each mini project may have its own dependencies.
# Navigate into the specific project directory and install as needed:
cd <mini-project>
pip install -r requirements.txt   # if a requirements.txt exists

# Run a mini project's script directly:
python <script>.py
```

### Python Environment

- Language: **Python 3**
- Each mini project may specify its own dependencies via `requirements.txt` or inline comments
- Common financial analysis libraries to expect: `numpy`, `pandas`, `scipy`, `matplotlib`, `yfinance`, `statsmodels`, `scikit-learn`

## Development Workflow

### Adding a New Mini Project

1. Create a new top-level directory with a clear, descriptive name (e.g., `var_calculation/`, `portfolio_optimization/`)
2. Add Python scripts within that directory
3. Include a `requirements.txt` if the project has dependencies beyond the Python standard library
4. Optionally add a `README.md` explaining the project's purpose and usage
5. Update this CLAUDE.md if the project introduces new conventions

### Branching Strategy

- Feature branches follow the pattern: `claude/<description>-<session-id>`
- Develop on feature branches and merge via pull requests

### Commit Conventions

- Write clear, descriptive commit messages
- Focus on the "why" rather than the "what"
- When a commit touches a specific mini project, name it in the commit message

### Code Style

- Follow PEP 8 conventions for Python code
- Use descriptive variable and function names relevant to the financial domain
- Include docstrings for non-trivial functions explaining the financial logic

## Key Conventions for AI Assistants

When working in this repository, follow these guidelines:

1. **Respect project boundaries** — Each mini project is independent; avoid cross-project imports or shared code unless explicitly requested
2. **Read before writing** — Always read existing files before modifying them
3. **Keep changes minimal** — Only make changes directly related to the task at hand
4. **Follow existing patterns** — Match the style and conventions already present in the codebase
5. **Run scripts to verify** — After making changes, run the relevant Python scripts to confirm they work
6. **Update this file** — When adding a new mini project or significant new structure, update CLAUDE.md to reflect the changes
7. **No over-engineering** — Each mini project should be simple and focused on its specific financial analysis task
8. **Security first** — Never commit secrets, API keys, credentials, or sensitive financial data; use environment variables or config files excluded via `.gitignore`
9. **Scope awareness** — When asked to work on a specific mini project, stay within that directory; don't modify unrelated projects
