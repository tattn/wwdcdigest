#!/usr/bin/env bash
set -o pipefail

# test.sh - Comprehensive lint and test script for wwdctools
# Usage: ./scripts/test.sh [options]
#   Options:
#     --lint-only: Run only linting checks, skip tests
#     --test-only: Run only tests, skip linting
#     --fix: Fix linting issues where possible
#     --help: Show this help message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Define color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_LINT=true
RUN_TESTS=true
FIX_ISSUES=false
HAS_FAILURES=false

# Process command line arguments
for arg in "$@"; do
  case $arg in
    --lint-only)
      RUN_TESTS=false
      ;;
    --test-only)
      RUN_LINT=false
      ;;
    --fix)
      FIX_ISSUES=true
      ;;
    --help)
      echo "Usage: ./scripts/test.sh [options]"
      echo "  Options:"
      echo "    --lint-only: Run only linting checks, skip tests"
      echo "    --test-only: Run only tests, skip linting"
      echo "    --fix: Fix linting issues where possible"
      echo "    --help: Show this help message"
      exit 0
      ;;
  esac
done

# Function to print section headers
print_header() {
  printf "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to check command result and print appropriate message
check_result() {
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    printf "${GREEN}✓ $1 passed${NC}\n"
  else
    printf "${RED}✗ $1 failed${NC}\n"
    HAS_FAILURES=true
    if [ "$2" != "continue" ]; then
      exit 1
    fi
  fi
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    printf "${YELLOW}uv is not installed. Installing...${NC}\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
    check_result "uv installation"
fi

# Sync dependencies with uv
print_header "Syncing dependencies with uv"
uv sync --extra dev
check_result "uv sync --extra dev" "continue"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Run linting checks
if [ "$RUN_LINT" = true ]; then
  print_header "Running Linting Checks"

  # Ruff Format
  printf "${YELLOW}Running Ruff Format...${NC}\n"
  if [ "$FIX_ISSUES" = true ]; then
    uv run --frozen ruff format .
    check_result "Ruff Format" "continue"
  else
    uv run --frozen ruff format --check .
    check_result "Ruff Format Check" "continue"
  fi

  # Ruff Lint
  printf "${YELLOW}Running Ruff Lint...${NC}\n"
  if [ "$FIX_ISSUES" = true ]; then
    uv run --frozen ruff check . --fix
    check_result "Ruff Lint" "continue"
  else
    uv run --frozen ruff check .
    check_result "Ruff Lint Check" "continue"
  fi

  # Pyright Type Checking
  printf "${YELLOW}Running Pyright Type Checking...${NC}\n"
  uv run --frozen pyright
  check_result "Pyright Type Checking" "continue"
fi

# Run tests
if [ "$RUN_TESTS" = true ]; then
  print_header "Running Tests"
  
  # Run pytest with anyio
  printf "${YELLOW}Running pytest...${NC}\n"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD="" uv run --frozen pytest
  check_result "Pytest" "continue"
fi

# Final summary
print_header "Summary"
if [ "$HAS_FAILURES" = true ]; then
  printf "${RED}Some checks failed. Please fix the issues and try again.${NC}\n"
  exit 1
else
  printf "${GREEN}All checks passed!${NC}\n"
fi
