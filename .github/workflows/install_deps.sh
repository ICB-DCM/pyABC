#!/usr/bin/env bash
set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

is_macos() { [[ "$(uname -s)" == "Darwin" ]]; }

_APT_UPDATED=0
apt_update_once() {
  if [[ "${_APT_UPDATED}" == "0" ]]; then
    _APT_UPDATED=1
    log_info "Updating apt package lists..."
    sudo apt-get update -y
  fi
}
apt_install() {
  apt_update_once
  log_info "Installing apt packages: $*"
  sudo apt-get install -y --no-install-recommends "$@"
}

export_env_var() {
  local key="$1"
  local value="$2"
  export "${key}=${value}"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "${key}=${value}" >> "$GITHUB_ENV"
  fi
}

log_info "Updating pip, wheel, tox..."
python -m pip install --upgrade pip wheel tox

install_base() {
  log_info "Installing base dependencies..."
  if is_macos; then
    brew install redis
  else
    apt_install redis-server
    sudo service redis-server start || true
  fi
}

install_r() {
  log_info "Installing R..."
  if is_macos; then
    brew install r
  else
    # Prefer distro packages in CI
    apt-get update && apt-get install -y libtirpc-dev r-base r-base-dev
    # Make R shared libs discoverable for the rest of the job
    export_env_var LD_LIBRARY_PATH "${LD_LIBRARY_PATH:-/usr/lib}:/usr/lib/R/lib:/usr/local/lib/R/lib"
  fi

  if command -v R >/dev/null 2>&1; then
    log_info "R installed: $(R --version | head -n1)"
  else
    log_error "R installation failed (R not on PATH)"
    exit 1
  fi
}

install_amici() {
  log_info "Installing AMICI dependencies..."
  if ! is_macos; then
    apt_install swig libatlas-base-dev libhdf5-serial-dev libboost-all-dev
  fi
  log_info "Installing AMICI Python package..."
  python -m pip uninstall -y amici pyabc || true
  python -m pip install --upgrade "pyabc[amici]"
}

install_doc_tools() {
  log_info "Installing documentation tools..."
  if is_macos; then
    brew install pandoc || true
  else
    apt_install pandoc
  fi
}

install_dev_tools() {
  log_info "Installing development tools..."
  python -m pip install --upgrade pre-commit ruff build twine pytest pytest-cov pytest-xdist
}

install_all() {
  install_base
  install_r
  install_amici
  install_doc_tools
  install_dev_tools
}

usage() {
  cat <<EOF
Usage: $0 [OPTION]...

Options:
  base        Install base dependencies (Redis)
  R           Install R (Ubuntu: apt; macOS: brew)
  amici       Install AMICI dependencies
  doc         Install documentation tools (Pandoc)
  dev         Install development tools
  all         Install all dependencies
  help        Display this help message
EOF
}

main() {
  if [[ $# -eq 0 ]]; then
    log_error "No arguments provided"
    usage
    exit 1
  fi

  for arg in "$@"; do
    case "$arg" in
      base)  install_base ;;
      R)     install_r ;;
      amici) install_amici ;;
      doc)   install_doc_tools ;;
      dev)   install_dev_tools ;;
      all)   install_all ;;
      help|--help|-h) usage; exit 0 ;;
      *) log_error "Unknown argument: ${arg}"; usage; exit 1 ;;
    esac
  done

  log_info "Installation completed"
}

main "$@"
