#!/usr/bin/env bash
# CI dependency installation script for pyABC
# Supports Ubuntu 24.04 and macOS
set -euo pipefail

# Color output helpers
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

log_info() {
  echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*"
}

# Update pip, wheel, and tox
log_info "Updating pip, wheel, and tox..."
python -m pip install --upgrade pip wheel tox

# Platform detection
is_macos() {
  [[ "$(uname -s)" == "Darwin" ]]
}

# APT management (Ubuntu/Debian)
_APT_UPDATED=0

apt_update_once() {
  if [[ "${_APT_UPDATED}" == "0" ]]; then
    export _APT_UPDATED=1
    log_info "Updating apt package lists..."
    sudo apt-get update -y
  fi
}

apt_install() {
  apt_update_once
  log_info "Installing apt packages: $*"
  sudo apt-get install -y --no-install-recommends "$@"
}

# Base dependencies (Redis)
install_base() {
  log_info "Installing base dependencies..."
  if is_macos; then
    brew install redis
  else
    apt_install redis-server
    # Ensure redis-server is running
    sudo service redis-server start || true
  fi
}

# R installation from source
build_and_install_r_from_source() {
  local R_VER="${1:-4.4.2}"
  local TAR="R-${R_VER}.tar.gz"
  local URL="https://cran.r-project.org/src/base/R-4/${TAR}"

  log_info "Building R ${R_VER} from source..."

  # Build dependencies for Ubuntu 24.04
  apt_install \
    build-essential gfortran wget ca-certificates \
    libreadline-dev libx11-dev libxt-dev \
    libpng-dev libjpeg-dev libcairo2-dev libtiff-dev \
    libglib2.0-dev liblzma-dev libbz2-dev libzstd-dev zlib1g-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libpcre2-dev libicu-dev \
    texinfo texlive texlive-fonts-extra texlive-latex-extra

  pushd /tmp >/dev/null

  log_info "Downloading R source..."
  wget -q "${URL}" -O "${TAR}"

  log_info "Extracting R source..."
  tar -xzf "${TAR}"
  cd "R-${R_VER}"

  log_info "Configuring R build..."
  ./configure --enable-R-shlib --with-blas --with-lapack --prefix=/usr/local

  log_info "Compiling R (this may take several minutes)..."
  make -j"$(nproc)"

  log_info "Installing R..."
  sudo make install

  # Verify installation
  if command -v R >/dev/null 2>&1; then
    log_info "R successfully installed: $(R --version | head -n1)"
  else
    log_error "R installation verification failed"
    popd >/dev/null
    return 1
  fi

  popd >/dev/null
}

# R installation
install_r() {
  log_info "Installing R..."
  if is_macos; then
    brew install r
  else
    # Ubuntu: compile from source for better compatibility
    build_and_install_r_from_source "4.4.2"
  fi

  # Set LD_LIBRARY_PATH for R shared libraries
  if ! is_macos; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib}:/usr/local/lib/R/lib"
    echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"" >> ~/.bashrc
  fi
}

# AMICI dependencies
install_amici() {
  log_info "Installing AMICI dependencies..."

  if ! is_macos; then
    apt_install \
      swig \
      libatlas-base-dev \
      libhdf5-serial-dev \
      libboost-all-dev
  fi

  log_info "Installing AMICI Python package..."
  # Clean install to avoid version conflicts
  python -m pip uninstall -y amici pyabc || true
  python -m pip install --upgrade "pyabc[amici]"
}

# Documentation tools
install_doc_tools() {
  log_info "Installing documentation tools..."

  if is_macos; then
    brew install pandoc || true
  else
    apt_update_once
    apt_install pandoc
  fi
}

# Julia installation
install_julia() {
  log_info "Installing Julia..."

  if is_macos; then
    brew install julia
  else
    # Install Julia via juliaup (recommended approach)
    curl -fsSL https://install.julialang.org | sh -s -- -y
    export PATH="$HOME/.juliaup/bin:$PATH"
  fi

  # Initialize PyJulia
  python -c "import julia; julia.install()" || log_warn "PyJulia initialization failed (non-critical)"
}

# Development tools
install_dev_tools() {
  log_info "Installing development tools..."

  python -m pip install --upgrade \
    pre-commit \
    ruff \
    build \
    twine \
    pytest \
    pytest-cov \
    pytest-xdist
}

# All dependencies
install_all() {
  log_info "Installing all dependencies..."
  install_base
  install_r
  install_amici
  install_doc_tools
  install_julia
  install_dev_tools
}

# Display usage
usage() {
  cat <<EOF
Usage: $0 [OPTION]...

Install CI dependencies for pyABC.

Options:
  base        Install base dependencies (Redis)
  R           Install R from source
  amici       Install AMICI dependencies
  doc         Install documentation tools (Pandoc)
  julia       Install Julia and PyJulia
  dev         Install development tools
  all         Install all dependencies
  help        Display this help message

Examples:
  $0 base R amici
  $0 all

EOF
}

# Main execution
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
      julia) install_julia ;;
      dev)   install_dev_tools ;;
      all)   install_all ;;
      help|--help|-h)
        usage
        exit 0
        ;;
      *)
        log_error "Unknown argument: ${arg}"
        usage
        exit 1
        ;;
    esac
  done

  log_info "Installation completed successfully!"
}

main "$@"
