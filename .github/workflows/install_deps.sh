#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -U wheel tox

is_macos() {
  [[ "$(uname -s)" == "Darwin" ]]
}

apt_update_once() {
  if [[ "${_APT_UPDATED:-0}" == "0" ]]; then
    export _APT_UPDATED=1
    sudo apt-get update -y
  fi
}

apt_install() {
  apt_update_once
  sudo apt-get install -y --no-install-recommends "$@"
}

install_base() {
  if is_macos; then
    brew install redis
  else
    apt_install redis-server
  fi
}

build_and_install_r_from_source() {
  # Adjust if you want a different R version
  local R_VER="${1:-4.4.0}"
  local TAR="R-${R_VER}.tar.gz"
  local URL="https://cran.r-project.org/src/base/R-4/${TAR}"

  # Toolchain + common R build deps (Ubuntu 24.04-friendly)
  apt_install \
    build-essential gfortran wget ca-certificates \
    libreadline-dev libx11-dev libxt-dev \
    libpng-dev libjpeg-dev libcairo2-dev libtiff-dev \
    libglib2.0-dev liblzma-dev libbz2-dev libzstd-dev zlib1g-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libpcre2-dev libicu-dev \
    texinfo texlive texlive-fonts-extra texlive-latex-extra

  pushd /tmp >/dev/null
  wget -q "${URL}"
  tar -xzf "${TAR}"
  cd "R-${R_VER}"
  ./configure --enable-R-shlib --with-blas --with-lapack
  make -j"$(nproc)"
  sudo make install
  popd >/dev/null
}

install_r() {
  if is_macos; then
    brew install r
  else
    # Ubuntu-latest may not have the desired CRAN apt repo available;
    # compiling R is the reliable option.
    build_and_install_r_from_source "4.4.0"
  fi
}

install_amici() {
  if ! is_macos; then
    apt_install swig libatlas-base-dev libhdf5-serial-dev libboost-all-dev
  fi

  # Ensure non-interactive uninstalls in CI
  python -m pip uninstall -y amici pyabc || true
  python -m pip install -U "pyabc[amici]"
}


for arg in "$@"; do
  case "$arg" in
    base)  install_base ;;
    R)     install_r ;;
    amici) install_amici ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done
