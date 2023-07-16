#!/bin/sh

# pip
python -m pip install --upgrade pip

# wheel
pip install wheel

# tox
pip install tox

# update apt package lists
sudo apt-get update

# optional dependencies
for par in "$@"
do
  case $par in
    base)
      # basic setup
      if [ "$(uname)" == "Darwin" ]; then
        # MacOS
        brew install redis
      else
        # Linux
        sudo apt-get install redis-server
      fi
    ;;

    R)
      # R environment
      if [ "$(uname)" == "Darwin" ]; then
        # MacOS
        brew install r
      else
        # Linux
        # install two helper packages we need
        sudo apt install --no-install-recommends software-properties-common dirmngr
        # add the signing key (by Michael Rutter) for these repos
        # To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
        # Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
        wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
        # add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
        sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
	sudo apt install --no-install-recommends r-base
      fi
    ;;

    amici)
      # AMICI dependencies
      sudo apt-get install swig libatlas-base-dev libhdf5-serial-dev
    ;;

    doc)
      # documentation
      sudo apt-get install pandoc
    ;;

    *)
      echo "Unknown argument" >&2
	  exit 1
    ;;

  esac
done
