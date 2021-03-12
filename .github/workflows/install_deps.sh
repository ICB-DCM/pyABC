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
        sudo apt-key adv \
          --keyserver keyserver.ubuntu.com \
          --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
        sudo add-apt-repository \
          'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
        sudo apt-get update
        sudo apt-get install r-base
      fi
    ;;

    amici)
      # AMICI dependencies
      sudo apt-get install swig3.0 libatlas-base-dev libhdf5-serial-dev
      if [ ! -e /usr/bin/swig ]; then
        sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
      fi
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
