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
        wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo gpg --dearmor -o /usr/share/keyrings/r-project.gpg
        echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | sudo tee -a /etc/apt/sources.list.d/r-project.list
        sudo apt-get update
        sudo apt-get install libtiff5 r-base
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
