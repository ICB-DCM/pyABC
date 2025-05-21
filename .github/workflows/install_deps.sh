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
         # Linux (Not compatible with Ubuntu 24)
        #wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo gpg --dearmor -o /usr/share/keyrings/r-project.gpg
        #echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | sudo tee -a /etc/apt/sources.list.d/r-project.list
        #sudo apt-get update
        #sudo apt-get install libtiff5 r-base
        sudo apt update
        sudo apt install -y build-essential libreadline-dev libx11-dev libxt-dev libpng-dev libjpeg-dev libcairo2-dev libssl-dev libcurl4-openssl-dev libxml2-dev texinfo texlive texlive-fonts-extra screen wget
        sudo apt install -y liblzma-dev

sudo apt install -y \
build-essential \
libreadline-dev \
libx11-dev \
libxt-dev \
libpng-dev \
libjpeg-dev \
libcairo2-dev \
libtiff-dev \
libglib2.0-dev \
liblzma-dev \
libbz2-dev \
libzstd-dev \
libcurl4-openssl-dev \
libssl-dev \
libxml2-dev \
texinfo \
texlive \
texlive-fonts-extra \
texlive-latex-extra \
zlib1g-dev \
gfortran \
libpcre2-dev \
libicu-dev

     cd /tmp
        wget https://cran.r-project.org/src/base/R-4/R-4.4.0.tar.gz
        tar -xvzf R-4.4.0.tar.gz
        cd R-4.4.0
        ./configure --enable-R-shlib --with-blas 
        make -j$(nproc)
        sudo make install
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
