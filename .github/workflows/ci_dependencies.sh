#!/bin/sh

# apt
sudo apt-get update
sudo apt-get install redis-server

# pip
python -m pip install --upgrade pip
pip install -r .ci_pip_reqs.txt

# install package
pip install -e .

# optional dependencies
for par in "$@"
do
    case $par in

        R)
            pip install rpy2>=3.2.0 cffi>=1.13.1
        ;;

        petab)
            sudo apt-get install swig3.0 libatlas-base-dev libhdf5-serial-dev
            sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
            pip install https://github.com/icb-dcm/petab/archive/develop.zip
            pip install -e \
                'git+https://github.com/icb-dcm/amici.git@develop#egg=amici&subdirectory=python/sdist'
            git clone --depth 1 \
                https://github.com/petab-dev/petab_test_suite .tmp/petab_test_suite
            pip install -e .tmp/petab_test_suite
        ;;

        *)
            echo "Unknown argument" >&2
	    exit 1
        ;;
    esac
done
