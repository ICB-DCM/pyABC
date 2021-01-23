#!/bin/bash


# Run selected notebooks, on error show output and return with error code.


declare -a notebooks
notebooks=(
  "adaptive_distances" "conversion_reaction" "early_stopping"
  "external_simulators" "model_selection" "noise"
  "parameter_inference" "resuming" "using_R")


dir="doc/examples"


run_notebook () {
    tempfile=$(tempfile)
    echo $@
    jupyter nbconvert --ExecutePreprocessor.timeout=-1 --debug \
      --stdout --execute --to markdown $@ &> $tempfile
    ret=$?
    if [[ $ret != 0 ]]; then
        cat $tempfile
        exit $ret
    fi
    rm $tempfile
}


for notebook in "${notebooks[@]}"; do
    run_notebook $dir/$notebook
done
