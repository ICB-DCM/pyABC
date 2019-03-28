#!/bin/bash

declare -a notebooks
notebooks=("adaptive_distances" "chemical_reaction" "conversion_reaction" "early_stopping" "multiscale_agent_based" "parameter_inference" "quickstart" "resuming" "using_R")


dir="doc/examples"


run_notebook () {
    tempfile=$(tempfile)
    jupyter nbconvert --ExecutePreprocessor.timeout=-1 --debug --stdout --execute --to markdown $@ &> $tempfile
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
