#!/bin/bash

# Run selected notebooks, on error show output and return with error code.


# Notebooks to run
declare -a notebooks
notebooks=(
  "adaptive_distances" "conversion_reaction" "early_stopping"
  "external_simulators" "model_selection" "noise"
  "parameter_inference" "resuming" "using_R")

# Notebooks repository
dir="doc/examples"

# Find uncovered notebooks
for nb in `ls $dir | grep -E "ipynb"`; do
  missing=true
  for nb_cand in "${notebooks[@]}"; do
    if [[ $nb == $nb_cand ]]; then
      missing=false
      continue
    fi
  done
  if $missing; then
    echo "Notebook $nb is not covered in tests."
  fi
done

run_notebook () {
  # Run a notebook and raise upon failure
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

# Run all notebooks in list

for notebook in "${notebooks[@]}"; do
    run_notebook $dir/$notebook
done
