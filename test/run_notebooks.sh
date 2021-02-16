#!/bin/bash

# Run selected notebooks, on error show output and return with error code.


# Notebooks to run
nbs_1=(
  "adaptive_distances" "conversion_reaction" "early_stopping"
  "model_selection" "noise"
  "parameter_inference" "resuming")
nbs_2=(
  "external_simulators" "using_R" "petab_yaml2sbml")

# All notebooks
nbs_all=("${nbs_1[@]}" "${nbs_2[@]}")

# Select which notebooks to run
if [ $# -eq 0 ]; then
  nbs=("${nbs_all[@]}")
elif [ $1 -eq 1 ]; then
  nbs=("${nbs_1[@]}")
elif [ $1 -eq 2 ]; then
  nbs=("${nbs_2[@]}")
else
  echo "Unexpected input: $1"
fi

# Notebooks repository
dir="doc/examples"

# Find uncovered notebooks
for nb in `ls $dir | grep -E "ipynb"`; do
  missing=true
  for nb_cand in "${nbs_all[@]}"; do
    if [[ $nb == "${nb_cand}.ipynb" ]]; then
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
echo "Run notebooks:"
for notebook in "${nbs[@]}"; do
    time run_notebook $dir/$notebook
done
