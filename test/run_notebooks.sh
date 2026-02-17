#!/bin/bash

# Run selected notebooks, on error show output and return with error code.

# set environment
export PYABC_MAX_POP_SIZE=20

# Notebooks to run
nbs_1=(
  "adaptive_distances"
  "wasserstein"
  "conversion_reaction"
  "early_stopping"
  "model_selection"
  "noise"
  "parameter_inference"
  "resuming"
  "chemical_reaction"
  "informative"
  "look_ahead"
  "data_plots"
  "discrete_parameters"
  "custom_priors"
  "aggregated_distances"
  #"sde_ion_channels"  # url error
)
nbs_2=(
  "external_simulators"
  "using_R"
  "optimal_threshold"
  "petab_application"
  # "petab_yaml2sbml"  # yaml2sbml does not work with current petab version
  # "multiscale_agent_based"  # too slow
  "using_copasi"
  "using_julia"
)

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
  tempfile=$(mktemp)
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
