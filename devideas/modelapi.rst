* it might not be possible to evaluate the model at all prior sample points
  * model evaluation exception?
* maybe require that epsilon=infty does not lead to early stopping
* if model operates in early stopping mode, than data is already there
* have model always return summary statistics, include early_stopped field in summary statistics
  * have distance function account for that
* early acceptance?
* separate model and evaluation? have a model class and an evaluator class?