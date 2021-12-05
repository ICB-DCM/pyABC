"""Interface pyABC with COPASI via BasiCO."""

import logging
import os
from typing import Dict, List

from .. import Parameter
from ..model import Model

logger = logging.getLogger("ABC.Copasi")

try:
    import basico
except ImportError:
    basico = None
    logger.error(
        "Install BasiCO (see https://basico.rtfd.io) to use the BasiCO model, "
        "e.g. via `pip install pyabc[copasi]`"
    )


class BasicoModel(Model):
    """COPASI time series simulations via BasiCO.

    This class is a :class:`pyabc.Model` compliant wrapper around
    `basico.run_time_course`, allowing to update model parameters and use
    various simulation methods.
    BasiCO (https://basico.rtfd.io) is a simple Python interface to
    COPASI (http://copasi.org).

    The implementation is derived from an implementation by Frank Bergmann at
    https://github.com/fbergmann/pyabc-copasi.
    """

    def __init__(
        self,
        sbml_file: str,
        changes: Dict[str, float] = None,
        change_unit: bool = True,
        method: str = "stochastic",
        t0: float = None,
        duration: float = None,
        num_steps: int = None,
        automatic: bool = True,
        use_numbers: bool = False,
        output: List[str] = None,
        model_name: str = None,
    ):
        """
        Parameters
        ----------
        sbml_file:
            SBML file containing the model definition.
        changes:
            Parametric changes to apply to the SBML model.
        change_unit:
            Whether to change units to 1, useful for discrete simulations
            (particle numbers).
        method:
            Simulation method, can be any method supported by
            `basico.run_time_course`, in particular:

            * deterministic, lsoda: the LSODA implementation
            * stochastic: the Gibson & Bruck Gillespie implementation
            * directMethod: Gillespie Direct Method
            * others: hybridode45, hybridlsoda, adaptivesa, tauleap, radau5,
              sde
        t0, duration, num_steps:
            Initial time point, duration, number of steps. Definition and
            combination as in `basico.run_time_course`.
        automatic:
            Whether to use automatic steps, or the specified interval / number
            of steps.
        use_numbers:
            Whether to return all elements collected.
        output:
            Species to output. Defaults to all.
        model_name:
            Model name, for identification e.g. in the database.
        """
        # a maybe informative model name
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(sbml_file))[0]
        self.model_name = model_name

        super().__init__(
            name=f"BasicoModel_{model_name}",
        )

        self.sbml_file = sbml_file
        self.dm = basico.load_model(sbml_file)

        # many sbml models do not define realistic units, and
        # since we compute in particle numbers, we usually do not run
        # for  particles, so set substance unit to 1
        if change_unit:
            basico.set_model_unit(substance_unit='1', model=self.dm)
        self.change_unit = change_unit

        # allow to override parameters
        if changes is not None:
            self.apply_parameters(changes)
        self.changes = changes

        self.t0 = t0
        self.duration = duration
        self.num_steps = num_steps
        self.automatic = automatic
        self.use_numbers = use_numbers
        self.output = output
        self.method = method

    def __call__(self, pars: Dict[str, float], return_raw: bool = False):
        """Simulate data for given parameters.

        Calls the time course and returns the selected result.
        """
        # apply parameters to model
        self.apply_parameters(pars)

        # parse time args by basico's logic
        if self.t0 is not None:
            args = self.t0, self.duration, self.num_steps
        elif self.num_steps is not None:
            args = self.duration, self.num_steps
        else:
            args = (self.duration,)

        # simulate
        tc = basico.run_time_course(
            *args,
            model=self.dm,
            method=self.method,
            automatic=self.automatic,
            use_seed=False,
            use_numbers=self.use_numbers,
        ).reset_index()

        if return_raw:
            return tc

        # cache output columns
        if self.output is None:
            self.output = list(set(tc.columns) - {"Time"})

        return {
            "t": tc.Time.to_numpy(),
            "X": tc[self.output].to_numpy(),
        }

    def sample(self, pars: Parameter):
        """Sample for parameters.

        This is the method called by pyABC. It calls `__call__` and reduces
        the output.
        """
        return self(pars, return_raw=False)

    def apply_parameters(self, pars: Dict[str, float]):
        """Set the parameters of the model.

        Parameters
        ----------
        pars:
            Parameters to apply, id-value dictionary.
            Local parameters are assumed to be named something like
            `(reaction).local_parameter`, where `reaction` is the name of the
            reaction, and `local_parameter` the local parameter.
            Specifically, local parameters are identified by the presence of
            brackets.
            Otherwise the parameter is expected to be a global one.
        """
        for key, val in pars.items():
            if '(' in key:
                basico.set_reaction_parameters(key, value=val, model=self.dm)
            else:
                basico.set_parameters(key, initial_value=val, model=self.dm)

    def __getstate__(self):
        # all arguments
        state = {
            "sbml_file": self.sbml_file,
            "changes": self.changes,
            "change_unit": self.change_unit,
            "method": self.method,
            "t0": self.t0,
            "duration": self.duration,
            "num_steps": self.num_steps,
            "automatic": self.automatic,
            "use_numbers": self.use_numbers,
            "output": self.output,
            "model_name": self.model_name,
        }
        return state

    def __setstate__(self, state):
        self.__init__(**state)
