from typing import Union
from ..random_variables import RV


class ModelPerturbationKernel:
    """Model perturbation kernel.

    Attributes
    ----------
    nr_of_models:
        Number of models
    probability_to_stay:
        If ``None``, probability to stay is set to 1/nr_of_models.
        Otherwise, the supplied value is used.
    """

    def __init__(self, nr_of_models: int,
                 probability_to_stay: Union[float, None] = None):
        self.nr_of_models = nr_of_models
        if nr_of_models == 1:
            self.probability_to_stay = 1
        else:
            if probability_to_stay is None:
                self.probability_to_stay = 1 / nr_of_models
            else:
                self.probability_to_stay = min(
                    max(probability_to_stay, 0), 1)

    def _get_discrete_rv(self, m):
        p_stay = self.probability_to_stay
        p_move = (1 - p_stay) / (self.nr_of_models - 1)
        probabilities = [p_stay if n == m else p_move
                         for n in range(self.nr_of_models)]
        return RV('rv_discrete',
                  values=(range(len(probabilities)), probabilities))

    def rvs(self, m: int) -> int:
        """Sample a Kernel jump from model ``m`` to another model.

        Parameters
        ----------
        m: int
            Model source nr.

        Returns
        -------
        target: int
            Target model nr.
        """

        if not 0 <= m <= self.nr_of_models - 1:
            raise Exception('m has to be between 0 and nr_of_models - 1')
        if self.nr_of_models == 1:
            return 0  # always stay, no other choice
        else:
            return int(self._get_discrete_rv(m).rvs())

    def pmf(self, n: int, m: int) -> float:
        """Probability mass function for a jump to target `n` from source `m`.

        Parameters
        ----------
        n: int
            Model target nr.
        m: int
            Model source nr.

        Returns
        -------
        probability: float
            Probability with which to jump from ``m`` to ``n``.
        """

        if not (0 <= n <= self.nr_of_models
                and 0 <= m <= self.nr_of_models - 1):
            raise Exception(
                'n and m have to be between 0 and nr_of_models - 1')
        if self.nr_of_models == 1:
            return 1 if n == m else 0
        else:
            return self._get_discrete_rv(m).pmf(n)
