import os
from tempfile import gettempdir

from pyabc.external import R
import pyabc


r_file = "doc/examples/myRModel.R"


def test_r():
    """
    This is basically just the using_R notebook.
    """
    r = R(r_file)
    r.display_source_ipython()
    model = r.model("myModel")
    distance = r.distance("myDistance")
    sum_stat = r.summary_statistics("mySummaryStatistics")
    prior = pyabc.Distribution(meanX=pyabc.RV("uniform", 0, 10),
                               meanY=pyabc.RV("uniform", 0, 10))
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)
    abc = pyabc.ABCSMC(model, prior, distance,
                       summary_statistics=sum_stat,
                       sampler=sampler)
    db = "sqlite:///" + os.path.join(gettempdir(), "test_external.db")
    abc.new(db, r.observation("mySumStatData"))
    history = abc.run(minimum_epsilon=0.9, max_nr_populations=2)
    history.get_weighted_sum_stats_for_model(m=0, t=1)[1][0]["cars"].head()
