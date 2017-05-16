from pyabc import (ABCSMC, Distribution, RV)
import scipy as sp

def test_resume(db_path):
    def model(parameter):
        return {"data": parameter["mean"] + sp.randn()}

    prior = Distribution(mean=RV("uniform", 0, 5))

    def distance(x, y):
        x_data = x["data"]
        y_data = y["data"]
        return abs(x_data - y_data)

    abc = ABCSMC(model, prior, distance)
    run_id = abc.new(db_path, {"data": 2.5})
    print("Run ID:", run_id)
    history = abc.run(minimum_epsilon=.1, max_nr_populations=2)

    abc_continued = ABCSMC(model, prior, distance)
    run_id_continued = abc_continued.load(db_path, run_id)
    print("Run ID continued:", run_id_continued)
    history_continued = abc_continued.run(minimum_epsilon=.1,
                                          max_nr_populations=2)
