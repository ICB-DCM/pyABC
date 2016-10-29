import os
import unittest

import pandas as pd

from pyabc.loader import ABCLoader

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "test_data")


@unittest.skip("Probably borken test")
class NewPosteriorPLotter(unittest.TestCase):
    def setUp(self):
        self.data_store = pd.HDFStore(os.path.join(DATA_FOLDER, "version2016.h5"))

    def tearDown(self):
        self.data_store.close()

    def test_max(self):
        plotter = ABCLoader(self.data_store)
        map = plotter.maximum_a_posteriori()
        self.assertTrue(self.data_store.map.equals(map))

    def test_gt_mass(self):
        plotter = ABCLoader(self.data_store)
        gt_mass = plotter.average_mass_at_tround_truth()
        self.assertTrue(self.data_store.average.equals(gt_mass))

    def test_confusion(self):
        plotter = ABCLoader(self.data_store)
        confusion = plotter.confusion_matrices_table
        self.assertTrue(self.data_store.confusion_matrices.equals(confusion))


if __name__ == "__main__":
    unittest.main()
