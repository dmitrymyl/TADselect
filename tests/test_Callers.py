"""
test_Callers
----------------------------------
Tests for `CallerClasses` module.
"""

from sys import path
from os.path import abspath, join, dirname, isdir, isfile
import unittest

from src.CallerClasses import *

class TestLavaburst(unittest.TestCase):
    """Test Lavaburst class"""

    def setUp(self):
        pass

    def test_creation_of_object(self):

        lc = LavaburstCaller(['S2'], ['../data/S2.20000.cool'], 'cool', assembly='dm3', resolution=20000, balance=True, chr='chr2L')
        lc.load_segmentation( lc.call(0.9) )
        lc.load_segmentation( lc.call(1.9) )
        df = lc.segmentation2df()

        self.assertEqual(len(lc._segmentations.keys()), 1)
        self.assertEqual(len(lc._segmentations['S2'].keys()), 2)
        # self.assertEqual(variable, value)
        # self.assertTrue(condition)

if __name__ == '__main__':
    unittest.main()