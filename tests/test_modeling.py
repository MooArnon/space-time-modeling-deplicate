#--------#
# Import #
#----------------------------------------------------------------------------#
import itertools
import os
import math

import pandas as pd
import unittest

#------#
# Path #
#----------------------------------------------------------------------------#
from space_time_modeling.modeling import BaseModeling
from space_time_modeling.preprocess import SeriesPreprocess


#----------#
# Variable #
#----------------------------------------------------------------------------#
COLUMN = "Open"
df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(345)

prep = SeriesPreprocess(window_size=3, column=COLUMN, diff=False)

## Use process
x, y = prep.process(df=df)

#------#
# Test #
#----------------------------------------------------------------------------#
# Preprocessing #
#---------------#
# Test class #
#------------#
class ModelingTest(unittest.TestCase):
    
    #--------------#
    # Main process #
    #------------------------------------------------------------------------#
    
    def test_train_test_sampling(self):
        """The train must not be in test"""
        # Initiate base model
        model = BaseModeling()
        
        # Sample
        train, test = model.sample(x)
        
        train_set = set(map(tuple, train))
        test_set = set(map(tuple, test))

        intersection = train_set.intersection(test_set)

        self.assertEqual(len(intersection), 0)
        
    
#--------------#
# Running test #
#----------------------------------------------------------------------------#

if __name__ == '__main__':
    
    unittest.main()
