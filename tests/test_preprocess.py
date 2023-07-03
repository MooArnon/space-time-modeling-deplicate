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
from space_time_modeling.preprocess import SeriesPreprocess


#----------#
# Variable #
#----------------------------------------------------------------------------#
COLUMN = "Open"
DIFF = False

#------#
# Test #
#----------------------------------------------------------------------------#
# Preprocessing #
#---------------#
# Test class #
#------------#
class PreprocessingTest(unittest.TestCase):
    
    #--------------#
    # Main process #
    #------------------------------------------------------------------------#
    
    def test_len_h345_w3(self):
        """Number of x and y must be equal """
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(345)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=3, column=COLUMN)

        ## Use process
        x, y = prep.process(df=df)
        
        self.assertTrue(
            len(x) == len(y)
        )
    
    #------------------------------------------------------------------------#

    def test_len_h645_h4(self):
        """Number of x and y must be equal """
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(645)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=4, column=COLUMN)

        ## Use process
        x, y = prep.process(df=df)
        
        self.assertTrue(
            len(x) == len(y)
        )
        
    #------------------------------------------------------------------------#

    def test_len_h351_h16(self):
        """Number of x and y must be equal """
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(351)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=16, column=COLUMN)

        ## Use process
        x, y = prep.process(df=df)
        
        self.assertTrue(
            len(x) == len(y)
        )
    
    #---------#
    # Element #
    #------------------------------------------------------------------------#
    # Length of Features #
    #--------------------#
    
    def test_len_feature_h25_w4(self):
        """The number of list in list must equal to number of record 
        divided WINDOW_SIZE. Which is floor estimated
        
        Examples
        --------
        record = 21, window_size = 5
            The len(list) must be floor(24/5) = floor(4.8) = 4 
        """
        
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(25)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=4, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df)
    
        self.assertTrue(
            (len(x) == math.floor(df.shape[0]//4)) | \
                (len(x) == math.floor(df.shape[0]//4) - 1)
        )
    
    #------------------------------------------------------------------------#
    
    def test_len_feature_h27_w4(self):
        """The number of list in list must equal to number of record 
        divided WINDOW_SIZE. Which is floor estimated
        
        Examples
        --------
        record = 21, window_size = 5
            The len(list) must be floor(24/5) = floor(4.8) = 4 
        """
        
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(27)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=4, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df)
    
        self.assertTrue(
            (len(x) == math.floor(df.shape[0]//4)) | \
                (len(x) == math.floor(df.shape[0]//4) - 1)
        )
    
    #------------------------------------------------------------------------#
    
    def test_len_feature_h35_w5(self):
        """The number of list in list must equal to number of record 
        divided WINDOW_SIZE. Which is floor estimated
        
        Examples
        --------
        record = 21, window_size = 5
            The len(list) must be floor(24/5) = floor(4.8) = 4 
        """
        
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(35)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=5, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df)
    
        self.assertTrue(
            (len(x) == math.floor(df.shape[0]//5)) | \
                (len(x) == math.floor(df.shape[0]//5) - 1)
        )
        
    #------------------------------------------------------------------------#
    # Similarity of each features #
    #-----------------------------#
    
    def test_element_feature_h25_w4(self):
        """Each element in x must be equal to the original record in 
        data frame
        
        Examples
        --------
        original = [4, 5, 6, 3, 7, 8]
        feature = [[4, 5, 6], [3, 7, 8]] -> [4, 5, 6, 3, 7, 8] 
            PASS
        feature = [[4, 5, 6], [6, 3, 7]] -> [4, 5, 6, 6, 3, 7] 
            FAILED
        """
        
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(24)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=4, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df, diff=DIFF)
        
        # join x in each element
        x_joined = list(itertools.chain.from_iterable(x))
        
        # Slice data in frame
        original = df[COLUMN].to_list()[:len(x_joined)]
        
        self.assertTrue(
            x_joined == original
        )
    
    #----------------------------------------------------------------------------#

    def test_element_feature_h37_w3(self):
        """Each element in x must be equal to the original record in 
        data frame
        
        Examples
        --------
        original = [4, 5, 6, 3, 7, 8]
        feature = [[4, 5, 6], [3, 7, 8]] -> [4, 5, 6, 3, 7, 8] 
            PASS
        feature = [[4, 5, 6], [6, 3, 7]] -> [4, 5, 6, 6, 3, 7] 
            FAILED
        """
        
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(37)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=3, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df, diff=DIFF)
        
        # join x in each element
        x_joined = list(itertools.chain.from_iterable(x))
        
        # Slice data in frame
        original = df[COLUMN].to_list()[:len(x_joined)]
        
        self.assertTrue(
            x_joined == original
        )
        
    #------------------------------------------------------------------------#
    # diff calculation #
    #------------------#
    
    def test_diff_cal(self):
        """Check the diff"""
        df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(1000)
        
        # Perform Series preprocess
        ## Initialize SeriesPreprocess
        prep = SeriesPreprocess(window_size=3, column=COLUMN)
        
        ## Use process
        x, y = prep.process(df=df, diff=True)
        
        # 1st calculated
        cal = x[0][0]
        
        # Target
        df_list = df[COLUMN].tolist()
        
        # manual diff
        manual_diff = df_list[1] - df_list[0]
        
        self.assertEqual(cal, manual_diff)
    
#--------------#
# Running test #
#----------------------------------------------------------------------------#

if __name__ == '__main__':
    
    unittest.main()
