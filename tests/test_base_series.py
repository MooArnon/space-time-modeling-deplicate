#--------#
# Import #
#----------------------------------------------------------------------------#
import itertools
import os
import math

import pandas as pd
import pytest

#------#
# Path #
#----------------------------------------------------------------------------#

from space_time_modeling import get_preprocess_engine

#----------#
# Constant #
#----------------------------------------------------------------------------#

WINDOW_SIZE = 10

#----------------------------------------------------------------------------#
class TestBasePreprocess:
    
    #------------------#
    # Engine selection #
    #------------------------------------------------------------------------#
    
    def test_engine_selection_nothing_selected(self):
        """Test default selection"""
        # Get preprocessing engine
        prep = get_preprocess_engine(
            column="Open", 
            window_size=WINDOW_SIZE,
            diff=False,
        )
        
    #------------------------------------------------------------------------#
    
    def test_engine_selection_series_selected(self):
        """Test series selection"""
        prep = get_preprocess_engine(
            column="Open", 
            window_size=WINDOW_SIZE,
            diff=False,
            engine="series"
        )
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#