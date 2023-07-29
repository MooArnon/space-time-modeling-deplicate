#--------#
# Import #
#----------------------------------------------------------------------------#
import math

import pandas as pd

from space_time_modeling.preprocess import BasePreprocessing

#----------------------------#
# Series preprocessing class #
#----------------------------------------------------------------------------#
class SeriesPreprocess(BasePreprocessing):
    
    def __init__(
            self, 
            column: str, 
            diff: bool,
            window_size: int, 
            mode: str = "csv"
    ) -> None:
        """Initiate the series processing

        Parameters
        ----------
        column : str
            Target column
        diff : bool, optional
            If True, calculate diff and use it as an features
            If False, Use the target column
        window_size : int, optional
            The size of input window, 
            by default 60
        mode : str, optional
            Mode of source of data, 
            by default "csv"
        """
        super().__init__(mode)
        
        # Window size
        self.window_size = window_size
        
        # Target column
        self.column = column
        
        # Diff activation
        self.diff = diff
    
    #------#
    # Main #
    #------------------------------------------------------------------------#
    def process(
            self, 
            df: pd.DataFrame
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Processing the data

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        tuple[list[list[float]], list[list[float]]]
            Return x and y
        """
        # Use the original DF as an input
        if not self.diff:
            
            # Select column for series
            series = df[self.column].to_list()

        # Generate the diff column 
        else:
            
            # Get diff df
            df['diff'] = df[self.column].diff()
            
            # Shift diff value
            df['diff'] = df['diff'].shift(-1)
            
            # Drop NaN
            df = df.dropna()

            # Select diff for series
            series = df['diff'].to_list()
        
        # Get features
        x = self.feature_engineer(series)

        # Get label
        # `fit` indicates that all data are fit in window
        # The error will not occurs from labeling
        y, fit = self.labeling(series)

        # Slice the last element from x
        if fit is False:

            x = x[:-1]

        return x, y
    
    #---------------------#
    # Feature Engineering #
    #------------------------------------------------------------------------#
    
    def feature_engineer(
            self, 
            series: pd.Series, 
    ) -> list[list[float]]:
        """_summary_

        Parameters
        ----------
        series: pd.Series :
            Input series, which slide from data frame
            
        Returns
        -------
        list[list[float]]
            Result of data frame
        """

        return self.loop_2_get_feature(series)
    
    #------------------------------------------------------------------------#
    
    def loop_2_get_feature(self, series: list[float]) -> list[list[float]]:
        """Loop for create x and return as list of list

        Parameters
        ----------
        series: pd.Series :
            Input series, which slide from data frame
            
        Returns
        -------
        list[list[float]]
            List of each partitioned series
        """
        # Result -> returned as list of list
        x = []
        
        # Loop to get features
        for index in range(self.get_num_feature(series)):
            
            # Start index = window_size * index
            # 0 * 4 = 0
            # 1 * 4 = 4
            index_start = self.window_size * index
            
            # End index = index_start + window_size
            # 0 + 4 = 4
            # 4 + 4 = 8
            index_end = index_start + self.window_size
            
            # Append the slide  into x
            x.append(
                series[index_start: index_end]
            )
            
        return x  
    
    #----------#
    # Labeling #
    #------------------------------------------------------------------------#
    
    def labeling(self, series: pd.Series) -> tuple[list[list[float]], bool]:
        """Label series data

        Parameters
        ----------
        series: pd.Series :
            Input series, which slide from data frame

        Returns
        -------
        tuple[list[list[float]], bool] :
            list[list[float]]
                the target label
            bool
                If True, nothing to do in this section.
                If false, the index_end + 1 is out of range of target
                series.
        """
        
        y = []

        for index in range(self.get_num_feature(series)):
            
            # index_end = start_index + window_size
            index_end = (self.window_size * index) + self.window_size

            try:

                # Append the label in y
                y.append([series[index_end]])

                # All y is fitted in number of features
                fit = True

            except:
                
                # Y Number of index_end + 1 is out of record ranges
                fit = False

        return y, fit
    
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    def get_num_feature(self, series: list[list[float]]) -> int:
        """Get the number of features.

        Parameters
        ----------
        series : list[list[float]]
            Target series

        Returns
        -------
        int
            Number of features, the len(list).
        """
        return math.floor(len(series) // self.window_size)
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
