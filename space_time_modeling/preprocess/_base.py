#--------#
# Import #
#----------------------------------------------------------------------------#

import pandas as pd

#-------#
# Class #
#----------------------------------------------------------------------------#

class BasePreprocessing:
    
    def __init__(self, mode: str="csv") -> None:
        """ Initialize BasePreprocessing
        
        Parameters
        ----------
        mode: str : 
            Mode of data reading
            DEFAULT = "csv"
        path: str :
            Path need to be indicated if mode is file-like 
        """
        self.mode = mode
    
    #---------------#
    # Preprocessing #
    #------------------------------------------------------------------------#
    
    def feature_engineer(self, df:pd.DataFrame):
        """Create features by receive the pandas.DataFrame 
        This method need to be implemented.
        
        Parameters
        ----------
        df: pd.DataFrame :
            the input data frame

        Raises
        ------
        NotImplementedError
            Child classes need to implement this fn
        """
        raise NotImplementedError("Child classes need to implement this fn")
    
    #------------------------------------------------------------------------#
    
    def labeling(self, df:pd.DataFrame):
        """Create label by receive the pandas.DataFrame 
        This method need to be implemented.
        
        Parameters
        ----------
        df: pd.DataFrame :
            the input data frame

        Raises
        ------
        NotImplementedError
            Child classes need to implement this fn
        """
        raise NotImplementedError("Child classes need to implement this fn")
    
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    def get_data(self, path: str=None) -> pd.DataFrame:
        """Get data from sources

        Parameters
        ----------
        path : str, optional
            Path of data, if file-liked, by default None

        Returns
        -------
        pd.DataFrame
            The result as pandas data frame
        """
        # Check if mode is csv
        if self.mode == "csv":

            df = pd.read_csv(path)
        
        # Check if mode is excel
        elif self.mode == "excel":
            
            df = pd.read_excel(path)
            
        return df
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
