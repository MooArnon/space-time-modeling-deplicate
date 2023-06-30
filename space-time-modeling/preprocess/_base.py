import pandas as pd


class BasePreprocessing:
    
    def __init__(self, mode: str="csv", path: str = None) -> None:
        """ Initialize BasePreprocessing
        
        Parameters
        ----------
        mode: str : 
            Mode of data reading
            DEFAULT = "csv"
        path: str :
            Path need to be indicated if mode is file-like 
        """
        if mode == "csv":

            self.df = pd.read_csv(path)
    
    #------------------------------------------------------------------------#
    
    def process_x(self, df: pd.DataFrame) -> list[float]:
        """ Process feature data 
        
        Parameters
        ----------
        df: pd.DataFrame :
            the input data frame
            
        Returns
        -------
        x: list[float] :
        """
        return list(map(self.feature_engineer, df))
    #------------------------------------------------------------------------#
    
    def label_y(self, df: pd.DataFrame) -> list[float]:
        """ Process label data  
        
        Parameters
        ----------
        df: pd.DataFrame :
            the input data frame
            
        Returns
        -------
        y: list[float] :
        """
        return list(map(self.labeling, df))
    
    #------------------------------------------------------------------------#
    
    
    
    #------------------------------------------------------------------------#
    
    
    
    #------------------------------------------------------------------------#