#--------#
# Import #
#----------------------------------------------------------------------------#
import os
import shutil

import pandas as pd
import pytest


#------#
# Path #
#----------------------------------------------------------------------------#
from space_time_modeling.modeling import BaseModeling, get_model_engine
from space_time_modeling.preprocess import SeriesPreprocess


#----------#
# Variable #
#----------------------------------------------------------------------------#
COLUMN = "Open"
WINDOW_SIZE = 3
df = pd.read_csv(
            os.path.join("tests", "BTC-USD.csv")
        ).head(345)

prep = SeriesPreprocess(window_size=WINDOW_SIZE, column=COLUMN, diff=False)

## Use process
x, y = prep.process(df=df)

#------#
# Test #
#----------------------------------------------------------------------------#
# Preprocessing #
#---------------#
# Test class #
#------------#
class TestModelingTest:
    
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

        assert not intersection
        
    #------------------------------------------------------------------------#
    
    def test_get_model_engine(self):
        """Test creating torch.nn.Module"""
        
        _ = get_model_engine(
            engine="deep",
            architecture = "nn",
            input_size = 60
        ) 
        
    #------------------------------------------------------------------------#
    
    #-------------#
    # Deep engine #
    #------------------------------------------------------------------------#
    # Model selection # 
    #-----------------#
    
    def test_set_all_none(self):
        """The package need to choose engine oif nothing determined"""
        
        self.make_dir()
        
        _ = get_model_engine(
            input_size = WINDOW_SIZE
        )
        
        # Train it
        self.train(_)
        
        assert "result" in os.listdir()
        
        shutil.rmtree("result")
    
    #------------------------------------------------------------------------#
    
    def test_set_deep_eng_arch_none(self):
        """The package need to choose arch oif nothing determined"""
        
        self.make_dir()
        
        _ = get_model_engine(
            engine="deep",
            input_size = WINDOW_SIZE
        )
        
        # Train it
        self.train(_)
        
        assert "result" in os.listdir()
        
        shutil.rmtree("result")
    
    #------------------------------------------------------------------------#
    # Exportation #
    #-------------#
    
    def test_write_model(self):
        """The module works smoothly without fatal problem"""
        
        self.make_dir()
        
        _ = get_model_engine(
            engine="deep",
            architecture = "nn",
            input_size = WINDOW_SIZE
        ) 
        
        self.train(_)
        
        assert "result" in os.listdir()
        
        shutil.rmtree("result")

    #------------------------------------------------------------------------#
    
    def test_element_in_exportation(self):
        """All element must be exported"""
        
        self.make_dir()
        
        _ = get_model_engine(
            engine="deep",
            architecture = "nn",
            input_size = WINDOW_SIZE
        ) 
        
        self.train(_)
        
        result_name = os.listdir(
            os.path.join("result")
        )[-1]
        
        result_path = os.path.join("result", result_name)
        
        result_lst = os.listdir(result_path)
        
        # Define the desired file extensions
        file_format = ['.txt', '.pth', '.png']
        
        filtered_list = [
            file for file in result_lst 
            if any(file.endswith(ext) for ext in file_format)
        ]
        
        shutil.rmtree("result")
        
        assert sorted(filtered_list) == sorted(filtered_list)
    
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    @staticmethod
    def make_dir():
        dir_lst = os.listdir()
        
        if "result" in dir_lst:
            
            shutil.rmtree("result")

    #------------------------------------------------------------------------#
    
    @staticmethod
    def train(model):
        "Just constructed as a utilities"
        model.modeling(
            x, 
            y, 
            result_name = "NN_TEST_MODEL",
            epochs=100,
            train_kwargs={"lr": 5e-5},
            test_ratio = 0.15, 
        )
        return model
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#