#--------#
# Import #
#----------------------------------------------------------------------------#
import os

import pandas as pd

from space_time_modeling import get_preprocess_engine
from space_time_modeling import get_model_engine

#----------#
# Variable #
#----------------------------------------------------------------------------#

WINDOW_SIZE = 10

#--------------#
# Process data #
#----------------------------------------------------------------------------#

# Read data
df = pd.read_csv(
    os.path.join("tests", "BTC-USD.csv")
)

# Get preprocessing engine
prep = get_preprocess_engine(
    column="Open", 
    window_size=WINDOW_SIZE,
    diff=False,
    engine="indicator"
)

# Calculate x and y
x, y = prep.process(df=df)


#----------#
# Modeling #
#----------------------------------------------------------------------------#
# Get engine #
#------------#

model_engine = get_model_engine(
    engine="deep",
    architecture = "nn",
    input_size = WINDOW_SIZE
)

#----------------------------------------------------------------------------#
# Train #
#-------#

model_engine.modeling(
    x, 
    y, 
    result_name = "RNN",
    epochs=100,
    train_kwargs={"lr": 5e-5},
    test_ratio = 0.15
)

#----------------------------------------------------------------------------#
