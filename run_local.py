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
    os.path.join("data", "BTC-Hourly.csv")
)

print(df.shape)

# Get preprocessing engine
prep = get_preprocess_engine(
    column="open", 
    window_size=WINDOW_SIZE,
    diff=False,
    engine="series"
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
    architecture = "n-beats",
    input_size = WINDOW_SIZE,
    hidden_size = 128,
    num_stacks = 8,
    num_blocks = 16
)

#----------------------------------------------------------------------------#
# Train #
#-------#

model_engine.modeling(
    x, 
    y, 
    result_name = "n-beats",
    epochs=100,
    train_kwargs={"lr": 3e-7},
    test_ratio = 0.15,
)

#----------------------------------------------------------------------------#
