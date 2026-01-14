# This branch of the hybrid model hopes to encapsulate the temporal dependencies

# Just some ridiculous notation; in their time column, they put YYYY/MM/DD followed by the military time
# But when the military time is 00:00:00, they just omit it. Just had to use a regex to capture this format
# and make sure this did not lead to undefined values in my analysis. 

# We have 170 days of complete data to work with and 5/6 of a 171th day. Let's see what we can do.

# The features given that I use for temporal modeling are the meteorological variables
# collected in the paper; they are 
"""
1. Precipitation
2. Relative Humidity
3. Pressure
4. Temperature
5. Wind Speed
6. Wind Direction
7. Time
"""

# Due to this being a time-based situation, I think employing an LSTM model would be an appropriate initial approach.
# LSTMs are well-suited for sequence prediction problems because they can maintain a memory of previous inputs due to their internal state.

# For the LSTM, the initial approach will be to treat each sensor's data and features individually
# and keep them UNCORRELATED. In the future, I will see to it that I evaluate what happens
# when their data are integrated together. 

# I call it LSTM_FIRST; it will take in an input at time t, which is a sliding window of the last L 
# hours of features; its output will be the predicted PM2.5 level at time t + h (h =1 means one hour ahead; h=0 means nowcast)
# Right now, 
