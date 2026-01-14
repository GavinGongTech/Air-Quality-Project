# This branch of the hybrid model hopes to encapsulate the spatial dependencies of air quality data 
# across different locations and times. Important features that I have so far to work with are latitude and
# longitude for now. 

# We have the latitude and longitude of sensors that recorded the PM2.5 levels. 
# The paper said that they used 13 sensors, but the data seems to only use data from 12 sensors.
# So I will use 12 sensors; may have just been negligence on the part of the authors

# There are some other columns in the data, not sure if they are particularly useful.

# The features given that I use for spatial modeling
# Sensor ID: Individual sensors may show patterns of PM2.5 levels over time. 
# Latitude, Longitude, coordinates for location of sensors
# The latitude and longitude are, obviously, identical for each individual sensor across all times
# So they do not provide any temporal information by themselves.
# Same with the lat_m and lon_m, which are the locations of the sensors on the UTM unit. 

# Since all spatial features given are constant over time, no temporal information obtained. 
# Therefore, I will just use them for structure in my other model, 


# For the spatial aspect, I will use the data of Pm2.5 concentration at locations near each sensor 
# to provide some contextual information about the air quality in the area.
# My original hypothesis would be that the closer the data point to sensor, the more relevant it is for
# predicting the sensor's readings. This feels like a kernel density estimation problem, where I want to
# weigh the influence of nearby data points more heavily than those further away. 
# Chapter 9 of MLCS! Splines, Basis expansions, kernels are good applications here. 

# I think, that since I can't get contextual data without regression first, the 
# spatial branch will need to be based off the temporal model. So I will evaluate temporal model first. 

