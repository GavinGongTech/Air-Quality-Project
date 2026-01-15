# Air-Quality-Project

# I developed multiple models to try and predict the PM2.5 levels.

# Starting with First_LSTM, Second_LSTM, Third_LSTM, these models did not incorporate the spatial dependencies yet, only the temporal dependencies. 

# The most recent kernel-based method, LSTM_with_Kernel.py, incorporate the spatial aspect via the kernel-based method. We use distance isotropic kernel and wind-anisotropic kernel for evaluation. 

# The last step from the model is inference across the entire city grid for Elizabeth, rather than the 12 sensors. This final model is Kernel_LSTM_generalize.py.

# Original Paper: An Air Quality Digital Twin for Real-Time Outdoor Air Quality Monitoring and Prediction published by the Department of Civil and Environmental
# Engineering of Rutgers University

# My model outperforms the original model in terms of efficiency of generating predictions and relative accuracy. Currently working on deployment so that customers/users can harness the capabilities of the model in live time. 
