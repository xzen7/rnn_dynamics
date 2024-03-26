The repo is for MATH 8803

The paper it is based on: Opening the Black Box: Low-Dimensional Dynamics
in High-Dimensional Recurrent Neural Networks

The basic_rnn dir replicates the 3rd example: find and visualize the slow points of the simple RNN in a 2-point averaging task

The neurondata_rnn dir use **Nuri's data** to train the rnn listed in the neuro_gym project, and find the fixed/slow points, or other interesting dynamics. If time permits, I'll **train different RNN subtypes** and make a user interface to visualize dynamics for each. compare if the internal dynamics are similar. 

update 3/25/2024
Listened to the neuro seminar today and it looks like the RNN dynamics won't be the same if the structures are different, but if neural data is available, probably we can compare the most similar network dynamics and infer the neuron dynamics. 