# AImusicmaker
An AI composer 

The aim was to do a very basic AI music composer in 100 lines of code. This script takes a midi file and extracts just the notes, converting the piece into a large array of notes.

The model is built on Keras, using the LSTM layer. Since each note is a vector of pitch, velocity, and time, I trained the model to predict the next note given an array of n_prev previous notes (where n_prev is a parameter to be tuned).
