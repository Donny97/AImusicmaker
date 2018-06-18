from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np
import mido
import os
########### PROCESS MIDI FILE #############

notes = []
#count = 10
for filename in os.listdir('data/train/'):
    
    mid = MidiFile('data/train/'+filename)

    for msg in mid:
        if not msg.is_meta:
            if msg.type == 'note_on':
                note = msg.bytes()
                note[0] = msg.channel
                note.append(msg.time)
                notes.append(note)
    # count -=1
    # if count == 0: break

tr_size = len(notes)
###########################################

######## SCALE DATA TO BETWEEN 0, 1 #######
max_val = np.max(notes,axis=0)
min_val = np.min(notes,axis=0)

for note in notes:
    for i in range(0,4):
        note[i] = (note[i]-min_val[i])/(max_val[i]-min_val[i])

###########################################

############ CREATE DATA, LABELS ##########
X = []
Y = []
n_prev = 100
# n_prev notes to predict the (n_prev+1)th note
for i in range(len(notes)-n_prev):
    x = notes[i:i+n_prev]
    y = notes[i+n_prev]
    X.append(x)
    Y.append(y)

# save a seed to do prediction later
ind = np.random.randint(tr_size-2*n_prev)
seed = notes[ind:ind+n_prev]

###########################################

model = 0
if not os.path.isfile("model_file.h5"):
    ############### BUILD MODEL ###############
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(n_prev, 4), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, input_shape=(n_prev, 4), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('linear'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer='rmsprop')
    checkpoint = ModelCheckpoint('model_file.h5')

else:
    print("Loding model from model_file.h5")
    model = load_model("model_file.h5")

X = np.array(X)
Y = np.array(Y)

model.fit(X, Y, batch_size=300, epochs=200, verbose=1, callbacks=[checkpoint])

###########################################

############ MAKE PREDICTIONS #############
prediction = []
y = seed
y = np.expand_dims(y, axis=0)

for i in range(1000):
    preds = model.predict(y)
    y = np.squeeze(y)
    y = np.concatenate((y, preds))
    y = y[1:]
    y = np.expand_dims(y, axis=0)
    preds = np.squeeze(preds)
    prediction.append(preds)
    # preds = model.predict(x)
    # x.append(preds)
    # x = x[1:]
    # prediction.append(preds)

for pred in prediction:
    for i in range(0,4):
        pred[i] = pred[i] * (max_val[i]-min_val[i]) + min_val[i]
        if pred[i] < min_val[i]:
            pred[i] = min_val[i]

        if pred[i] >= max_val[i]:
            pred[i] = max_val[i]
	
###########################################


###### SAVING TRACK FROM BYTES DATA #######
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for note in prediction:
    # 147 means note_on
    note = np.insert(note, 1, 144)
    bytes = np.round(note).astype(int)
    msg = Message.from_bytes(bytes[1:4])
    msg.time = int(note[4]/0.00125) # to rescale to midi's delta ticks. arbitrary value for now.
    msg.channel = bytes[0]
    print(msg)
    track.append(msg)

mid.save('new_song.mid')

