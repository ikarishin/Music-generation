import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

def create_network(output_dim, input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def get_notes():
    notes = []
    midi_files = glob.glob("midi_songs/*.mid") + glob.glob("midi_songs/*.midi")
    for file in midi_files:
        midi = converter.parse(file)
        print("Parsing", file)
        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                duration = element.quarterLength
                notes.append((pitch, duration))
            elif isinstance(element, chord.Chord):
                chord_symbol = '.'.join(str(n) for n in element.normalOrder)
                duration = element.quarterLength
                notes.append((chord_symbol, duration))
    os.makedirs('data', exist_ok=True)
    with open('data/notes_dual', 'wb') as f:
        pickle.dump(notes, f)
    return notes

def prepare_sequences(notes, sequence_length=100):
    pitches = sorted(set(n[0] for n in notes))
    durations = sorted(set(n[1] for n in notes))
    pitch_to_int = {p: i for i, p in enumerate(pitches)}
    dur_to_int = {d: i for i, d in enumerate(durations)}

    input_seq = []
    pitch_output = []
    dur_output = []

    for i in range(len(notes) - sequence_length):
        seq = notes[i:i+sequence_length]
        input_vec = [[pitch_to_int[n[0]], dur_to_int[n[1]]] for n in seq]
        input_seq.append(input_vec)
        pitch_output.append(pitch_to_int[notes[i+sequence_length][0]])
        dur_output.append(dur_to_int[notes[i+sequence_length][1]])

    input_seq = np.array(input_seq) / [len(pitches), len(durations)]
    pitch_output = to_categorical(pitch_output, num_classes=len(pitches))
    dur_output = to_categorical(dur_output, num_classes=len(durations))

    return input_seq, pitch_output, dur_output, pitches, durations

def train_models():
    notes = get_notes()
    X, y_pitch, y_dur, pitches, durations = prepare_sequences(notes)

    with open("data/pitches.pkl", "wb") as f:
        pickle.dump(pitches, f)
    with open("data/durations.pkl", "wb") as f:
        pickle.dump(durations, f)

    model_pitch = create_network(len(pitches), (X.shape[1], X.shape[2]))
    model_dur = create_network(len(durations), (X.shape[1], X.shape[2]))

    os.makedirs("weights", exist_ok=True)
    cp1 = ModelCheckpoint("weights/weights-pitch.keras", save_best_only=True, monitor='loss')
    cp2 = ModelCheckpoint("weights/weights-duration.keras", save_best_only=True, monitor='loss')
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    model_pitch.fit(X, y_pitch, epochs=100, batch_size=64, callbacks=[cp1, es])
    model_dur.fit(X, y_dur, epochs=100, batch_size=64, callbacks=[cp2, es])
