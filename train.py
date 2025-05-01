import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("midi_songs", exist_ok=True)

def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.midi"):
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
    with open('data/notes_dual', 'wb') as f:
        pickle.dump(notes, f)
    return notes

def prepare_sequences_adaptive(notes, max_length=100):
    sequence_length = min(len(notes) - 1, max_length)
    pitches = sorted(set(n[0] for n in notes))
    durations = sorted(set(n[1] for n in notes))
    pitch_to_int = {p: i for i, p in enumerate(pitches)}
    dur_to_int = {d: i for i, d in enumerate(durations)}
    sequence_data = []
    targets = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        pitch_seq = [pitch_to_int[n[0]] for n in seq_in]
        dur_seq = [dur_to_int[n[1]] for n in seq_in]
        combined_seq = [[p, d] for p, d in zip(pitch_seq, dur_seq)]
        sequence_data.append(combined_seq)
        targets.append(pitch_to_int[seq_out[0]])
    network_input = np.asarray(sequence_data, dtype=np.float32)
    network_input[:, :, 0] /= float(len(pitches))
    network_input[:, :, 1] /= float(len(durations))
    network_output = to_categorical(targets, num_classes=len(pitches))
    return network_input, network_output, len(pitches)

def create_network(network_input, n_pitch):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_pitch))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train_model():
    create_directories()
    notes = get_notes()
    network_input, network_output, n_pitch = prepare_sequences_adaptive(notes)
    model = create_network(network_input, n_pitch)
    filepath = "weights/weights-best.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=[checkpoint])

if __name__ == "__main__":
    train_model()
