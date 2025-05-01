import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("midi_songs", exist_ok=True)
    os.makedirs("output", exist_ok=True)

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
    with open('data/notes_dual', 'wb') as f:
        pickle.dump(notes, f)
    return notes

def prepare_sequences_adaptive(notes, max_length=100):
    if len(notes) < 10:
        raise ValueError("Not enough data")
    sequence_length = min(len(notes) - 1, max_length)
    pitches = sorted(set(n[0] for n in notes))
    durations = sorted(set(n[1] for n in notes))
    pitch_to_int = {p: i for i, p in enumerate(pitches)}
    dur_to_int = {d: i for i, d in enumerate(durations)}
    sequence_data = []
    pitch_targets = []
    dur_targets = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        pitch_seq = [pitch_to_int[n[0]] for n in seq_in]
        dur_seq = [dur_to_int[n[1]] for n in seq_in]
        combined_seq = [[p, d] for p, d in zip(pitch_seq, dur_seq)]
        sequence_data.append(combined_seq)
        pitch_targets.append(pitch_to_int[seq_out[0]])
        dur_targets.append(dur_to_int[seq_out[1]])
    network_input = np.asarray(sequence_data, dtype=np.float32)
    network_input[:, :, 0] /= float(len(pitches))
    network_input[:, :, 1] /= float(len(durations))
    pitch_output = to_categorical(pitch_targets, num_classes=len(pitches))
    dur_output = to_categorical(dur_targets, num_classes=len(durations))
    return network_input, pitch_output, dur_output, len(pitches), len(durations)

def create_network(output_size, input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_input, network_output, filename):
    filepath = f"weights/{filename}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=[checkpoint, early_stop])

if __name__ == "__main__":
    create_directories()
    notes = get_notes()
    network_input, pitch_output, dur_output, n_pitch, n_dur = prepare_sequences_adaptive(notes)
    model_pitch = create_network(n_pitch, (network_input.shape[1], network_input.shape[2]))
    model_dur = create_network(n_dur, (network_input.shape[1], network_input.shape[2]))
    train(model_pitch, network_input, pitch_output, filename="weights-pitch")
    train(model_dur, network_input, dur_output, filename="weights-duration")
