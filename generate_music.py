import os
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization as BatchNorm

def create_network(input_shape, n_pitch):
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
    model.add(Dense(n_pitch))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def create_midi(prediction_output):
    os.makedirs("output", exist_ok=True)
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        pitch, dur = pattern
        if '.' in pitch or pitch.isdigit():
            notes_in_chord = pitch.split('.')
            notes_ = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes_:
                n.storedInstrument = instrument.Piano()
            chord_obj = chord.Chord(notes_)
            chord_obj.offset = offset
            chord_obj.quarterLength = dur
            output_notes.append(chord_obj)
        else:
            n = note.Note(pitch)
            n.offset = offset
            n.quarterLength = dur
            n.storedInstrument = instrument.Piano()
            output_notes.append(n)
        offset += dur
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="output/generated_music.mid")

def generate_music():
    with open("data/notes_dual", "rb") as f:
        notes = pickle.load(f)

    pitches = sorted(set(n[0] for n in notes))
    durations = sorted(set(n[1] for n in notes))
    pitch_to_int = {p: i for i, p in enumerate(pitches)}
    dur_to_int = {d: i for i, d in enumerate(durations)}
    int_to_pitch = {i: p for p, i in pitch_to_int.items()}
    int_to_dur = {i: d for d, i in dur_to_int.items()}

    sequence_length = 100
    pitch_seq = [pitch_to_int[n[0]] for n in notes[:sequence_length]]
    dur_seq = [dur_to_int[n[1]] for n in notes[:sequence_length]]
    pattern = [[p / len(pitches), d / len(durations)] for p, d in zip(pitch_seq, dur_seq)]

    model = create_network((sequence_length, 2), len(pitches))
    model.load_weights("weights/weights-best.keras")

    output = []
    for _ in range(300):
        input_seq = np.reshape(pattern, (1, len(pattern), 2))
        prediction = model.predict(input_seq, verbose=0)
        pitch_idx = np.argmax(prediction)
        pitch = int_to_pitch[pitch_idx]
        dur = 0.5
        output.append((pitch, dur))
        pattern.append([pitch_idx / len(pitches), dur_to_int[dur] / len(durations)])
        pattern = pattern[1:]

    create_midi(output)

if __name__ == "__main__":
    generate_music()
