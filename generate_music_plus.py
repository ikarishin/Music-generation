import os
import pickle
import numpy as np
from music21 import note, chord, stream, instrument
from keras.models import load_model
from train import create_network

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(preds)
    return np.random.choice(len(preds), p=preds)

def generate_music_plus():
    with open("data/notes_dual.pkl", "rb") as f:
        notes = pickle.load(f)

    pitches = sorted(set(n[0] for n in notes))
    durations = sorted(set(n[1] for n in notes))
    pitch_to_int = {p: i for i, p in enumerate(pitches)}
    dur_to_int = {d: i for i, d in enumerate(durations)}
    int_to_pitch = {i: p for p, i in pitch_to_int.items()}
    int_to_dur = {i: d for d, i in dur_to_int.items()}

    sequence_length = 100
    if len(notes) <= sequence_length:
        raise ValueError("Too few notes for generation.")
    pitch_seq = [pitch_to_int[n[0]] for n in notes[:sequence_length]]
    dur_seq = [dur_to_int[n[1]] for n in notes[:sequence_length]]
    pattern = [[p / len(pitches), d / len(durations)] for p, d in zip(pitch_seq, dur_seq)]

    model_pitch = create_network(len(pitches), (sequence_length, 2))
    model_dur = create_network(len(durations), (sequence_length, 2))
    model_pitch.load_weights("weights/weights-pitch.keras")
    model_dur.load_weights("weights/weights-duration.keras")

    allowed_durations = [0.25, 0.5, 1.0, 1.5, 2.0]
    allowed_dur_indices = [i for i, d in int_to_dur.items() if d in allowed_durations]

    output = []
    for _ in range(300):
        input_seq = np.reshape(pattern, (1, len(pattern), 2))

        pitch_prediction = model_pitch.predict(input_seq, verbose=0)
        dur_prediction = model_dur.predict(input_seq, verbose=0)

        pitch_idx = sample_with_temperature(pitch_prediction[0], temperature=1.2)

        raw_probs = dur_prediction[0].astype("float64")
        mask = np.zeros_like(raw_probs)
        mask[allowed_dur_indices] = 1
        masked_probs = raw_probs * mask

        if masked_probs.sum() == 0:
            dur_idx = np.random.choice(allowed_dur_indices)
        else:
            masked_probs /= masked_probs.sum()
            dur_idx = np.random.choice(len(masked_probs), p=masked_probs)

        pitch = int_to_pitch.get(pitch_idx, int_to_pitch[0])
        dur = int_to_dur.get(dur_idx, 0.5)

        output.append((pitch, dur))
        pattern.append([pitch_idx / len(pitches), dur_idx / len(durations)])
        pattern = pattern[1:]

    os.makedirs("output", exist_ok=True)
    offset = 0
    output_notes = []

    for pitch, dur in output:
        if '.' in pitch or pitch.isdigit():
            chord_notes = [note.Note(int(n)) for n in pitch.split('.')]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            new_chord.quarterLength = dur
            output_notes.append(new_chord)
        else:
            n = note.Note(pitch)
            n.offset = offset
            n.quarterLength = dur
            n.storedInstrument = instrument.Piano()
            output_notes.append(n)
        offset += dur

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="output/generated_music.mid")
    print("✅ Music generated and saved to output/generated_music.mid")
