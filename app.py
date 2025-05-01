import streamlit as st
import os
from generate_music import generate_music

st.set_page_config(page_title="🎼 LSTM Music Generator", layout="centered")
st.title("🎼 LSTM-Based Music Generator")

st.markdown("""
This tool generates piano music using a pre-trained LSTM model.  
Make sure the model is trained and weights exist in `weights/weights-pitch.keras` and `weights/weights-duration.keras`.  
Click the button below to generate music.
""")

if st.button("🎶 Generate Music"):
    with st.spinner("Generating music... please wait."):
        try:
            generate_music()
            midi_path = "output/generated_music.mid"
            if os.path.exists(midi_path):
                with open(midi_path, "rb") as f:
                    st.success("Music generated successfully! 🎵")
                    st.download_button(
                        label="⬇️ Download MIDI File",
                        data=f,
                        file_name="generated_music.mid",
                        mime="audio/midi"
                    )
            else:
                st.warning("MIDI file not found after generation.")
        except Exception as e:
            st.error("An error occurred during music generation.")
            st.text(str(e))
