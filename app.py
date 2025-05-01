import streamlit as st
import subprocess
import os
from generate_music import generate_music

st.set_page_config(page_title="🎼 LSTM Music Generator", layout="centered")
st.title("🎼 LSTM-Based Music Generator")

st.markdown("""
Welcome to the LSTM Music Generator 🎹  
This tool lets you generate new piano music using a pre-trained LSTM model.  
Make sure the file `weights/weights-best.keras` exists (i.e. you've trained the model).  
Then click the button below to generate music.
""")

# Generate button
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
                st.warning("MIDI file was not found after generation.")
        except Exception as e:
            st.error("An error occurred during generation.")
            st.text(str(e))
