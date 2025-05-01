import streamlit as st
import subprocess
import os

st.set_page_config(page_title="🎼 LSTM Music Generator", layout="centered")
st.title("🎼 LSTM-Based Music Generator")

st.markdown("""
Welcome to the LSTM Music Generator 🎹  
This tool allows you to generate new piano music using a pre-trained LSTM model.  
Make sure you have already **trained the model** and saved weights at `weights/weights-best.keras`.  
Then, click the button below to generate music.
""")

# Generate button
if st.button("🎶 Generate Music"):
    with st.spinner("Generating music... please wait."):
        result = subprocess.run(["python", "generate_music.py"], capture_output=True, text=True)

    if result.returncode == 0:
        st.success("Music generated successfully! 🎵")
        midi_path = "output/generated_music.mid"
        if os.path.exists(midi_path):
            with open(midi_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Generated MIDI File",
                    data=f,
                    file_name="generated_music.mid",
                    mime="audio/midi"
                )
        else:
            st.warning("MIDI file not found. Please check the output directory.")
    else:
        st.error("An error occurred during generation.")
        st.text(result.stderr)
