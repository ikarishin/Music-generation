import streamlit as st
from generate_music import generate_music
import shutil
import os

st.set_page_config(page_title="🎵 AI Music Generator", layout="centered")
st.title("🎼 AI Music Generator")
st.markdown("Generate MIDI music using an LSTM-based model.")

num = st.number_input("How many songs would you like to generate?", min_value=1, max_value=10, value=1, step=1)

if st.button("Generate 🎶"):
    st.info("Generating music... Please wait.")
    os.makedirs("output_multi", exist_ok=True)

    for i in range(num):
        generate_music()
        filename = f"music_{i+1}.mid"
        shutil.copy("output/generated_music.mid", f"output_multi/{filename}")

        st.success(f"✅ Generated: {filename}")
        with open(f"output_multi/{filename}", "rb") as f:
            st.download_button(
                label=f"Download {filename}",
                data=f,
                file_name=filename,
                mime="audio/midi"
            )

    st.balloons()