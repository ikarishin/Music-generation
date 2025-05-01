import streamlit as st
from generate_music_plus import generate_music_plus
import shutil
import os

st.write("📁 Files in /data:", os.listdir("data"))

st.set_page_config(page_title="🎵 AI Music Generator", layout="centered")

st.title("🎼 AI Music Generator")
st.markdown("Generate MIDI music using an LSTM-based AI model.")

num = st.number_input("How many songs do you want to generate?", min_value=1, max_value=10, value=1, step=1)

if st.button("Generate 🎶"):
    st.info("Please wait while the music is being generated...")

    for i in range(num):
        filename = f"music_{i+1}.mid"
        generate_music_plus()  # Always generates output/generated_music.mid

        os.makedirs("output_multi", exist_ok=True)
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
