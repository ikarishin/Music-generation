import streamlit as st
from generate_music_plus import generate_music  # function used to generate music

# page title
st.title("🎵 AI music generator 🎵")

# Type in: number of music to generate
num = st.number_input("Number of music to generate？", min_value=1, max_value=10, value=1)

# Press button
if st.button("Generating music 🎶"):
    st.write("Generating needs time my friend...")

    for i in range(num):
        fname = f"music_{i+1}.mid"
        generate_music(filename=fname)
        st.success(f"{i+1} music have been generated：{fname}")
        with open(f"output/{fname}", "rb") as f:
            st.download_button(
                label=f"Download music_{i+1}.midi",
                data=f,
                file_name=fname,
                mime="audio/midi"
            )

    st.balloons()
