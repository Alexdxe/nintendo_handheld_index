#List of Pages for Navigation:
import streamlit as st
import pandas as pd

st.logo("photos/nintendo.png", size= 'large')

pages = [
    st.Page("Home.py", title="Home", icon="🏡"),
    st.Page("pages/Gameboy.py", title="Game Boy"),
    st.Page("pages/GBA.py", title="GBA"),
    st.Page("pages/DS.py", title="DS"),
    st.Page("pages/three_DS.py", title="3DS"),
    st.Page("pages/Switch.py", title="Switch"),
    st.Page("pages/Switch2.py", title="Switch 2"),
    st.Page("pages/testing.py", title="Testing")
]

# Run Navigation:
pg = st.navigation(pages)
pg.run()