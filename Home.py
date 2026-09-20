import pandas as pd
import streamlit as st
from PIL import Image

#Configure Page Settings:
st.set_page_config(layout="centered")


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap');

/*1. GLOBAL FONT*/
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/*2. UNIFIED TEXTBOX COLORS */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="datepicker"] > div {
    background-color: #ffffff !important; 
    border-radius: 8px !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    box-shadow: none !important;
}

input, textarea, div[data-baseweb="select"] * {
    color: black !important;
}

/*3. SIDEBAR STYLING */
[data-testid="stSidebar"] {
    background-color: #3B181A;
    padding: 20px;
}

[data-testid="stSidebar"] div {
    color: white;
    font-family: sans-serif;
}

[data-testid="stSidebar"] a {
    color: #ffcc00 !important;
    text-decoration: none;
    font-weight: bold;
}

/*4. BACKGROUND COLORS */
.stApp {
    background-color: #5E3032;
    color: #FFFFFF;
}

/*5. BUTTON STYLING */
/* Change background color for all standard Streamlit buttons */
.stButton > button {
    background-color: #ba3c3c !important;
    color: #FFFFFF !important;
    border: 1px solid #7A222B !important;
}

/* Change color when hovering over the button */
.stButton > button:hover {
    background-color: #83A1CD !important;
    border-color: #ffcc00 !important;
}
</style>

""",
    unsafe_allow_html=True,
)


#Title: 
st.markdown("<h1 style='text-align: center;'>Nintendo's Handheld History</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'> <b>Written by:</b> Alexander Cai</h5>", unsafe_allow_html=True)
st.write("")

#Photo of Consoles: 
photo = Image.open("photos/handhelds.jpg")

#Centering the Image: 
left, center, right = st.columns([1, 5, 1])
with center:
    st.image(photo, width=750)

#Caption:
st.markdown("<h6 style='text-align: center;'> <i>Photo Credit: Lost in Cult's Handheld History</i></h6>", unsafe_allow_html=True)

st.divider()


#Project Description: 
with st.container(border=True): 
    st.markdown("<h3 style='text-align: center;'><u>Description</u></h3>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<strong>Gaming is one of the most popular and influential mediums in the world.</strong> " \
    "It’s a unique form of media that immerses users through interactive challenges and combines various other art forms (music, cinematography, etc) " \
    "to craft a one-of-a-kind experience. From hardcore games such as " \
    '<a href="https://en.wikipedia.org/wiki/Dark_Souls" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Dark Souls</strong></a>, '
    " to casual experiences like " \
    '<a href="https://pokemongo.fandom.com/wiki/Pok%C3%A9mon_GO" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Pokémon Go</strong></a> '
    ", everyone has once been addicted to a video game. " \
    "However, although people's passions for gaming differ, one thing remains consistent: <strong>almost everyone can share an experience with a Nintendo console.</strong> ", unsafe_allow_html=True)

    st.markdown("With over 800 million consoles sold worldwide and a dozen famous franchises, <strong>Nintendo is gaming's most iconic company.</strong> " \
    "Most notably, Nintendo has always been the flag bearer for handheld gaming, a market aiming to bring larger-than-life experiences on the go. " \
    "With iconic devices such as the Game Boy, DS, and Switch, <strong>Nintendo’s passion bleeds through these pieces of hardware</strong>, " \
    "illustrating why people love collecting these handhelds.", unsafe_allow_html=True)

    st.markdown("<strong>This project aims to analyze the rich history of Nintendo’s handhelds, observing sales numbers, "
    "iconic games, and answering hypotheses using various statistical methods. <strong>The goal is to learn more about Nintendo through the lens of " \
    "a Data Scientist.</strong>", unsafe_allow_html=True)
 
    #Photo of Consoles: 
    photo = Image.open("photos/n_handhelds.jpg")

    #Centering the Image: 
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.image(photo, width=500)

    #Caption:
    st.markdown("<h6 style='text-align: center;'> <i>Image of Nintendo's Handheld Consoles </i></h6>", unsafe_allow_html=True)


#Overview
with st.container(border=True): 
    st.markdown("<h3 style='text-align: center;'><u>Overview</u></h3>", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'> 
    <strong>This project aims to cover the following topics:</strong>""", unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stPageLink"] {
        background-color: #E90000;
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 2px;
    }
    </style>
    """, unsafe_allow_html=True)




    left, left_center, right_center, right = st.columns([0.5, 2, 1.5, 0.5])
    with left_center:
        st.page_link("pages/Gameboy.py", label="1\) Game Boy")
        st.page_link("pages/GBA.py", label="2\) Game Boy Advance")
        st.page_link("pages/DS.py", label="3\) Nintendo DS")
        st.page_link("pages/three_DS.py", label="4\) Nintendo 3DS")
        st.page_link("pages/Switch.py", label="5\) Nintendo Switch")
        st.page_link("pages/Switch2.py", label="6\) Nintendo Switch 2")
        st.page_link("pages/testing.py", label="7\) General Testing")

    with right_center: 
        #Console:
        console = Image.open("photos/gb1.jpeg") 
        st.image(console, width=300)

        st.markdown("<h6 style='text-align: center;'> <i>Image the Game Boy</i></h6>", unsafe_allow_html=True)


    st.write("")
    st.markdown("While there a few consoles missing (" \
    '<a href="https://game-watch.fandom.com/wiki/Game_and_Watch_Wiki" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Game and Watch</strong></a>'
    ") and information for some scarce (Game Boy and Switch 2), this is mostly due to the difficulties of finding/sourcing the data." \
    " Therefore, for these reasons:", unsafe_allow_html=True)

    st.markdown("""
    <h6 style='text-align: center; line-height: 1.75;'>
    1) There is no Game & Watch page due to lack of overall data. <br>
    2) The Gameboy page lacks most graphs and a hypothesis test. <br>
    3) The Switch 2 page lacks all visualizations. <br>
    </h6>
    """, unsafe_allow_html=True)

    



#Personal Attachment:
with st.expander("**Personal Attachment**"):
    st.markdown("""I am a massive Nintendo fan. <strong>From the day I received a DS for Christmas in 2010 to buying my first-ever console (Switch 2) in 2026, it’s been a company I have loved throughout my life.</strong>
    """,unsafe_allow_html=True)

    st.markdown("""It’s a company that has given me so many memorable moments and brought me endless joy while blasting through their iconic library. A company with a rich (and interesting) history full of unique trials and triumphs
        that has shaped the gaming industry into what it is today. A company with a litany of problems and annoying quirks that many roll their eyes at in frustration, but that clearly define who Nintendo is at its core. It’s a company
        that is as interesting as it is controversial, but I cannot ignore the feeling that whenever I play one of Nintendo’s finest games, I lose myself in the medium like no other.""")

    #Console:
    screen = Image.open("photos/collection.jpg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(screen, width=300)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the Consoles I currently own.</i></h6>", unsafe_allow_html=True)


    st.markdown("Given that most of my experience with Nintendo has been through their handheld devices, I wanted to learn about its history through the eyes of a statistician. <strong>I wanted to understand what made a handheld a success, " \
    "analyze sales numbers across all regions, and even predict the sales of both Switch consoles.</strong> It’s this burning passion to both learn about Nintendo through statistical application that birthed this project.",unsafe_allow_html=True)



#Data Source:
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>Data Sources</u></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>This project was made possible through two different sources:</h5>", unsafe_allow_html=True)
    
    st.markdown("<h6 style='text-align: center;'><a href='https://www.vgchartz.com/' target='_blank'><b>VGChartz:</b></a> For information on individual game sales and scores.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'><a href='https://www.nintendo.co.jp/ir/en/library/index.html' target='_blank'><b>Nintendo’s IR Library:</b></a> For information on cumulative sales by region and time.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'><a href='https://nintendo.fandom.com/wiki/Nintendo_Wiki' target='_blank'><b>Nintendo Wiki:</b></a> For fun facts and specific sales numbers.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'><b>Others:</b> For other more niche information and inspiration.</h6>", unsafe_allow_html=True)
    st.write("")


    #Photo of Consoles: 
    photo = Image.open("photos/cartridges.png")

    #Centering the Image: 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(photo, width=350)

    #Caption:
    st.markdown("<h6 style='text-align: center;'><i>Image of N64 Game Cartridges:</i></h6>", unsafe_allow_html=True)
    



#Goals: 
st.markdown("<h3 style='text-align: center;'><u>Goals</u></h3>", unsafe_allow_html=True)
st.markdown(""" <h5 style='text-align: center; line-height: 2;'> 
    1) Summarize each handheld's history through facts and details. <br>
    2) Visualize software/hardware sales by variant, region and time. <br>
    3) Answer hypothesis questions exclusive and across all handhelds. <br>
    4) Deepen my knowledge of Data Science through application.
    """,unsafe_allow_html=True)

#Next Page:
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/Gameboy.py")