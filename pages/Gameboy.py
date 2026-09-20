#Packages
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import geopandas as gpd


#Datasets:
game_df = pd.read_csv('data/game_list.csv')
console_time_gdf = gpd.read_file('data/console_by_year.geojson')
console_total_gdf = gpd.read_file('data/console_total_sales.geojson')
software_time_gdf = gpd.read_file('data/software_by_year.geojson')
software_total_gdf = gpd.read_file('data/software_total_sales.geojson')

######################################################################################################

#FROM EDA:
#Coverting to datetime:
game_df['release_date'] = pd.to_datetime(game_df['release_date'], format = 'mixed')


#Making a seperate df for each console: 
GB_df = game_df[game_df['console'] == 'GB'].copy()
GBA_df = game_df[game_df['console'] == 'GBA'].copy()
DS_df = game_df[game_df['console'] == 'DS'].copy()
three_DS_df = game_df[game_df['console'] == '3DS'].copy()
switch_df = game_df[game_df['console'] == 'NS'].copy()

#Console Time Series DF:
GBA_sales_gdf = console_time_gdf[console_time_gdf['Console'] == 'GBA'].copy()
DS_sales_gdf = console_time_gdf[console_time_gdf['Console'] == 'DS'].copy()
three_DS_sales_gdf = console_time_gdf[console_time_gdf['Console'] == '3DS'].copy()
switch_sales_gdf = console_time_gdf[console_time_gdf['Console'] == 'Switch'].copy()

#Console Total DF:
GBA_total_sales_gdf = console_total_gdf[console_total_gdf['Console'] == 'GBA'].copy()
DS_total_sales_gdf = console_total_gdf[console_total_gdf['Console'] == 'DS'].copy()
three_DS_total_sales_gdf = console_total_gdf[console_total_gdf['Console'] == '3DS'].copy()
switch_total_sales_gdf = console_total_gdf[console_total_gdf['Console'] == 'Switch'].copy()

#Software Time Series DF:
GBA_game_sales_gdf = software_time_gdf[software_time_gdf['Console'] == 'GBA'].copy()
DS_game_sales_gdf = software_time_gdf[software_time_gdf['Console'] == 'DS'].copy()
three_DS_game_sales_gdf = software_time_gdf[software_time_gdf['Console'] == '3DS'].copy()
switch_game_sales_gdf = software_time_gdf[software_time_gdf['Console'] == 'Switch'].copy()

######################################################################################################

#Best Selling Games on GB: 
GB_df['million_units_sold'] = GB_df['units_sold'] / 1000000
GB_df = GB_df.sort_values(by = 'million_units_sold', ascending=False)

GB_top_10_selling = {
 'name' : GB_df['game'].head(10).reset_index(drop=True),
 'Units Sold (millions)' : GB_df['million_units_sold'].head(10).reset_index(drop=True)
}
GB_top_10_selling_df  = pd.DataFrame(GB_top_10_selling)

GB_top_10_selling_df = GB_top_10_selling_df.rename(columns={'name': 'Game Title', 'Units Sold (millions)': 'Units Sold (millions)'})
GB_top_10_selling_df = GB_top_10_selling_df.reset_index(drop=True)
######################################################################################################

#Configure Page Settings:
st.set_page_config(layout="centered")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap');

/* 1. GLOBAL FONT */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/* 2. UNIFIED TEXTBOX COLORS */
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

/* 3. SIDEBAR STYLING */
[data-testid="stSidebar"] {
    background-color: #3B181A !important;
    padding: 20px;
}

[data-testid="stSidebar"] div {
    color: white !important;
    font-family: sans-serif;
}

[data-testid="stSidebar"] a {
    color: #ffcc00 !important;
    text-decoration: none;
    font-weight: bold;
}

/* 4. BACKGROUND COLORS */
.stApp {
    background-color: #71906e !important;
    color: #FFFFFF !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #4b5563 !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #8D8D8D !important; /* Light grey cell background */
    color: #111827 !important;             /* Dark text inside table cells */
}

/* 6. HIDE THE INDEX COLUMN */
[data-testid="stTable"] table th:first-child,
[data-testid="stTable"] table td:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

######################################################################################################


# -------------------------------------- Title: -------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

        /* Target only this specific class to override the global !important */
        h1.retro-header, h1.retro-header a {
            font-family: 'Press Start 2P', cursive !important;
            text-align: center;
        }
    </style>

    <h1 class="retro-header">
        <a href="https://nintendo.fandom.com/wiki/Game_Boy" target="_blank" style="color: #a91101; text-decoration: none;">Game Boy</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
st.write("")

#Dividing: 
left, center, right = st.columns([3, 0.10, 2])

# -------------------------------------- Summary: -------------------------------------------------
with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("The Game Boy is one of the most iconic gaming consoles in the world. " \
        "It is defined by making the most of a bad situation. With lackluster hardware that was dated at release, the Game Boy was still home to some of gaming’s most iconic franchises. " \
        "Series such as " \
        '<a href="https://en.wikipedia.org/wiki/Pok%C3%A9mon" target="_blank" style="color: #a91101; text-decoration: none;"><strong>Pokémon</strong></a>, '
        '<a href="https://en.wikipedia.org/wiki/Tetris" target="_blank" style="color: #a91101; text-decoration: none;"><strong>Tetris</strong></a>, '
        "and "
        '<a href="https://en.wikipedia.org/wiki/Kirby_(series)" target="_blank" style="color: #a91101; text-decoration: none;"><strong>Kirby</strong></a>,'
        " kicked off their adventures on the grey brick.", unsafe_allow_html=True)
    
    st.markdown(
        "The Game Boy personifies what Nintendo believes in. While faced with competition from superior hardware such as the "
        '<a href="https://sega.fandom.com/wiki/Game_Gear" target="_blank" style="color: #a91101; text-decoration: none;"><strong>Sega Game Gear</strong></a> '
        "and "
        '<a href="https://en.wikipedia.org/wiki/Atari_Lynx" target="_blank" style="color: #a91101; text-decoration: none;"><strong>Atari Lynx</strong></a>, '
        "both fell short of the Game Boy due to its incredible game library and affordable price. In an era of rapid "
        "technological development, the Game Boy stood out as an anomaly, emphasizing the importance of simplicity and quality.",
        unsafe_allow_html=True)

    
    st.divider()

    #Games:
    st.markdown(
        """
        <style>
            div[data-testid="stImage"] {
                display: flex;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    games = Image.open("photos/gb_carts.jpeg")
    st.image(games, width=350)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the Game Boy Cartridges</i></h6>", unsafe_allow_html=True)

# -------------------------------------- Photo/Facts: -------------------------------------------------


with right:
    #Logo: 
    #Photo of Logo: 
    logo = Image.open("photos/gameboy_logo.png")
    st.image(logo, width=500)
    #Photo:
    #Photo of Consoles: 
    console = Image.open("photos/gb.jpeg")
    st.image(console, width=500)
    #Facts: 
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 04/21/1989</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price: </u>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> $90 (USD)</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>🥉 118.69 million </h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Unit Variants:</u></h5>", unsafe_allow_html=True)
    

    st.markdown(
    """
    <div style="text-align: center;">
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Pocket" target="_blank" style="color: #a91101; text-decoration: none;">Game Boy Pocket</a></strong> (1996)<br>
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Light" target="_blank" style="color: #a91101; text-decoration: none;">Game Boy Light</a></strong> (1998)<br>
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Color" target="_blank" style="color: #a91101; text-decoration: none;">Game Boy Color</a></strong> (1998)
    </div>
    """,
    unsafe_allow_html=True)





st.divider()

# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):

    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 🥉 501.11 million</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> ~4 Games per conole</h5>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h4 style='text-align: center;'> Top 10 Best Selling Games:</h4>", unsafe_allow_html=True)

    #Styling the DF based on GB Colors: 
    styled_GB_df = (
        GB_top_10_selling_df.style
        .set_properties(**{
            'background-color': '#8D8D8D',  # Table cells background
            'color': '#111827'               # Text color for cells
        })
        .set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#555555'), # Darker grey tint for headers
                    ('color', '#ffffff'),
                    ('font-weight', 'bold')
                ]
            }
        ])
        .format({'Units Sold (millions)': '{:.2f}'})
        .hide()
    )

    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([3, 0.5, 1.5])

    with left:
        st.table(styled_GB_df)

    with right:
        st.markdown("""
                <div style="text-align: center;">
                <u>Notes:</u>
                </div>
        
                - Sales Data was collected from VGChartz.
                - Scores on Games are limited due to lack of game reviews prior to 2000.
                """, unsafe_allow_html=True)
        
        st.write("")
        tetris = Image.open("photos/tetris.png")
        st.image(tetris, width=500)
        st.markdown("<h6 style='text-align: center;'><i>Image of Tetris Game play<i/>", unsafe_allow_html=True)
        st.write("")

        

st.divider()


# ------------------------------------------ Graphs: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Graphs:</h3>", unsafe_allow_html=True)
    #Finding the Yearly Counts:
    GB_year_counts = GB_df['release_date'].dt.year.value_counts().sort_values(ascending = True).reset_index()
    GB_year_counts.columns = ['Release Date', 'Game Count']
    GB_year_counts = GB_year_counts.sort_values(by = 'Release Date', ascending = True)

    GB_game_ts = px.line( 
        GB_year_counts,
        x = 'Release Date',
        y = 'Game Count',
        title = 'Gameboy Games by Release Year:',
        markers = True,
        template = 'ggplot2',
        color_discrete_sequence= ['#88b49b']
    )

    #Update Line Width:
    GB_game_ts.update_traces(
    line=dict(width=5),  # Set line thickness (default is usually 2)
    marker=dict(size=8)  # Optional: scale markers to match the thicker line
    )

    
    # Set the background colors and adjust text color for readability
    GB_game_ts.update_layout(
    paper_bgcolor='#8D8D8D',
    plot_bgcolor='#9B9B9B',
    
    # Title font settings
    title=dict(
        text='Gameboy Games by Release Year:',
        font=dict(size=25, color='white')
    ),
    
    # X-Axis font settings
    xaxis=dict(
        title=dict(font=dict(size=16, color='white')), # Axis label ('Release Date')
        tickfont=dict(size=17, color='white')          # Numbers/dates along the axis
    ),
    
    # Y-Axis font settings
    yaxis=dict(
        title=dict(font=dict(size=16, color='white')), # Axis label ('Game Count')
        tickfont=dict(size=17, color='white')          # Numbers along the axis
    )
)
    st.plotly_chart(GB_game_ts, use_container_width=True)

st.divider()


# ------------------------------------------ Fun Facts: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>Game Boy Fun Facts:</h3>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    1. It's the 
    <a href="https://en.wikipedia.org/wiki/List_of_best-selling_game_consoles" target="_blank" style="color: #a91101; text-decoration: none;">4th best selling console</a>
    of all time. 
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    2. Despite having hundreds of exclusive games, the 
    <a href="https://nintendo.fandom.com/wiki/Game_Boy_Color" target="_blank" style="color: #a91101; text-decoration: none;">Gameboy Color</a>
     is often not considered it's own console generation.
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    3. It was Nintendo’s longest-running console in production (14 years)
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    4. Despite being the Game Boy's best selling title, 
    <a href="https://tetris.wiki/Tetris_(Game_Boy)" target="_blank" style="color: #a91101; text-decoration: none;">Tetris</a>
     is not a Nintendo owned intellectual property.
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'> 
    5. The hanheld infamously survived a
    <a href="https://www.esquire.com/lifestyle/a27183316/nintendo-game-boy-survived-gulf-war/" target="_blank" style="color: #a91101; text-decoration: none;">bomb blast</a>
    during a Gulf War barracks bombing.
    </h5>""", unsafe_allow_html=True)
st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/GBA.py")
