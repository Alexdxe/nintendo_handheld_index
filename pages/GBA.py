#Packages


#General Packages: 
import pandas as pd
import numpy as np
from scipy import stats

#Statistical Testing:
import statsmodels
import pingouin as pg
import sklearn
#import scikit_posthocs as sp

#Machine Learning:
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Time Series: 
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Web-Dev:
import streamlit as st
from PIL import Image

#Visuals:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px

#Geospatial Packages:
import geopandas as gpd
import folium
from folium import Choropleth
from streamlit_folium import st_folium

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
#Mean Scores for Games:
#Finding the mean for each console (top 100 games):
GBA_mean_score = round(GBA_df['critic_score'].dropna().head(100).mean(), 2)
DS_mean_score = round(DS_df['critic_score'].dropna().head(100).mean(), 2)
three_DS_mean_score = round(three_DS_df['critic_score'].dropna().head(100).mean(), 2)
switch_mean_score = round(switch_df['critic_score'].dropna().head(100).mean(), 2)

mean_scores = {'GBA Mean Score': GBA_mean_score,
               'DS Mean Score':DS_mean_score, 
               '3DS Mean Score' : three_DS_mean_score, 
               'Switch Mean Score': switch_mean_score}
######################################################################################################


#Best Selling Games:
#Dividing by 1,000,000:
GB_df['million_units_sold'] = GB_df['units_sold'] / 1000000
GBA_df['million_units_sold'] = GBA_df['units_sold'] / 1000000
DS_df['million_units_sold'] = DS_df['units_sold'] / 1000000
three_DS_df['million_units_sold'] = three_DS_df['units_sold'] / 1000000
switch_df['million_units_sold'] = switch_df['units_sold'] / 1000000

#Sorting by Units Sold:
GB_df = GB_df.sort_values(by = 'million_units_sold', ascending=False)
GBA_df = GBA_df.sort_values(by = 'million_units_sold', ascending=False)
DS_df = DS_df.sort_values(by = 'million_units_sold', ascending=False)
three_DS_df = three_DS_df.sort_values(by = 'million_units_sold', ascending=False)
switch_df = switch_df.sort_values(by = 'million_units_sold', ascending=False)



#GB: 
GB_top_10_selling = {
 'name' : GB_df['game'].head(10).reset_index(drop=True),
 'Units Sold (millions)' : GB_df['million_units_sold'].head(10).reset_index(drop=True)
}

GB_top_10_selling_df  = pd.DataFrame(GB_top_10_selling)


#GBA: 
GBA_top_10_selling = {
'Name' : GBA_df['game'].head(10).reset_index(drop=True),
'Units Sold (millions)' : GBA_df['million_units_sold'].head(10).reset_index(drop=True)
}
GBA_top_10_selling_df = pd.DataFrame(GBA_top_10_selling)


#DS:
DS_top_10_selling = {
'name' : DS_df['game'].head(10).reset_index(drop=True),
'Units Sold (millions)' : DS_df['million_units_sold'].head(10).reset_index(drop=True)
}
DS_top_10_selling_df = pd.DataFrame(DS_top_10_selling)


#3DS: 
three_ds_top_10_selling = {
'Name' : three_DS_df['game'].head(10).reset_index(drop=True),
'Units Sold (millions)' : three_DS_df['million_units_sold'].head(10).reset_index(drop=True)
}
three_ds_top_10_selling_df = pd.DataFrame(three_ds_top_10_selling)


#Switch:
switch_top_10_selling = {
'Name' : switch_df['game'].head(10).reset_index(drop=True),
'Units Sold (millions)' : switch_df['million_units_sold'].head(10).reset_index(drop=True)
}
switch_top_10_selling_df = pd.DataFrame(switch_top_10_selling)

######################################################################################################
#Finding the top rated games: 
GBA_top_10_rated = (GBA_df.sort_values(by='critic_score', ascending=False).head(10).reset_index(drop=True))
DS_top_10_rated = (DS_df.sort_values(by='critic_score', ascending=False).head(10).reset_index(drop=True))
three_ds_top_10_rated = (three_DS_df.sort_values(by='critic_score', ascending=False).head(10).reset_index(drop=True))
switch_top_10_rated = (switch_df.sort_values(by='critic_score', ascending=False).head(10).reset_index(drop=True))


#GBA:
GBA_top_10_rated = {
'Name' : GBA_top_10_rated['game'].head(10).reset_index(drop=True),
'Critic Score' : GBA_top_10_rated['critic_score'].head(10).reset_index(drop=True)
}
GBA_top_10_rated_df = pd.DataFrame(GBA_top_10_rated)
GBA_top_10_rated_df['Critic Score'] = round(GBA_top_10_rated_df['Critic Score'], 2)

#DS:
DS_top_10_rated = {
'Name' : DS_top_10_rated['game'].head(10).reset_index(drop=True),
'Critic Score' : DS_top_10_rated['critic_score'].head(10).reset_index(drop=True)
}
DS_top_10_rated_df = pd.DataFrame(DS_top_10_rated)
DS_top_10_rated_df = round(DS_top_10_rated_df['Critic Score'], 2)

#3DS:
three_ds_top_10_rated = {
'Name' : three_ds_top_10_rated['game'].head(10).reset_index(drop=True),
'Critic Score' : three_ds_top_10_rated['critic_score'].head(10).reset_index(drop=True)
}
three_ds_top_10_rated_df = pd.DataFrame(three_ds_top_10_rated)
three_ds_top_10_rated_df['Critic Score'] = round(three_ds_top_10_rated_df['Critic Score'], 2)

#Switch:
switch_top_10_rated = {
'Name' : switch_top_10_rated['game'].head(10).reset_index(drop=True),
'Critic Score' : switch_top_10_rated['critic_score'].head(10).reset_index(drop=True)
}
switch_top_10_rated_df = pd.DataFrame(switch_top_10_rated)
switch_top_10_rated_df['Critic Score'] = round(switch_top_10_rated_df['Critic Score'], 2)
######################################################################################################
#Appending Estimated Sales for Original 3DS, DS, GBA, and GBA Micro:

#Rows for GBA:
new_rows = [
    {'index': 12, 'Console': 'GBA', 'Model': 'Micro', 'Region': 'Total', 'Total Sales': 2500000, 'geometry': None},
    {'index': 13, 'Console': 'GBA', 'Model': 'Original', 'Region': 'Total', 'Total Sales': 35440000, 'geometry': None}
]

# Append both rows at once:
GBA_total_sales_gdf = pd.concat([GBA_total_sales_gdf, pd.DataFrame(new_rows)], ignore_index=True)


#Original DS: 
DS_total_sales_gdf.loc[len(DS_total_sales_gdf)] = {'index': 28,'Console': 'DS', 'Model': 'Original','Region': 'Total','Total Sales': 18790000, 'geometry': None}

#Original 3DS:
three_DS_total_sales_gdf.loc[len(three_DS_total_sales_gdf)] = {'index': 33,'Console': '3DS', 'Model': 'Original','Region': 'Total','Total Sales': 26240000, 'geometry': None}

#Dictonary of all consoles:
consoles = {
    'GBA': GBA_total_sales_gdf, 
    'DS': DS_total_sales_gdf,
    '3DS' : three_DS_total_sales_gdf, 
    'Switch': switch_total_sales_gdf}

all_console_total = {}
for name, console in consoles.items(): 
    all_console_total[f'{name}_console_total'] = console[console['Region'] == 'Total'].reset_index().copy()
    
######################################################################################################

consoles = {
    'GBA': GBA_total_sales_gdf, 
    'DS': DS_total_sales_gdf,
    '3DS' : three_DS_total_sales_gdf, 
    'Switch': switch_total_sales_gdf}

all_region_total = {}
for name, console in consoles.items(): 
    all_region_total[f'{name}_region_total'] = console[(console['Region'] != 'Total') & (console['Model'] == 'All')].reset_index(drop = True).copy()


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
    background-color: #4C367E !important;
    color: #FFFFFF !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #435274 !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #8395c1 !important; /* Light grey cell background */
    color: #ffffff !important;             /* Dark text inside table cells */
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
import streamlit as st

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Jersey+25&display=swap');

        /* Target this specific class on h1 to override global header font rules */
        h1.jersey-header {
            font-family: 'Jersey 25', sans-serif !important;
            text-align: center;
            color: #a91101 !important;
        }
    </style>

    <h1 class="jersey-header">
        <a href="https://nintendo.fandom.com/wiki/Game_Boy_Advance" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy Advance</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
# -------------------------------------- Summary: -------------------------------------------------
#Dividing: 
left, center, right = st.columns([3, 0.10, 2])


with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("Labeled as “The Super Nintendo on the Go”, the Game Boy Advance was touted as a 32-bit powerhouse that finally featured a brighter, " \
        "more colorful screen. The GBA upgraded almost every aspect of the original Game Boy, with the addition of shoulder buttons, backwards compatibility, and a new landscape form factor. " \
        "Simply put, the GBA was a technological leap and a continuation of a winning formula.")
    
    st.markdown("Although 25 years old, its library is still one of the most treasured today. With exclusives such as " 
        '<a href="https://metroid.fandom.com/wiki/Metroid_Fusion" target="_blank" style="color: #A9A9A9; text-decoration: none;"><strong>Metroid Fusion</strong> </a> '
        "showing off the stunning sprite-work, " 
        '<a href="https://finalfantasy.fandom.com/wiki/Final_Fantasy_VI" target="_blank" style="color: #A9A9A9; text-decoration: none;"><strong>Final Fantasy VI</strong> </a> '
        "presenting a lengthy yet thrilling story, and even a " 
        '<a href="https://en.wikipedia.org/wiki/Game_Boy_Advance_Video" target="_blank" style="color: #A9A9A9; text-decoration: none;"><strong>video player</strong></a> ' 
        "that could play full-length movies. The GBA did it all and pushed the boundaries for what a gaming handheld could achieve.",
        unsafe_allow_html=True)
    st.write("")


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

    games = Image.open("photos/gba_games.jpeg") 
    st.image(games, width=350)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the GBA Surrounded by Games</i></h6>", unsafe_allow_html=True)


# -------------------------------------- Photo/Facts: -------------------------------------------------



with right: 
    logo = Image.open("photos/gba_logo.png") 
    st.image(logo, width=500)


    console = Image.open("photos/gba.jpeg")
    st.image(console, width=500)

    #Facts: 
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 03/21/2001</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price: </u>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> $100 (USD)</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 81.51 million</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Other Variants:</u></h5>", unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: center;">
    <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Advance_SP" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy Advance SP</a> (2003)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Micro" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy Micro</a> (2005)
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Backwards Compatiable:</u> ", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center;">
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy </a> &
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Color" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy Color</a>
        """, unsafe_allow_html=True)
    st.write("")


st.divider()


# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 377.42 million</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> 🥉 ~5 Games per Console</h5>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'> <u> Average Score of Games:</u> {GBA_mean_score}</h5>", unsafe_allow_html=True)
    st.write("")

    #Styling the DF based on GB Colors: 
    styled_GBA_games_df = (
        GBA_top_10_selling_df.style
        .format({'Units Sold (millions)': '{:.2f}'})
        .hide()

    )
    #Styling the DF based on GB Colors: 
    styled_GBA_scores_df = (
        GBA_top_10_rated_df.style
        .format({'Critic Score': '{:.2f}'})
        .hide()
    )


    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([3, 0.5, 3])

    with left:
        st.markdown("<h5 style='text-align: center;'> Top 10 Best Selling Games:</h5>", unsafe_allow_html=True)
        st.table(styled_GBA_games_df)

    with right:
        st.markdown("<h5 style='text-align: center;'> Top 10 Highest Rated Games:</h5>", unsafe_allow_html=True)
        st.table(styled_GBA_scores_df)
st.divider()


# ------------------------------------------ Graphs: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Graphs:</b></h3>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Console Sales By Variant",  "Console Sales by Region", "Software by Time", 'Console Sales by Time and Region'])
    with tab1:
        #Setting an Order:
        gba_order = ['Original', 'SP', 'Micro', 'All']

        #Creating the Barplot:
        gba_total_bar = px.bar(
            all_console_total['GBA_console_total'],
            x = 'Model',
            y = 'Total Sales',
            title = 'Sales of All GBA Consoles',
            template = 'ggplot2',
            color_discrete_sequence = ['#0047AB'])

        #Updating the order by chronological order of release:
        gba_total_bar.update_xaxes(categoryorder='array', categoryarray = gba_order)

        # Set the background colors and adjust text color for readability
        gba_total_bar.update_layout(
        paper_bgcolor='#AEC6CF',
        plot_bgcolor='#e5e5e5',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=15,
        yaxis_tickfont_size=15,
        
        # Title font settings
        title=dict(
            text='Gameboy Advance Sales by Variant:',
            font=dict(size=25, color='white')
        ),
        
    )
        st.plotly_chart(gba_total_bar, use_container_width=True)




        
    with tab2: 
        st.markdown("<h4 style='text-align: center;'><b>Total GBA Sales By Region (By Millions)</b></h4>", unsafe_allow_html=True)
        sales_in_millions = all_region_total['GBA_region_total']['Total Sales'] / 1_000_000
        m_1 = folium.Map(location=[20, 0], zoom_start=1, min_zoom=1,max_zoom=10,tiles="OpenStreetMap")

        c_map = Choropleth(geo_data=all_region_total['GBA_region_total'].__geo_interface__, 
                data=sales_in_millions, 
                key_on="feature.id", 
                fill_color='YlGnBu', 
                bins=[0, 10, 20, 30, 40, 50]
                ).add_to(m_1)

        folium.GeoJson(
            all_region_total['GBA_region_total'].__geo_interface__,
            style_function=lambda x: {'fillColor': '#transparent', 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=[ 'Region', 'Total Sales'],  # Pulls the data column
                aliases=['Region:', 'Cumulative Sales:'],    # The label shown next to the value
                localize=True
            )
        
        ).add_to(m_1)

        # 4. Fit the map bounds automatically to your GeoJSON features
        m_1.fit_bounds(m_1.get_bounds())

        st_folium(m_1, use_container_width=True, height=400)

        st.markdown("<h6 style='text-align: center;'><i>Note: Sales only recorded for US/Japan/Other</i></h6>", unsafe_allow_html=True)





    with tab3:
        #Finding the Yearly Counts:
        GBA_year_counts = GBA_df['release_date'].dt.year.value_counts().sort_values(ascending = True).reset_index()
        GBA_year_counts.columns = ['Release Date', 'Game Count']
        GBA_year_counts = GBA_year_counts.sort_values(by = 'Release Date', ascending = True)

        GBA_game_ts = px.line( 
            GBA_year_counts,
            x = 'Release Date',
            y = 'Game Count',
            title = 'GBA Games by Release Year (of Top 250): ',
            markers = True,
            template = 'ggplot2',
            color_discrete_sequence = ['#485CC7']
        )
     #Update Line Width:
        GBA_game_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
        
        # Set the background colors and adjust text color for readability
        GBA_game_ts.update_layout(
        paper_bgcolor='#AEC6CF',
        plot_bgcolor='#e5e5e5',
        
        # Title font settings
        title=dict(
            text='Gameboy Advance Games by Release Year:',
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
        st.plotly_chart(GBA_game_ts, use_container_width=True)





    with tab4: 
        GBA_console_ts = px.line( 
        GBA_sales_gdf,
        x = 'Year',
        y = 'Sales',
        color = 'Region',
        title = 'GBA Sales by Time and Region:',
        markers = True,
        template = 'ggplot2',
        color_discrete_map={
            'Total': '#28282B',  
            'Japan': '#BC002D',
            'United States': '#0A3161',
            'Other' : '#006400'
        }
    )


        # Add explicit On/Off buttons
        GBA_console_ts.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0,
                    y=1.15,
                    showactive=True,
                    )])


     #Update Line Width:
        GBA_console_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
    
        
        # Set the background colors and adjust text color for readability
        GBA_console_ts.update_layout(
        paper_bgcolor='#AEC6CF',
        plot_bgcolor='#e5e5e5',
        
        # Title font settings
        title=dict(
            text='GBA Sales by Time and Region:',
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
        st.plotly_chart(GBA_console_ts, use_container_width=True)

        st.markdown("<h6 style='text-align: center;'><i>Note: Toggle Regions by clicking on the right.</i></h6>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------ Statistical Testing: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Hypothesis Question:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'><u>Can the Gameboy Micro Be Considered a Failure?</u></h4>", unsafe_allow_html=True)
    st.markdown("""
    Despite the GBA having two iconic designs, many forget there was a 3rd model. The 
    <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Micro" target="_blank" style="color: #A9A9A9; text-decoration: none;">Game Boy Micro</a><strong>
     was an expensive revision 
    of the Game Boy Advance, shrinking the device into the size of a 
    <strong><a href="https://nintendo.fandom.com/wiki/Joy-Con_Controller" target="_blank" style="color: #A9A9A9; text-decoration: none;">Nintendo Switch Joy-con</a><strong>.
    """, unsafe_allow_html=True)

    #Console:
    micro = Image.open("photos/gba_micro.jpg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(micro, width=300)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the GBA Micro</i></h6>", unsafe_allow_html=True)



    st.markdown("""
    The deivce was novel for it's time, but despite it's intriguing concept, the console was only sold for 4 years and failed to make much of an impact. Many cited it's gimmick to be both impractical and expensive, leading to low sales. 
    With this knowledge, <strong>I wanted to learn how much of a failure this console was by comparing it's sales of other Nintendo handhelds<strong>. 
    """,unsafe_allow_html=True)

    with st.expander("**The Nerdy stuff**"):
        st.markdown("""We will measure the console's success by observing in <strong>what percentile the Game Boy Micro falls in sales</strong>. By comparing it's sales against the console's peers, we can guage the hardware's relative success.
         """,unsafe_allow_html=True)

        st.markdown("""If the console was not a failure, it would be above the 25th percentile of handheld sales. <strong>This meaning the GB Micro should have sold better than 25% of Nintendo's other handheld models</strong>.
        """,unsafe_allow_html=True)

    st.divider()
    st.markdown("<h4 style='text-align: center;'><b><u>Conducting Test: </b></u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Failure</u>: Game Boy Micro's Percentile <= 25% </b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Not a Failure</u>: Game Boy Micro's Percentile > 25% </b></h5>", unsafe_allow_html=True)
    st.divider()

    #Selecting Only Totals:
    console_total = console_total_gdf[(console_total_gdf['Region'] == 'Total') & (console_total_gdf['Model'] != 'All')]
    console_total = pd.DataFrame(console_total.drop(columns=['geometry'], errors='ignore'))
    #Appending GBA, Micro, DS, and 3DS:
    new_rows = [
        {'index': 5, 'Console': 'GBA', 'Model': 'Micro', 'Region': 'Total', 'Total Sales': 2500000},
        {'index': 12, 'Console': 'GBA', 'Model': 'Original', 'Region': 'Total', 'Total Sales': 35440000},
        {'index': 15,'Console': 'DS', 'Model': 'Original','Region': 'Total','Total Sales': 18790000},
        {'index': 33,'Console': '3DS', 'Model': 'Original','Region': 'Total','Total Sales': 26240000}
    ]

    console_total_df = pd.concat([console_total, pd.DataFrame(new_rows)], ignore_index = True)
    console_total_df = console_total_df.sort_values(by = 'Total Sales', ascending= False)
    console_total_df['Total Sales (Millions)'] = console_total_df['Total Sales'] / 1_000_000
    console_total_df = console_total_df.drop(columns = ['index', 'Region', 'Total Sales'])


    st.markdown("<h5 style='text-align: center;'><b>Ranking Console's by Sales: </b></h5>", unsafe_allow_html=True)
    


    left, center, right = st.columns([1, 3, 1])

    with center:
        st.table(console_total_df)
    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Result:</u></h>", unsafe_allow_html=True)
    st.markdown("""<strong>Conclusion:</strong> The GBA Micro is tied with the New 3DS as Nintendo's worst-selling handheld variant
    (Excluding Game Boy Models). <strong>Therefore, the Game Boy Micro can be considered a financial failure.</strong> Given it's poor value ($100 at launch), release date (around the middle of the DS),
    and it's overall novelty, it's not difficult to see why the console underperformed.""", unsafe_allow_html=True)




# ------------------------------------------ Fun Facts: -----------------------------------------------------
st.write("")
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>Game Boy Advance Fun Facts:</u></h3>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    1. Is the shortest family of handhelds in production (~8 years & 6 months) </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    2. It is the only handheld without an original 2D or 3D Mario Game</h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    3. The 
    <a href="https://en.wikipedia.org/wiki/Game_Boy_Advance_Video" target="_blank" style="color: #A9A9A9; text-decoration: none;">GBA Movie Player</a>
    allowed users to watch full-length videos/movies.
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    4. The GBA is Nintendo’s 2nd 32-bit system, with their first attempt being the infamous 
    <a href="https://nintendo.fandom.com/wiki/Virtual_Boy" target="_blank" style="color: #A9A9A9; text-decoration: none;">Virtual Boy</a>.
    </h5>""", unsafe_allow_html=True)
    st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/DS.py")