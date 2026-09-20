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

#From Analysis:

#Question-Specific Datasets:
console_adjusted_df = pd.read_csv('data/Adjusted_Console_Total.csv')
ds_ts_df = pd.read_csv('data/DS_Touchscreen.csv')
three_ds_sales_df = pd.read_csv('data/3DS_Consoles.csv')
switch_sales_df = pd.read_csv('data/Switch_Sales.csv')
console_sales_df = pd.read_csv('data/Nintendo_Consoles.csv')

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
    background-color: #6B6B6B !important;
    color: #0D0D0D !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #525252 !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #808080 !important; /* Light grey cell background */
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
        @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:ital,wght@0,200..900;1,200..900&display=swap');

        /* Target this specific class on h1 to override global header font rules */
        h1.pro-header {
            font-family: 'Source Code Pro', sans-serif !important;
            text-align: center;
            color: #a91101 !important;
        }
    </style>

    <h1 class="pro-header">
        <a href="https://nintendo.fandom.com/wiki/Nintendo_DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">Nintendo DS</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
# -------------------------------------- Summary: -------------------------------------------------
left, center, right = st.columns([3, 0.10, 2])


with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("Named by former Nintendo President Satoru Iwata as Nintendo’s “third pillar”, the DS was the company’s most experimental project yet. " \
    "The console's design took inspiration from Nintendo’s " \
    '<a href="https://nintendo.fandom.com/wiki/Game_%26_Watch_(series)" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Game & Watch</strong></a>' \
    " handhelds, flaunting an impressive dual screen setup. Additionally, all DS consoles were backlit, " \
    "featured a touch screen, and backwards compatibility with the Game Boy Advance. Add-on revisions such as the DS Lite and DSi, and it's clear that Nintendo had big plans for" \
    " this device.", unsafe_allow_html=True)
    
    st.markdown("Nintendo’s “blue ocean” strategy to attract those who wouldn’t usually play games (with titles such as " 
    '<a href="https://nintendogs.fandom.com/wiki/Nintendogs" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Nintendogs</strong> </a> '
    "and " 
    '<a href="https://en.wikipedia.org/wiki/Brain_Age:_Train_Your_Brain_in_Minutes_a_Day!" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Brain Age</strong></a>'
    ") worked, as it’s touch screen simplified many button-heavy games into ones that felt natural and intuitive. Although many thought the two screens would be a gimmick, " 
    "many developers loved the system, as its versatility made developing games for the DS a unique challenge. " 
    "First-person shooters such as " 
    '<a href="https://callofduty.fandom.com/wiki/Call_of_Duty:_Black_Ops_(Nintendo_DS)" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Call of Duty: Black Ops</strong> </a> '
    "and " 
    '<a href="https://metroid.fandom.com/wiki/Metroid_Prime_Hunters" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Metroid Prime Hunters</strong> </a> '
    "used the touch screen for camera controls; " 
    '<a href="https://sonic.fandom.com/wiki/Sonic_Rush" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Sonic Rush</strong> </a> '
    "split the action of Sonic between both screens, while "
    '<a href="https://twewy.fandom.com/wiki/The_World_Ends_with_You" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>The World Ends With You</strong> </a> '
    "allowed you to control two characters across button and touch controls simultaneously. The DS’s uniqueness carved itself a niche in gaming, which both developers and gamers could appreciate. "
    "The handheld invented a new way to play games, revolutionizing Nintendo’s philosophy and birthing the origins of mobile gaming.",
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

    games = Image.open("photos/ds_games.png")
    st.image(games, width=350)

    st.markdown("<h6 style='text-align: center;'> <i>Image of DS and Games</i></h6>", unsafe_allow_html=True)

# -------------------------------------- Photo/Facts: -------------------------------------------------



with right: 
    logo = Image.open("photos/ds_logo.png") 
    st.image(logo, width=500)


    console = Image.open("photos/ds.jpeg")
    st.image(console, width=500)

    #Facts: 
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 11/21/2004</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> $150 (USD)", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 🥈 154.02 million </h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Other Variants:</u></h5>", unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: center;">
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_DS_Lite" target="_blank" style="color: #1D1D1D; text-decoration: none;">DS Lite</a> (2006)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_DSi" target="_blank" style="color: #1D1D1D; text-decoration: none;">DSi</a> (2008)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_DSi_XL" target="_blank" style="color: #1D1D1D; text-decoration: none;">DSi XL</a> (2009)
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Backwards Compatiable:</u> ", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center;">
        <strong><a href="https://nintendo.fandom.com/wiki/Game_Boy_Advance" target="_blank" style="color: #1D1D1D; text-decoration: none;">Game Boy Advance</a>
        """, unsafe_allow_html=True)
    st.write("")    




st.divider()



# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 🥈 948.76 million</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> 🥈 ~6 Games per Console</h5>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'> <u> Average Score of Games:</u> 🥈 {DS_mean_score}</h5>", unsafe_allow_html=True)
    st.write("")

    #Styling the DF based on GB Colors: 
    styled_DS_games_df = (
        DS_top_10_selling_df.style
        .format({'Units Sold (millions)': '{:.2f}'})
        .hide()

    )
    #Styling the DF based on GB Colors: 
    styled_DS_scores_df = (
        DS_top_10_rated_df.style
        .format({'Critic Score': '{:.2f}'})
        .hide()
    )


    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([3, 0.5, 3])

    with left:
        st.markdown("<h5 style='text-align: center;'> Top 10 Best Selling Games:</h5>", unsafe_allow_html=True)
        st.table(styled_DS_games_df)

    with right:
        st.markdown("<h5 style='text-align: center;'> Top 10 Highest Rated Games:</h5>", unsafe_allow_html=True)
        st.table(styled_DS_scores_df)
st.divider()



# ------------------------------------------ Graphs: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Graphs:</b></h3>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Console Sales By Variant & Console Sales by Region", "Software by Time & Console Sales by Time and Region"])
    with tab1:
        #Setting an Order:
        ds_order = ['Original', 'Lite', 'DSi', 'DSi XL', 'All']

        #Creating the Barplot:
        ds_total_bar = px.bar(
            all_console_total['DS_console_total'],
            x = 'Model',
            y = 'Total Sales',
            title = 'Sales of All DS Consoles',
            template = 'ggplot2',
            color_discrete_sequence = ['#A8A9AD'])

        #Updating the order by chronological order of release:
        ds_total_bar.update_xaxes(categoryorder='array', categoryarray = ds_order)


        # Set the background colors and adjust text color for readability
        ds_total_bar.update_layout(
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=15,
        yaxis_tickfont_size=15,
        
        # Title font settings
        title=dict(
            text='Nintendo DS Sales by Variant:',
            font=dict(size=25, color='white')
        ),
        
    )
        st.plotly_chart(ds_total_bar, use_container_width=True)


        st.divider()


        st.markdown("<h4 style='text-align: center;'><b>Total DS Sales By Region (millions)</b></h4>", unsafe_allow_html=True)
        ds_sales_in_millions = all_region_total['DS_region_total']['Total Sales'] / 1_000_000
        m_2 = folium.Map(location=[20, 0], zoom_start=1, min_zoom=1,max_zoom=10, tiles="OpenStreetMap")

        ds_c_map = Choropleth(geo_data=all_region_total['DS_region_total'].__geo_interface__, 
                data=ds_sales_in_millions, 
                key_on="feature.id", 
                fill_color='RdYlBu', 
                bins=[0, 15, 30, 45, 60, 75]
                ).add_to(m_2)

        folium.GeoJson(
            all_region_total['DS_region_total'].__geo_interface__,
            style_function=lambda x: {'fillColor': '#transparent', 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=[ 'Region', 'Total Sales'],  # Pulls the data column
                aliases=['Region:', 'Cumulative Sales:'],    # The label shown next to the value
                localize=True
            )
        ).add_to(m_2)


        # 4. Fit the map bounds automatically to your GeoJSON features
        m_2.fit_bounds(m_2.get_bounds())

        st_folium(m_2, use_container_width=True, height=400)

        st.markdown("<h6 style='text-align: center;'><i>Note: Sales only recorded for US/Japan/Other</i></h6>", unsafe_allow_html=True)





    with tab2:
        #Finding the Yearly Counts:
        DS_year_counts = DS_df['release_date'].dt.year.value_counts().sort_values(ascending = True).reset_index()
        DS_year_counts.columns = ['Release Date', 'Game Count']
        DS_year_counts = DS_year_counts.sort_values(by = 'Release Date', ascending = True)

        DS_game_ts = px.line( 
            DS_year_counts,
            x = 'Release Date',
            y = 'Game Count',
            title = 'DS Games by Release Year:',
            markers = True,
            template = 'plotly_white',
            color_discrete_sequence = ['#A8A9AD']
        )



     #Update Line Width:
        DS_game_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
        
        # Set the background colors and adjust text color for readability
        DS_game_ts.update_layout(

        
        # Title font settings
        title=dict(
            text='Nintendo DS Games by Release Year (of Top 250):',
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
        st.plotly_chart(DS_game_ts, use_container_width=True)


        st.divider()


        DS_console_ts = px.line( 
        DS_sales_gdf,
        x = 'Year',
        y = 'Sales',
        color = 'Region',
        title = 'DS Console Sales by Year:',
        markers = True,
        template = 'plotly_white',
        color_discrete_map={
            'Total': '#E7E7E7',  
            'Japan': '#BC002D',
            'United States': '#0A3161',
            'Other' : '#006400'
        }
    )

        DS_console_ts.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0,
                    y=1.15,
                    showactive=True,)])
        # Add explicit On/Off buttons
        DS_console_ts.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0,
                    y=1.15,
                    showactive=True,
                    )])


     #Update Line Width:
        DS_console_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
    
        
        # Set the background colors and adjust text color for readability
        DS_console_ts.update_layout(
        
        # Title font settings
        title=dict(
            text='Nintendo DS Sales by Time/Region:',
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
        st.plotly_chart(DS_console_ts, use_container_width=True)

        st.markdown("<h6 style='text-align: center;'><i>Note: Toggle Regions by clicking on the right.</i></h6>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------ Statistical Testing: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Hypothesis Question:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'><u>Was the DS Touch-Screen a Gimmick or a Fully Utlized Feature?:</u></h4>", unsafe_allow_html=True)
    st.markdown("One of the most unique features of the DS was it's bottom touch screen. Being able to control games using a stylus was ground-breaking at the time, " 
    "as while other devices have ultized a touch screen previously, the DS was the first to do it in the mainstream market. Despite this unique feature, some considered the touch screen a gimmick rather than a proper way to play games.", unsafe_allow_html=True)

    #Console:
    screen = Image.open("photos/ds_screen.jpg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(screen, width=300)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the DS Touch Screen</i></h6>", unsafe_allow_html=True)


    st.markdown("""
    While certain games fully utilize the feature, popular games such as
    <a href="https://www.mariowiki.com/New_Super_Mario_Bros." target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>New Super Mario Bros</strong>  </a> 
    or 
    <a href="https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_Diamond_and_Pearl_Versions" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Pokemon Diamond and Pearl</strong> </a>
    did not. <strong> Therefore, I wanted to test if developers took advantage of the DS's touch screen through unique play methods, or if only a few utilized it due to its novelty. </strong>
    """, unsafe_allow_html=True)
   


    with st.expander("**The Nerdy stuff**"):
        st.markdown("""We will be <strong>pulling subjects from the Top 250 DS Games list by using the stratified sampling method</strong>. I will extract samples from: <br>
        1) The Top 20 Best Selling Games (1-20)<br>
        2) The 'Middle' 20 Best Selling Games (115-134)<br>
        3) The 'Lowest' 20 Best Selling Games (231-250)<br><br>
        By sampling different sales groups, we are able to <strong>reduce bias</strong> by <strong>ensuring every category is well represented</strong>.""", unsafe_allow_html=True)

        st.write("")
        st.markdown("""
        <strong>The results of the test are found through observing a bar plot. If the number of games which utilize the touch screen is less than 25%, it's safe to assume most developers
        and consumers viewed the touch screen feature more of a quirk rather than a legitimate control option.</strong>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Conducting Test: </u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Failure</u>: <= 25% of games utilized the touch screen. </b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Not a Failure</u>: > 25% of games utilized the touch screen. </b></h5>", unsafe_allow_html=True)
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

    left, center, right = st.columns([0.5, 3, 0.5])
    with center:
        st.dataframe(ds_ts_df, height=400, hide_index=True)


    st.divider()
    st.markdown("<h5 style='text-align: center;'><b>Visualizing the Data: </b></h5>", unsafe_allow_html=True)

    ds_ts_bar = px.histogram(
    ds_ts_df,
    x= 'Touch Control',
    color = 'Touch Control',
    title= 'Count of DS Games With/Without Touch Screen Focus',
    labels= {'Touch Control' : 'Did the Game Focus on Touch Control?', 'count' : 'Number of Games'},
    color_discrete_map={'Yes': '#00FF00', 'No': '#FF0000'}
    )


    ds_ts_bar.update_layout(showlegend=False)


    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([1, 5, 1])

    with center:
        st.plotly_chart(ds_ts_bar, use_container_width=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Result:</u></h>", unsafe_allow_html=True)
    st.markdown("**Conclusion:** With a little less than half of our total games utilizing the DS's touch screen, it's safe to assume <strong>the control method was fairly common, and " \
    "not a gimmick.</strong> With games at every sale category using touch control in some way, many developers were able to take advantage of the unique hardware and push their creativity.</u>", unsafe_allow_html=True)




# ------------------------------------------ Fun Facts: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>DS Fun Facts:</u></h3>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    1. The DS is Nintendo’s best-selling (pure) handheld (154.02 million) </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    2. It is the first Nintendo system to feature built-in wireless/Wi-Fi multiplayer.
    </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    3. The DS Stands for both “Dual Screen” and “Developer’s System”
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    4. Former Nintendo President Hiroshi Yamauchi infamously stated, <i>"If it [The DS] succeeds, we rise to the heavens; if it fails, we sink into hell."</i> 
    <a href="https://web.archive.org/web/20060127211555/http://game-science.com/news/000406.html" target="_blank" style="color: #1D1D1D; text-decoration: none;">[Source]</a>
    , emphasizing the DS as one of Nintendo's largest financial gambles.
    </h5>""", unsafe_allow_html=True)

    st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/three_DS.py")