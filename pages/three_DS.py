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
    background-color: #126E65 !important;
    color: #0D0D0D !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #168A7E !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #1BA597 !important; /* Light grey cell background */
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
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,200..900;1,200..900&display=swap');

        /* Target this specific class on h1 to override global header font rules */
        h1.source-header {
            font-family: 'Source Sans 3', sans-serif !important;
            text-align: center;
            color: #a91101 !important;
        }
    </style>

    <h1 class="source-header">
        <a href="https://nintendo.fandom.com/wiki/Nintendo_3DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">Nintendo 3DS</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
# -------------------------------------- Summary: -------------------------------------------------
left, center, right = st.columns([3, 0.10, 2])


with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("The 3DS is quite similar to the GBA in many aspects. The device sought to improve on everything that made the DS successful. It was more powerful, featured improved software, and a sharper camera (compared to the DSi). " \
    "However, the 3DS's main selling point was its ability to deliver 3D video/visuals without glasses. It was a unique feature, and while the idea was intriguing, Nintendo scared off most of its consumers with its launch price.")
    
    st.markdown("The device cost \$250, almost \$100 more than the " 
    '<a href="https://nintendo.fandom.com/wiki/Wii" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Wii</strong> </a> '
    "in 2011. It was a hard pill to swallow, and paired with a lackluster launch lineup, left many owners with little to do. The console wasn't selling, and with the holiday season approaching, Nintendo slashed the price to " 
    '<a href="https://www.theguardian.com/technology/gamesblog/2011/jul/28/3ds-price-cut-nintendo-loss" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>$170</strong></a> '
    " to finally bolster sales. With strong first-party support from Nintendo and many revisions, the 3DS overcame its initial growing pains and became the company’s main focus in the early 2010s. Games such as " 
    '<a href="https://residentevil.fandom.com/wiki/Resident_Evil:_Revelations" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Resident Evil Revelations</strong></a> '
    "showcased the device’s graphical capabilities, while " 
        '<a href="https://www.mariowiki.com/Super_Mario_3D_Land" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Super Mario 3D Land</strong></a> '
    " utilized 3D visuals for problem-solving. Simply, the console improved on what the DS lacked: power. ", unsafe_allow_html=True)
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

    games = Image.open("photos/3ds_games.jpeg")
    st.image(games, width=350)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the 3DS Games & AR Cards</i></h6>", unsafe_allow_html=True)

# -------------------------------------- Photo/Facts: -------------------------------------------------




with right: 
    logo = Image.open("photos/3ds_logo.png") 
    st.image(logo, width=500)


    console = Image.open("photos/3ds.jpg")
    st.image(console, width=500)

    #Facts: 
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 02/26/2011</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>$250 (USD)</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 75.94 million </h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Other Variants:</u></h5>", unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: center;">
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_3DS_XL" target="_blank" style="color: #1D1D1D; text-decoration: none;">3DS XL</a> (2012)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_2DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">2DS</a> (2013)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/New_Nintendo_3DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">New 3DS</a> (2015)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/New_Nintendo_3DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">New 3DS XL</a> (2015)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/New_Nintendo_2DS_XL" target="_blank" style="color: #1D1D1D; text-decoration: none;">New 2DS XL</a> (2017)
    """, unsafe_allow_html=True)
    st.write("")


    st.markdown("<h5 style='text-align: center;'> <u> Backwards Compatiable:</u> ", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center;">
        <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_DS" target="_blank" style="color: #1D1D1D; text-decoration: none;">Nintendo DS</a>
        """, unsafe_allow_html=True)
    st.write("")    

st.divider()

# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 392.31 million (including eShop)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> 🥉 ~5 Games per Console</h5>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'> <u> Average Score of Games:</u> 🥉 {three_DS_mean_score}</h5>", unsafe_allow_html=True)
    st.write("")

    #Styling the DF based on GB Colors: 
    styled_3DS_games_df = (
        three_ds_top_10_selling_df.style
        .format({'Units Sold (millions)': '{:.2f}'})
        .hide()

    )
    #Styling the DF based on GB Colors: 
    styled_3DS_scores_df = (
        three_ds_top_10_rated_df.style
        .format({'Critic Score': '{:.2f}'})
        .hide()
    )


    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([3, 0.5, 3])

    with left:
        st.markdown("<h5 style='text-align: center;'> Top 10 Best Selling Games:</h5>", unsafe_allow_html=True)
        st.table(three_ds_top_10_selling_df)

    with right:
        st.markdown("<h5 style='text-align: center;'> Top 10 Highest Rated Games:</h5>", unsafe_allow_html=True)
        st.table(styled_3DS_scores_df)
st.divider()



# ------------------------------------------ Graphs: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Graphs:</b></h3>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Console Sales By Variant & Console Sales by Region", "Software by Time & Console Sales by Time and Region"])
    with tab1:
        #Setting an Order:
        three_ds_order = ['Original', 'XL', '2DS', 'New 3DS', 'New 3DS XL', '2DS XL', 'All']

        #Creating the Barplot:
        three_ds_total_bar = px.bar(
            all_console_total['3DS_console_total'],
            x = 'Model',
            y = 'Total Sales',
            title = 'Sales of All 3DS Consoles',
            template = 'ggplot2',
            color_discrete_sequence = ['#ba3c3c'])

        #Updating the order by chronological order of release:
        three_ds_total_bar.update_xaxes(categoryorder='array', categoryarray = three_ds_order)


        # Set the background colors and adjust text color for readability
        three_ds_total_bar.update_layout(
        paper_bgcolor='#168A7E',
        plot_bgcolor='#e5e5e5',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=15,
        yaxis_tickfont_size=15,
        
        # Title font settings
        title=dict(
            text='Nintendo 3DS Sales by Variant:',
            font=dict(size=25, color='white')
        ),
        
    )
        st.plotly_chart(three_ds_total_bar, use_container_width=True)


        st.divider()


        st.markdown("<h4 style='text-align: center;'><b>Total 3DS Sales By Region (By Millions)</b></h4>", unsafe_allow_html=True)
        three_ds_sales_in_millions = all_region_total['3DS_region_total']['Total Sales'] / 1_000_000
        m_3 = folium.Map(location=[20, 0], zoom_start=1, min_zoom=1,max_zoom=10, tiles="OpenStreetMap")

        three_ds_c_map = Choropleth(geo_data=all_region_total['3DS_region_total'].__geo_interface__, 
                data=three_ds_sales_in_millions, 
                key_on="feature.id", 
                fill_color='RdYlBu', 
                bins=[0, 6, 12, 18, 24, 30],
                ).add_to(m_3)

        folium.GeoJson(
            all_region_total['3DS_region_total'].__geo_interface__,
            style_function=lambda x: {'fillColor': '#transparent', 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=[ 'Region', 'Total Sales'],  # Pulls the data column
                aliases=['Region:', 'Cumulative Sales:'],    # The label shown next to the value
                localize=True
            )
        ).add_to(m_3)


        # 4. Fit the map bounds automatically to your GeoJSON features
        m_3.fit_bounds(m_3.get_bounds())

        st_folium(m_3, use_container_width=True, height=400)

        st.markdown("<h6 style='text-align: center;'><i>Note: Sales only recorded for US/Japan/Other</i></h6>", unsafe_allow_html=True)





    with tab2:
        #Finding the Yearly Counts:
        three_DS_year_counts = three_DS_df['release_date'].dt.year.value_counts().sort_values(ascending = True).reset_index()
        three_DS_year_counts.columns = ['Release Date', 'Game Count']
        three_DS_year_counts = three_DS_year_counts.sort_values(by = 'Release Date', ascending = True)

        three_DS_game_ts = px.line( 
            three_DS_year_counts,
            x = 'Release Date',
            y = 'Game Count',
            title = '3DS Games by Release Year:',
            markers = True,
            template = 'plotly_white',
            color_discrete_sequence = ['#6dc1e3']
        )


     #Update Line Width:
        three_DS_game_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
        
        # Set the background colors and adjust text color for readability
        three_DS_game_ts.update_layout(
        paper_bgcolor='#168A7E',
        plot_bgcolor='#e5e5e5',

        
        # Title font settings
        title=dict(
            text='Nintendo 3DS Games by Release Year (of Top 250):',
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
        st.plotly_chart(three_DS_game_ts, use_container_width=True)


        st.divider()


        three_DS_console_ts = px.line( 
        three_DS_sales_gdf,
        x = 'Year',
        y = 'Sales',
        color = 'Region',
        title = '3DS Console Sales by Year:',
        markers = True,
        template = 'plotly_white',
        color_discrete_map={
            'Total': '#222222',  
            'Japan': '#BC002D',
            'United States': '#0A3161',
            'Other' : '#006400'
        }
    )

        three_DS_console_ts.update_layout(
        paper_bgcolor='#168A7E',
        plot_bgcolor='#e5e5e5',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0,
                    y=1.15,
                    showactive=True,)])


     #Update Line Width:
        three_DS_console_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
    
        
        # Set the background colors and adjust text color for readability
        three_DS_console_ts.update_layout(
        
        # Title font settings
        title=dict(
            text='Nintendo 3DS Sales by Time/Region:',
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
        st.plotly_chart(three_DS_console_ts, use_container_width=True)

        st.markdown("<h6 style='text-align: center;'><i>Note: Toggle Regions by clicking on the right.</i></h6>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------ Statistical Testing: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Hypothesis Question:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'><u>Which Are the Most Successful 3DS Models?</u></h4>", unsafe_allow_html=True)
    st.markdown("The Nintendo 3DS had many different variants through it's lifespan. From the infamous 2DS to the beloved New 3DS XL, the 3DS family of systems comes in many shapes and sizes. Additionally, the revisions have all released at different periods of the 3DS's life, showing the evolution of the 3DS's design and philosphy.", unsafe_allow_html=True)

    #Models:
    screen = Image.open("photos/3ds_models.jpg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(screen, width=300)

    st.markdown("<h6 style='text-align: center;'> <i>Image of All 3DS Consoles</i></h6>", unsafe_allow_html=True)

    st.markdown("Because this variety, I wanted to explore <strong>which models can be considered successes through recording various measures related to the success of a console.</strong>", unsafe_allow_html=True)

    with st.expander("**The Nerdy stuff**"):
        st.markdown("We will be running the <strong>K-Means Clustering algorithm to group the 3DS consoles based on their sales performance and features</strong>. By categorizing the consoles into distinct groups, we can better understand a variant's similarities and differences." \
        " After grouping the consoles into distinct clusters, we can analyze the average performance metrics for each group, and identify which consoles can be considered successes/failures.", unsafe_allow_html=True)

        st.markdown("""The algorithm will be run on the following features:<br>
        1) Units Sold<br>
        2) Months on Market<br>
        3) Launch Price<br>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<h4 style='text-align: center;'><u>Running the Algorithms:</u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Goal</u>: Identify the Most Successful 3DS Models</b></h5>", unsafe_allow_html=True)
    st.divider()

    #Creating a Copy
    three_ds_sales_df = three_ds_sales_df.copy()

    full_sales_df = three_ds_sales_df[['Model', 'Units Sold', 'Time in Market (Months)', 'Launch Price']].copy()
    #Defining columnd used to predit our model:
    x = three_ds_sales_df[['Units Sold', 'Time in Market (Months)', 'Launch Price']].copy()

    st.markdown("<h5 style='text-align: center;'><b>Observing DataFrame: </b></h5>", unsafe_allow_html=True)



    st.table(full_sales_df)

    #Scaling our measures:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    #Fitting a K-means model: 
    kmeans = KMeans(n_clusters=2, random_state=127, n_init=10)
    three_ds_sales_df['Cluster'] = kmeans.fit_predict(x_scaled)

    #Observing Results:
    results = three_ds_sales_df[['Model', 'Cluster']]


    cluster_averages = three_ds_sales_df.groupby('Cluster')[ ['Units Sold', 'Time in Market (Months)',\
                                                          'Launch Price']].mean()

    cluster_averages = cluster_averages.reset_index()

    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Observing the Results: </b></h5>", unsafe_allow_html=True)
    st.table(results)

    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Cluster Averages: </b></h5>", unsafe_allow_html=True)
    st.table(cluster_averages)

    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Result:</u></h>", unsafe_allow_html=True)
    st.markdown("""<strong>Conclusion:</strong> <strong>The original 2 models of the 3DS's can be considered the most successful models in the family</strong>. On average, the original 3DS's cluster
    has sold 3x more than sales than the newer models and original 2DS, despite spending less time on the market. This is likely because these models were released earlier into the 3DS lifespan, which generated more hype/attention than the later revisions.""", unsafe_allow_html=True)




# ------------------------------------------ Fun Facts: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>3DS Fun Facts:</u></h3>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    1. The 3DS is Nintendo’s worst-selling handheld (75.94 million)
    </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    2. It is the most revised handheld of all time (6 models)
    </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    3. Many 3DS models cost more today than they did during their launch. 
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    4. The 3DS has one of the largest modding communities of any console, with many making demakes of famous games such as, 
    <a href="https://gbatemp.net/threads/release-sonic-mania-3ds-port.618771/" target="_blank" style="color: #1D1D1D; text-decoration: none;">Sonic Mania </a>
    and 
    <a href="https://www.reddit.com/r/3dspiracy/comments/1sm9f25/i_made_a_3ds_remake_of_balatro_and_heres_version/" target="_blank" style="color: #1D1D1D; text-decoration: none;">Balatro</a>
    </h5>""", unsafe_allow_html=True)



col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/Switch.py")
