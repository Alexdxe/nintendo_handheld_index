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
    background-color: #850000 !important;
    color: #0D0D0D !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #AF5058 !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #9F6066 !important; /* Light grey cell background */
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
        @import url('https://fonts.googleapis.com/css2?family=Asap+Sharp:ital,wght@0,100..900;1,100..900&display=swap');

        /* Target this specific class on h1 to override global header font rules */
        h1.sharp-header {
            font-family: 'Asap Sharp', sans-serif !important;
            text-align: center;
            color: #a91101 !important;
        }
    </style>

    <h1 class="sharp-header">
        <a href="https://nintendo.fandom.com/wiki/Nintendo_Switch" target="_blank" style="color: #1D1D1D; text-decoration: none;">Switch</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
# -------------------------------------- Summary: -------------------------------------------------
left, center, right = st.columns([3, 0.10, 2])


with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("The Nintendo Switch is a hybrid console, meaning it could be played both on a TV (docked, through HDMI) or handheld. While the concept was intriguing, some were skeptical. " \
    "The "
    '<a href="https://nintendo.fandom.com/wiki/Wii_U" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Wii U</strong></a> '
    " is Nintendo’s worst-selling console, as confusion around its gimmick and marketing left many questioning it's identity. However, when the Switch launched, it received universal acclaim. " \
    "Its innovative design, projected first-party games, and processing power led the Switch to sell out at many locations during its launch.", unsafe_allow_html=True)
    
    st.markdown("Being faster than the Wii U, it ran many home-console exclusives that previous handhelds could only dream of. Most notably, " \
       '<a href="https://zelda.fandom.com/wiki/The_Legend_of_Zelda:_Breath_of_the_Wild" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>The Legend of Zelda: Breath of the Wild</strong></a> '   
       "The game showcased a vast, in-depth open world which players could sink hundreds of hours into exploring. Having an experience this large on a device so small was shocking, "
       "and (alongside other titles) convinced many to pick up the console. ", unsafe_allow_html=True)
   
    st.markdown("Today, the Nintendo Switch is not only Nintendo’s best-selling console, but is projected to outsell the  "
       '<a href="https://playstation.fandom.com/wiki/PlayStation_2" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>PlayStation 2</strong></a> '
       ", the best-selling gaming console of all time. Nintendo learned from mistakes during the Wii U era and ensured they had a unique idea, a strong game line-up," \
       " and broad appeal. The Switch merged the convenience of handheld gaming with the larger-than-life home console experiences to create one of the greatest consoles of all time. ", unsafe_allow_html=True)
   
    st.write("")



# -------------------------------------- Photo/Facts: -------------------------------------------------




with right: 
    logo = Image.open("photos/switch_logo.png") 
    st.image(logo, width=500)


    console = Image.open("photos/switch.jpeg")
    st.image(console, width=500)

    #Facts: 
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 03/03/2017</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>$300 (USD)</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 🥇 156.59+ million</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Other Variants:</u></h5>", unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: center;">
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_Switch_Lite" target="_blank" style="color: #1D1D1D; text-decoration: none;">Nintendo Switch Lite</a> (2020)<br>
    <strong><a href="https://nintendo.fandom.com/wiki/Nintendo_Switch_OLED" target="_blank" style="color: #1D1D1D; text-decoration: none;">Nintendo Switch OLED</a> (2021)
    """, unsafe_allow_html=True)

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

    games = Image.open("photos/switch_games.jpg")
    st.image(games, width=350)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the Nintendo Switch and it's biggest releases in 2023</i></h6>", unsafe_allow_html=True)


st.divider()

# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 🥇 1,561.95+ Million Units (Includes eShop)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> 🥇 ~10 Games per Console</h5>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'> <u> Average Score of Games:</u> 🥇 {switch_mean_score}</h5>", unsafe_allow_html=True)
    st.write("")

    #Styling the DF based on GB Colors: 
    styled_switch_games_df = (
        switch_top_10_selling_df.style
        .format({'Units Sold (millions)': '{:.2f}'})
        .hide()

    )
    #Styling the DF based on GB Colors: 
    styled_switch_scores_df = (
        switch_top_10_rated_df.style
        .format({'Critic Score': '{:.2f}'})
        .hide()
    )


    # 2. Make the table skinnier using st.columns (e.g., 1:2:1 ratio)
    left, center, right = st.columns([3, 0.5, 3])

    with left:
        st.markdown("<h5 style='text-align: center;'> Top 10 Best Selling Games:</h5>", unsafe_allow_html=True)
        st.table(styled_switch_games_df)

    with right:
        st.markdown("<h5 style='text-align: center;'> Top 10 Highest Rated Games:</h5>", unsafe_allow_html=True)
        st.table(styled_switch_scores_df)
st.divider()



# ------------------------------------------ Graphs: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Graphs:</b></h3>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Console Sales By Variant",  "Console Sales by Region", "Software by Time", 'Console Sales by Time and Region'])
    with tab1:
        #Setting an Order:
        switch_order = ['Original', 'Lite', 'OLED', 'All']

        #Creating the Barplot:
        switch_total_bar = px.bar(
            all_console_total['Switch_console_total'],
            x = 'Model',
            y = 'Total Sales',
            title = 'Sales of All Switch Consoles',
            template = 'ggplot2',
            color_discrete_sequence = ['#1FC0FF'])

        #Updating the order by chronological order of release:
        switch_total_bar.update_xaxes(categoryorder='array', categoryarray = switch_order)

        # Set the background colors and adjust text color for readability
        switch_total_bar.update_layout(
        paper_bgcolor='#AF5058',
        plot_bgcolor='#e5e5e5',
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=15,
        yaxis_tickfont_size=15,
        
        # Title font settings
        title=dict(
            text='Switch Sales by Variant:',
            font=dict(size=25, color='white')
        ),
        
    )
        st.plotly_chart(switch_total_bar, use_container_width=True)




        
    with tab2: 
        st.markdown("<h4 style='text-align: center;'><b>Total Switch Sales By Region (By Millions)</b></h4>", unsafe_allow_html=True)
        sales_in_millions = all_region_total['Switch_region_total']['Total Sales'] / 1_000_000
        m_4 = folium.Map(location=[20, 0], zoom_start=1, min_zoom=1,max_zoom=10,tiles="OpenStreetMap")

        c_map = Choropleth(geo_data=all_region_total['Switch_region_total'].__geo_interface__, 
                data=sales_in_millions, 
                key_on="feature.id", 
                fill_color='RdBu', 
                bins=[0, 10, 20, 30, 40, 50, 60]
                ).add_to(m_4)

        folium.GeoJson(
            all_region_total['Switch_region_total'].__geo_interface__,
            style_function=lambda x: {'fillColor': '#transparent', 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=[ 'Region', 'Total Sales'],  # Pulls the data column
                aliases=['Region:', 'Cumulative Sales:'],    # The label shown next to the value
                localize=True
            )
        
        ).add_to(m_4)

        # 4. Fit the map bounds automatically to your GeoJSON features
        m_4.fit_bounds(m_4.get_bounds())

        st_folium(m_4, use_container_width=True, height=400)

        st.markdown("<h6 style='text-align: center;'><i>Note: Sales only recorded for US/Japan/Europe/Other</i></h6>", unsafe_allow_html=True)





    with tab3:
        #Finding the Yearly Counts:
        switch_year_counts = switch_df['release_date'].dt.year.value_counts().sort_values(ascending = True).reset_index()
        switch_year_counts.columns = ['Release Date', 'Game Count']
        switch_year_counts = switch_year_counts.sort_values(by = 'Release Date', ascending = True)

        switch_game_ts = px.line( 
            switch_year_counts,
            x = 'Release Date',
            y = 'Game Count',
            title = 'Switch Games by Release Year:',
            markers = True,
            template = 'seaborn',
            color_discrete_sequence = ['#0AB9E6']
        ) 

     #Update Line Width:
        switch_game_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
        
        # Set the background colors and adjust text color for readability
        switch_game_ts.update_layout(
        paper_bgcolor='#A4A4A4',
        plot_bgcolor='#e5e5e5',
        
        # Title font settings
        title=dict(
            text='Switch Games by Release Year:',
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
        st.plotly_chart(switch_game_ts, use_container_width=True)





    with tab4: 
        switch_console_ts = px.line( 
        switch_sales_gdf,
        x = 'Year',
        y = 'Sales',
        color = 'Region',
        title = 'Switch Sales by Time and Region:',
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
        switch_console_ts.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.0,
                    y=1.15,
                    showactive=True,
                    )])


     #Update Line Width:
        switch_console_ts.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
    
        
        # Set the background colors and adjust text color for readability
        switch_console_ts.update_layout(
        paper_bgcolor='#AF5058',
        plot_bgcolor='#e5e5e5',
        
        # Title font settings
        title=dict(
            text='Switch Sales by Time and Region:',
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
        st.plotly_chart(switch_console_ts, use_container_width=True)

        st.markdown("<h6 style='text-align: center;'><i>Note: Toggle Regions by clicking on the right.</i></h6>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------ Statistical Testing: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Hypothesis Question:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'><u>When is the Switch Expected to Outsell the PlayStation 2?</u></h4>", unsafe_allow_html=True)
    st.markdown("The "
    '<a href="https://playstation.fandom.com/wiki/PlayStation_2" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Playstation 2</strong></a> '
    " is one of the greatest video game consoles of all time. With a diverse game library, affordable pricing, and versatile features, the console was owned by millions of players worldwide." \
    " However, with each passing year, the Nintendo Switch has slowly been gaining ground as the next best selling console." \
    " While not as versatile as the PS2, the Switch makes up for it with an incredible first-party library and a hybrid form factor which has made it popular amongst both casuals and hardcore gamers." \
    "", unsafe_allow_html=True)

    #Console:
    console = Image.open("photos/ps2.jpeg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(console, width=300)

    st.markdown("<h6 style='text-align: center;'> <i>Image of the PS2</i></h6>", unsafe_allow_html=True)

    st.markdown("As the Switch's sales numbers slowly approach the PS2's, many began speculating when the Switch would ultimately outsell the PS2. As of September 2026, the Switch is just 3.41 million units shy of tying with the PS2 as the best selling console of all time." 
    " <strong>My goal is to figure out when (theoretically) the Switch will outsell the PS2, using the Switch's historical quarterly sales data</strong>."            
    , unsafe_allow_html=True)   


    with st.expander("**The Nerdy stuff**"):
        st.markdown("I'll be using a SARIMAX model to predict the Switch's future sales. <strong>SARIMAX accounts for both seasonality and trends in our data</strong>. This is important, because <b>Nintendo tends to sell more consoles during the holiday season, and consoles have a rough 'bell-curve' distribution when it comes to hardware sales.</b> " \
     ,unsafe_allow_html=True)
        st.markdown("After training and chosing the best model using SARIMAX, we'll try to <b>predict the Switch's sales for the next 5 years to identify when the Switch will outsell the PS2<b>.", unsafe_allow_html=True)

    st.divider()
    
    st.markdown("<h4 style='text-align: center;'><u>Running the Algorithm: </u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>Goal: Identify When the Switch Will Outsell the PS2</b></h5>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Observing Historical Quarterly Sales:</b></h5>", unsafe_allow_html=True)

    #updating date column to be type datetime:
    switch_sales_df['Date'] = pd.to_datetime(switch_sales_df['Date'], format = '%m/%Y')


    switch_q_sales = px.line( 
    switch_sales_df,
    x = 'Date',
    y = 'Quarterly Sales',
    title = 'Switch Hardware Sales by Quarter:',
    markers = True,
    template = 'plotly_white',
    color_discrete_sequence = ['#FF3131']
)

    #Update Line Width:
    switch_q_sales.update_traces(
        line=dict(width=5),  # Set line thickness (default is usually 2)
        marker=dict(size=8)  # Optional: scale markers to match the thicker line
        )
            
            # Set the background colors and adjust text color for readability
    switch_q_sales.update_layout(
        paper_bgcolor='#AF5058',
        plot_bgcolor='#e5e5e5',
            
     # Title font settings
        title=dict(
            text='Switch Games by Release Year:',
            font=dict(size=25, color='white')
        ))

    st.plotly_chart(switch_q_sales, use_container_width=True)


    #Applying Log Transformation to our data:
    y = np.log(switch_sales_df['Quarterly Sales'])

    #Finding the Best Model: 
    best_model = pm.auto_arima(
        y,
        seasonal = True,
        m = 4,
        stepwise = True,
        suppress_warnings = True,
        error_action = 'ignore'
    )
    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Finding the Best Model:</b></h5>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'> <u>SARIMAX: (0,1,1,4)</u></h5>", unsafe_allow_html=True)
    with st.expander("**Interpreting the Model:**"):
        st.markdown("1) <strong>AR of 0:</strong> Current quarter sales do NOT learn/are modeled from any direct previous term.", unsafe_allow_html=True)
        st.markdown("2) <strong>Difference of 1:</strong> Used to ensure stationarity.", unsafe_allow_html=True)
        st.markdown("3) <strong>Seasonal MA of 1:</strong> Current quarter's prediction is based on the forecast error exactly a year ago (check below). ", unsafe_allow_html=True)
        st.markdown("4) <strong>Seasonal period of 4:</strong> A repeating cycle of every 4 quarters. ", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Forecasting the Next 5 Years:</b></h5>", unsafe_allow_html=True)
    #PS2 Total Sales:
    goal = 160

    #Forecasting the log transformation of the data:
    forecast_log = best_model.predict(n_periods=20)

    #Converting back:
    forecast = np.exp(forecast_log)

    #Defining Current Sales as of 8/17/2026
    current_total = switch_sales_df['Quarterly Sales'].sum()

    #Finding the Cumulative Sales Numbers: 
    cumulative_sales = current_total + forecast.cumsum()

    #Isolate milestone tracking
    milestone_df = pd.DataFrame(
        {
            "Quarterly_Forecast": forecast,
            "Cumulative_Sales": cumulative_sales,
            "Exceeds_160M": cumulative_sales >= goal,
        }
    )

    st.table(milestone_df)    

    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Result:</u></h>", unsafe_allow_html=True)
    st.markdown("**Conclusion:** According to our model, <strong>the Switch is expected to outsell the PS2 in Q1 of 2028</strong>. Given how slow sales move after another home console is released, this number does not seem unreasonable. Additionally, with Nintendo still supporting the Switch with first party games, interest around the console will still be present until 2028.", unsafe_allow_html=True)

# ------------------------------------------ Fun Facts: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><u>Switch Fun Facts:</u></h3>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    1. The Switch is Nintendo’s best-selling console of all time. (156.59 million)
    </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    2. It has the largest software library of any handheld (4,000+)
    </h5>""", unsafe_allow_html=True)

    st.markdown("""<h5 style='text-align: center;'>
    3. It's best selling game, 
    <a href="https://mariokart.fandom.com/wiki/Mario_Kart_8_Deluxe" target="_blank" style="color: #1D1D1D; text-decoration: none;">Mario Kart 8 Deluxe</a>
    , is simultaneously a remaster and the best selling game on the 
    <a href="https://nintendo.fandom.com/wiki/Wii_U" target="_blank" style="color: #1D1D1D; text-decoration: none;">Wii U</a>
    . 
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    4. It is the first Nintendo console to receive a price bump ($330, 08/01/2026).
    </h5>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;'>
    5. It took ~10 months for the Switch to outsell the Wii U.
    </h5>""", unsafe_allow_html=True)
    st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/Switch2.py")