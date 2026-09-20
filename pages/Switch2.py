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
    background-color: #963939 !important;
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
    background-color: #ba3c3c !important;
    color: #0D0D0D !important;
}

/* 5. TABLE STYLING */
[data-testid="stTable"] th {
    background-color: #9F6066 !important; /* Darker grey header */
    color: #ffffff !important;             /* White header text */
}

[data-testid="stTable"] td, 
[data-testid="stTable"] td * {
    background-color: #8F7073 !important; /* Light grey cell background */
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
        @import url('https://fonts.googleapis.com/css2?family=Pliant:ital,wght@0,100..900;1,100..900&display=swap');

        /* Target this specific class on h1 to override global header font rules */
        h1.pliant-header {
            font-family: 'Pilant', sans-serif !important;
            text-align: center;
            color: #a91101 !important;
        }
    </style>

    <h1 class="pliant-header">
        <a href="https://nintendo.fandom.com/wiki/Nintendo_Switch_2" target="_blank" style="color: #1D1D1D; text-decoration: none;">Switch 2</a>
    </h1>
""", unsafe_allow_html=True)
st.write("")
# -------------------------------------- Summary: -------------------------------------------------
left, center, right = st.columns([3, 0.10, 2])


with left:
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown("The Switch 2 launched in May 2025 to great success. Following the mold of the GBA/3DS, the Switch 2 is essentially a better Switch. " \
    "The console was larger, had vastly more processing power, and felt more premium. It also featured new unique ideas, such as mouse mode and video chat. While not as groundbreaking as the original Switch, " \
    "the main feature consumers wanted was more power, which the Switch more than delivered.", unsafe_allow_html=True)
    
    st.markdown("Roughly 10x faster than the original, the Switch 2 was a massive upgrade. Although not as powerful as other home consoles, the Switch 2 still offers enough power to run most games competently. " \
        "In fact, for the first time, it seems that many developers are releasing games across all three platforms (" \
        '<a href="https://xbox.fandom.com/wiki/Xbox_Series_X" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Xbox Series X</strong></a>, '
        '<a href="https://playstation.fandom.com/wiki/PlayStation_5" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>PS5</strong></a>, '
        "Switch 2) simultaneously. To finally see big third-party games (without many compromises) on Nintendo platforms is shocking, " \
        "as the company has almost always relied on it’s in house IPs to carry their game libraries. With a combination of record-breaking sales numbers and competent hardware, the Switch 2 seems to be Nintendo’s finest console yet.", unsafe_allow_html=True)
    
    st.write("")



# -------------------------------------- Photo/Facts: -------------------------------------------------




with right: 
    logo = Image.open("photos/switch2_logo.png") 
    st.image(logo, width=500)


    console = Image.open("photos/switch2.jpg")
    st.image(console, width=500)

    #Facts:
    st.markdown("<h5 style='text-align: center;'> <u> Launch Date:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 06/05/2025</h6>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5 style='text-align: center;'> <u> Launch Price:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>$450 (USD)</h6>", unsafe_allow_html=True)
    st.write("") 

    st.markdown("<h5 style='text-align: center;'> <u> Units Sold:</u> ", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'> 23.68+ million</h6>", unsafe_allow_html=True)
    st.write("")



st.divider()

# -------------------------------------- Summary of Software -------------------------------------------------

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Software Facts:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Software Sold:</u> 58.17+ million (including eShop)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <u> Average Software Sold:</u> ~3 Games per Console</h5>", unsafe_allow_html=True)
    st.write("")

st.divider()

# ------------------------------------------ Statistical Testing: -----------------------------------------------------
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'><b>Hypothesis Question:</b></h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'><u>Can We Predict the Switch 2's Future Sales?</u></h4>", unsafe_allow_html=True)
    st.markdown("The Switch 2 has been out for a little over a year, and it's impact on the gaming industry has been massive. Despite only a handful of first-party exclusives, the console has sold over 23.68 million units. It's impressive sales performance despite a lack of a new gimmick is a testament to the Nintendo branding, as well as the faith many fans are putting into the company. ",unsafe_allow_html=True)
    st.markdown("Given the Switch 2's early success, I wanted to see if I could <strong>predict the Switch 2's future sales given Nintendo's previous console's sales data</strong>. Similar to the Switch's sales prediction, <strong>we'll look at historical numbers and additional features each Nintendo console had to train our algorithm</strong>. We want to see the Switch 2's total sales numbers in 2 years.", unsafe_allow_html=True)
                


    with st.expander("**The Nerdy stuff**"):
        st.markdown("""I'll be using a Random Forest Regressor to predict the Switch 2's future sales. <strong>Random Forest Regressor is perfect for our scenario, as it's predicting a continuous value using non-linear relationships</strong>. The model will be trained on:<br><br>
        1) Is it a Home Console? (Yes/No) <br>
        2) Is it a Handheld Console? (Yes/No) <br>
        3) Is it a Hybrid Console? (Yes/No) <br>
        4) Is it Backwards Compatible? (Yes/No) <br>
        5) Adjusted Launch Price (USD) <br>
        6) Number of Console Variants <br>
        7) Number of First Party Games (1st Year) <br>
        8) Quarter Index <br>
        9) Sales (Millions) <br>
        10) Is the Quarter a Holiday (Yes/No)

        """,unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Running the Algorithm: </u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>Goal: Predict the Total Sales of the Switch 2 by July 2028. </b></h5>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Observing the Historical Data:</b></h5>", unsafe_allow_html=True)

    #Defining a Training DF:
    train_df = console_sales_df[~((console_sales_df['Console'] == 'Switch 2') & (console_sales_df['Quarter Index'] > 5))].copy()

    #Testing DF:
    test_df = console_sales_df[((console_sales_df['Console'] == 'Switch 2') & (console_sales_df['Quarter Index'] > 5))].copy()

    #Defining Predictors:
    train_x = train_df[['Home Console', 'Handheld', 'Hybrid', 'Backwards Compatiable',
                        'Adjusted Launch Price', '# First Year Nintendo Games',
                        'Quarter Index', 'Holiday']]

    test_x = test_df[['Home Console', 'Handheld', 'Hybrid', 'Backwards Compatiable',
                        'Adjusted Launch Price', '# First Year Nintendo Games',
                        'Quarter Index', 'Holiday']]

    #Defining Response Variable:
    train_y = train_df['Sales (Millions)']

    test_y = test_df['Sales (Millions)']


    adjusted_df = train_df.rename(columns={"# First Year Nintendo Games": "First Yr Nintendo Games"})
    st.markdown("<h6 style='text-align: center;'><i>Scroll horizontally to observe other columns:</i></h6>", unsafe_allow_html=True)
    st.dataframe(adjusted_df, hide_index=True) 
    st.markdown("<h6 style='text-align: center;'><i>Note: 0 indicates No, 1 indicates Yes. </i></h6>", unsafe_allow_html=True)


    st.divider()

    st.markdown("<h5 style='text-align: center;'><b>Observing the Predictions Using the Data:</b></h5>", unsafe_allow_html=True)




    #Training the model:
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=127
    )

    model.fit(train_x, train_y)


    #Making Predictions using the training model:
    predictions = model.predict(test_x)

    #Appending to Test DF:
    test_df['Predicted Sales'] = predictions




    #Pulling Switch 2 Sales From Train DF: 
    switch2 = train_df[train_df['Console'] == 'Switch 2']

    #Combining Sales: 
    switch2_df = pd.concat([switch2, test_df], axis = 0)

    #Filling in 0s:
    switch2_df = switch2_df.fillna(0)

    #Combining both theoretical sales and real into one column:
    switch2_df['Total Sales (Millions)'] = switch2_df['Sales (Millions)'] + switch2_df['Predicted Sales']


    #Adding Quarter Values:
    q_dates = ['09/2025', '12/2025', '03/2026', '06/2026', '09/2026', '12/2026',
                '03/2027', '06/2027', '09/2027', '12/2027', '03/2028', '06/2028']
            

    #Appending the Values to the DF: 
    switch2_df['Quarter Dates'] = q_dates 

    #Changing to datetime:
    switch2_df['Quarter Dates'] = pd.to_datetime(switch2_df['Quarter Dates'], format='%m/%Y')


    switch2_q_sales = px.line( 
    switch2_df,
    x = 'Quarter Dates',
    y = 'Total Sales (Millions)',
    title = 'Switch 2 Hardware Sales by Quarter:',
    markers = True,
    template = 'plotly_white',
    color_discrete_sequence= ['#1F51FF']
)

    switch2_q_sales.update_layout(
        paper_bgcolor='#9F6066',
        plot_bgcolor='#e5e5e5',
        
        # Title font settings
        title=dict(
            text='Switch 2 Hardware Sales by Quarter:',
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

    st.plotly_chart(switch2_q_sales, use_container_width=True)    


    total_sales = switch2_df['Total Sales (Millions)'].sum().round(4)

    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Result:</u></h>", unsafe_allow_html=True)
    st.markdown(f"**Conclusion:** By the end of June 2028, the <strong>Switch 2 is expected to have sold {total_sales} million units.</strong> Since the console has not seen an original game from its most popular IPs (Smash Bros, Mario, Zelda), and the continued support/success of the system, these sales numbers are likely not far off from the truth.", unsafe_allow_html=True)

# ------------------------------------------ Fun Facts: -----------------------------------------------------
st.markdown("<h3 style='text-align: center;'><u>Switch 2 Fun Facts:</u></h3>", unsafe_allow_html=True)
st.markdown("""<h5 style='text-align: center;'>
1. The Switch 2 is the fastest-selling console ever, selling 3.5 million units in four days. 
 </h5>""", unsafe_allow_html=True)

st.markdown("""<h5 style='text-align: center;'>
2. It may be the only mainstream gaming console to continue producing physical media. 
 </h5>""", unsafe_allow_html=True)

st.markdown("""<h5 style='text-align: center;'>
3. The Switch 2’s 
<a href="https://www.mariowiki.com/Mario_Kart_World" target="_blank" style="color: #1D1D1D; text-decoration: none;"><strong>Mario Kart World</strong></a>,
 is the first game in the modern era to be priced at $80.
</h5>""", unsafe_allow_html=True)
st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="primary", use_container_width=True):
        st.switch_page("pages/testing.py")