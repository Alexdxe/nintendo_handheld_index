#Packages


#General Packages: 
import pandas as pd
import numpy as np
from scipy import stats

#Statistical Testing:
import statsmodels
import pingouin as pg
import sklearn
import scikit_posthocs as sp


#Machine Learning::
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
    background-color: #1515A5 !important;
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


# -------------------------------------- Introduction -------------------------------------------------
st.markdown("<h2 style='text-align: center;'>Overall Testing</h2>", unsafe_allow_html=True)
st.write("")
with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Overview</h3>", unsafe_allow_html=True)
    st.markdown("One of my favorite ways to stay updated with all Nintendo news is through content creators. The way many would review games with analogies, skits, and critques gravitated me to not only the subject itself, but " \
        "their mindset. To be curious, ask questions, and look at Nintendo with a critical eye. It's partially inspiration for this project, and many of the research questions I've asked in every webpage.")

    st.markdown("<strong>This page aims to answer hypotheses I've had since following Nintendo.</strong> All tests will use non-parametric methods, as the small sample size of the data means we’ll need to use “unconventional” testing techniques. <strong>All questions will be related to a Nintendo handheld’s sales and performance.</strong>", unsafe_allow_html=True)

   
    #Photo of Consoles: 
    photo = Image.open("photos/mario.jpeg")

    #Centering the Image: 
    left, center, right = st.columns([1, 5, 1])
    with center:
        st.image(photo, width=500)

    #Caption:
    st.markdown("<h6 style='text-align: center;'> <i>Image of Mario and Friends, Nintendo's most iconic franchise. </i></h6>", unsafe_allow_html=True)


# ------------------------------------------ Statistical Testing: -----------------------------------------------------

# ------------------------------------------ Test #1: -----------------------------------------------------


with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Does Nintendo Profit Across All Regions Equally?</h3>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h5><u>Summary:</u></h5>", unsafe_allow_html=True)
    st.markdown("Nintendo has released 6 generations of handhelds across various regions. While some argue Nintendo puts it's Japanese fans first, others would point to the company's continued success in the West as their biggest market. When you also consider the rest of the world (Europe, Australia, etc) has shown support for Nintendo, it becomes increasing more difficult to pinpoint if the company makes significantly more money from one specific region.")
    st.write("")

    #image:
    image = Image.open("photos/nintendo_europe.jpeg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(image, width=300)
    
    st.markdown("<h6 style='text-align: center;'> <i>Image of Nintendo of Europe, a region Nintendo started accounting seperate sales for in 2017.</i></h6>", unsafe_allow_html=True)
    

    st.markdown("<h5><u>Goal: </u></h5>", unsafe_allow_html=True)
    st.markdown("<strong> Therefore, I wanted to see if there was any substantial difference in sales across North America, Japan, and other regions.</strong> The test will observe sales from the Game Boy Advance to the Nintendo 3DS.", unsafe_allow_html=True)

    with st.expander("**The Nerdy stuff**"):
        st.markdown("We will be using <strong>Friedman's Test</strong>, a ranked, non-parametric test to see if all 3 regions' medians are roughly similar.",unsafe_allow_html=True)
        st.markdown("<strong>If we achieve a p-value of < 0.05, we may claim Nintendo's sales across all three regions are not the same, and that they differ signficantly.</strong>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Conducting Test:</u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Null Hypothesis:</u> Sales across all regions are similar. (p-value >= 0.05)</b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Alternative Hypothesis:</u> Sales differ significantly across regions. (p-value < 0.05)</b></h5>", unsafe_allow_html=True)
    st.divider()

   #Making a copy of our DF, deleting switch & Total Region:
    old_console_df = console_total_gdf[(console_total_gdf['Console'] != 'Switch') & 
    (console_total_gdf['Console'] != 'Switch 2') & 
    (console_total_gdf['Region'] != 'Total')]

    # Running the Test: 
    friedman_result = pg.friedman(
        data = old_console_df,
        dv = 'Total Sales',
        within = 'Region',
        subject = 'Console')

    st.markdown("<h5 style='text-align: center;'><u>Data Used:</u></h>", unsafe_allow_html=True)


    left, center, right = st.columns([0.5, 3, 0.5])
    with center:
        st.dataframe(old_console_df[['Console','Model', 'Region', 'Total Sales']], height = 400, hide_index=True)

    st.divider()

    
    st.markdown("<h5 style='text-align: center;'><u>Test Result:</u></h>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center;'><b>Friedman Test Statistic:</b> " + str(round(friedman_result['p_unc'].values[0], 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("**Meaning:** The p-value of ~0.105 would fail to reject our null hypothesis. This means there is <strong>not enough statistical evidence to prove the average sales of consoles across different regions differ significantly.</strong>", unsafe_allow_html=True)
    st.divider()


    st.markdown("<h5 style='text-align: center;'><u>Conclusion:</u></h>", unsafe_allow_html=True)
    st.markdown("Because we failed to reject our null hypothesis, this means <strong>Nintendo handheld sales do not differ across all three regions.</strong> While I expected North America to be the most profitable region, it seems <strong>Nintendo has been able to market itself to NA/Japan/the rest of the world fairly equally.</strong>", unsafe_allow_html=True)



# ------------------------------------------ Test #2: -----------------------------------------------------

st.divider()

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Does a Cheaper Console Yield Better Sales?</h3>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<h5><u>Summary:</u></h5>", unsafe_allow_html=True)
    st.markdown("When buying a console, one of the biggest concerns a consumer has is the price. Hardware is expensive, and if a potential buyer believes the price is not worth paying for, they may either wait for a cheaper model, or skip the generation entirely.", unsafe_allow_html=True)
    st.write("")

    #Console:
    console = Image.open("photos/2DS.jpeg") 
    left, center, right = st.columns([1.25, 2, 1])
    with center:
        st.image(console, width=250)
    
    st.markdown("<h6 style='text-align: center;'> <i>Image of the Nintendo 2DS, the cheapest listed handheld Nintendo offically sold ($80 in 2016).</i></h6>", unsafe_allow_html=True)

    st.markdown("<h5><u>Goal:</u></h5>", unsafe_allow_html=True)
    
    st.markdown("<strong>I wanted to see if there is a correlation between price and sales</strong> to observe if consumers are more willing to purchase a cheaper device.", unsafe_allow_html=True)

    with st.expander("**The Nerdy stuff**"):
        st.markdown("We will be using <strong>Spearman’s correlation</strong> to identify if there is a monotonic relationship between sales and price.", unsafe_allow_html=True)
        st.markdown("<strong>If there is a correlation of > 0.5, we can assume there is some-sort of relationship between the two factors. Additionally, if the p-value is < 0.05, we can conclude that the relationship is significant.</strong>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><b>Conducting Test: </b></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Null Hypothesis:</u> There is no monotonic relationship between price and sales. (p-value >= 0.05 & ρ < 0.5)</b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Alternative Hypothesis:</u> There is a monotonic relationship between price and sales. (p-value < 0.05 and/or ρ >= 0.5)</b></h5>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Data Used:</u></h>", unsafe_allow_html=True)

    left, center, right = st.columns([0.5,3,0.5])
    with center:
        st.dataframe(console_adjusted_df[['Console', 'Adjusted MSRP:', 'Total Sales']], height = 400, hide_index=True)

    st.divider()

    rho, p_val_rho = stats.spearmanr(console_adjusted_df["Adjusted MSRP:"], console_adjusted_df["Total Sales"]) 

    
    
    st.markdown("<h5 style='text-align: center;'><u>Test Result:</u></h>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center;'><b>Spearman Correlation:</b> " + str(round(rho, 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>p-value:</b> " + str(round(p_val_rho, 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("**Meaning:** The p-value of ~0. 62 would fail to reject our null hypothesis. <strong>This means there is not enough statistical evidence to prove that a cheaper console yields better sales.</strong> Additionally, the correlation value of 0.137 further indicates there is little to no monotonic relationship between both values.", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Conclusion:</u></h>", unsafe_allow_html=True)
    st.markdown("Because we failed to reject our null hypothesis, <strong>there is likely no correlation between price and sales.</strong> This is likely because consumers often consider other factors when purchasing a console, such as game libraries and unique hardware features.", unsafe_allow_html=True)



# ------------------------------------------ Test #3: -----------------------------------------------------

st.divider()

with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Does Nintendo First-Party Games Sell better Than 3rd-Party Software?:</h3>", unsafe_allow_html=True)
    st.markdown("<h5><u>Summary:</u></h5>", unsafe_allow_html=True)
    st.markdown("""When someone buys a Nintendo console, the first few games a user purchases are often first-party titles. Whether it’s the latest 
    <a href="https://en.wikipedia.org/wiki/Mario" target="_blank" style="color: #e7ceb9; text-decoration: none;"><strong>Mario</strong></a>, 
     game or 
    <a href="https://en.wikipedia.org/wiki/Pok%C3%A9mon" target="_blank" style="color: #e7ceb9; text-decoration: none;"><strong>Pokémon</strong></a> 
     generation, a handheld's first-party games define the console. However, many handhelds also have excellent 3rd-party support. Franchises like 
    <a href="https://en.wikipedia.org/wiki/Sonic_the_Hedgehog" target="_blank" style="color: #e7ceb9; text-decoration: none;"><strong>Sonic</strong></a>, 
    <a href="https://en.wikipedia.org/wiki/Resident_Evil" target="_blank" style="color: #e7ceb9; text-decoration: none;"><strong>Resident Evil</strong></a>, 
    and 
    <a href="https://en.wikipedia.org/wiki/Grand_Theft_Auto" target="_blank" style="color: #e7ceb9; text-decoration: none;"><strong>Grand Theft Auto</strong></a> 
     are all massive properties that helped push each of Nintendo’s handhelds' limits and libraries.""", unsafe_allow_html=True)
    st.write("")

    #Game:
    game = Image.open("photos/mario_sonic.jpeg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(game, width=300)
    
    st.markdown("<h6 style='text-align: center;'> <i>Image of the Mario and Sonic at the Olympic Games.</i></h6>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<h5><u>Goal:</u></h5>", unsafe_allow_html=True)
    st.markdown("<strong>Does Nintendo’s first-party games tend to sell better than it’s third party offerings?</strong> To do this, I will observe the top 250 games of each handheld (GBA-Switch) and observe whether or not Nintendo-published games sold more units than those not licensed by Nintendo.", unsafe_allow_html=True)
    with st.expander("**The Nerdy stuff**"):
        st.markdown("I will be using a <strong>two-sided Mann-Whitney U test </strong> to see if the two distributions differ. Using data scraped from VGChartz, we will observe whether games published by Nintendo sell better than those not.", unsafe_allow_html=True)
        st.markdown("<strong>If we achieve a p-value of < 0.05, we may claim Nintendo sells significantly more first-party games than third party ones.</strong>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<h4 style='text-align: center;'><b>Conducting Test: </b></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Null Hypothesis:</u> First-party games do not sell better than third-party games. (p-value >= 0.05)</b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Alternative Hypothesis:</u> First-party games sell better than third-party games. (p-value < 0.05)</b></h5>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Data Used:</u></h>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'><i>Scroll horizontally to see other columns</i></h6>", unsafe_allow_html=True)


    #Establishing both first/third party:
    valid_df = game_df.dropna(subset=['publisher', 'units_sold'])

    first_party = valid_df[valid_df['publisher'] == 'Nintendo']
    third_party = valid_df[valid_df['publisher'] != 'Nintendo']

    first_party_scores = valid_df[valid_df['publisher'] == 'Nintendo']['units_sold']
    third_party_scores = valid_df[valid_df['publisher'] != 'Nintendo']['units_sold']


    #Running 2-sided Mann Whitney:
    stat, p_value = stats.mannwhitneyu(
        first_party_scores, 
        third_party_scores, 
        alternative='two-sided'
    )
    
    left, center, right = st.columns([5, 0.5, 5])

    with left:
        st.markdown("<h5 style='text-align: center;'><u>First-Party Games:</u></h>", unsafe_allow_html=True)
        st.dataframe(first_party, height = 400, hide_index=True)


    with right:
        st.markdown("<h5 style='text-align: center;'><u>Third-Party Games:</u></h>", unsafe_allow_html=True)
        st.dataframe(third_party, height = 400, hide_index=True)
    
   
    st.divider()
    
    st.markdown("<h5 style='text-align: center;'><u>Test Result:</u></h>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center;'><b>Mann-Whitney U Test:</b> " + str(round(stat, 4)) +  "</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>p-value:</b> " + str(round(p_value, 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("**Meaning:**  The p-value of ~0 would reject our null hypothesis. This means <strong>there is enough statistical evidence to prove First party games tend to sell better on Nintendo handhelds.</strong>", unsafe_allow_html=True)


    st.divider()

    st.markdown("<h5 style='text-align: center;'><u>Conclusion:</u></h>", unsafe_allow_html=True)
    st.markdown("Because we rejected our null hypothesis, we can conclude that <strong>Nintendo's first-party games sell better than third-party games.</strong> Although 3rd-party games can help diversify a handheld's library, consumers are often buying it with the potential of playing the newest installment of their favorite Nintendo franchises.", unsafe_allow_html=True)
   



# ------------------------------------------ Test #4: -----------------------------------------------------

st.divider()


with st.container(border=True):
    st.markdown("<h3 style='text-align: center;'>Have Critic Scores Shifted Heavily From the GBA to Switch?</h3>", unsafe_allow_html=True)
    st.markdown("<h5><u>Summary:</u></h5>", unsafe_allow_html=True)
    st.markdown("With games becoming increasingly complex and more people playing games than ever, it’s safe to say there are different standards for each console generation. A game that was rated a 10/10 in 1990 differs in quality from what is expected of a 10/10 game in 2020.")
    st.write("")

    #Game:
    game = Image.open("photos/NSO.jpeg") 
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(game, width=300)
    
    st.markdown("<h6 style='text-align: center;'> <i>A photo of Nintendo Switch Online's consoles, allowing you to play previous generations of games.</i></h6>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<h5><u>Goal:</u></h5>", unsafe_allow_html=True)
    st.markdown("<strong>Have critics' review scores have changed from the GBA to the Switch generation?</strong> The goal is to see if there are points when critics changed their scoring criteria, and began awarding more/less points to a specific handheld.", unsafe_allow_html=True)
    with st.expander("**The Nerdy stuff**"):
        st.markdown("We will be using the <strong>Kruskal-Wallis H test</strong>, which identifies whether there are differences in scores amongst non-repeated subjects. We will also be using the same top 250 best-selling games used in the previous test.", unsafe_allow_html=True)
        st.markdown("<strong>If our p-value is < 0.05, we may conclude that Nintendo scores have differed across handheld generations. Additionally, if we reject the null hypothesis, we will run Dunn's Test to identify where game scores differed the most.</strong>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><u>Conducting Test: </u></h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Null Hypothesis:</u> There is no significant difference in critic scores between the GBA and Switch generations. (p-value >= 0.05)</b></h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b><u>Alternative Hypothesis:</u> There is a significant difference in critic scores between the GBA and Switch generations. (p-value < 0.05)</b></h5>", unsafe_allow_html=True)
    st.divider()

    gba_scores = game_df[game_df['console'] == 'GBA']['critic_score'].dropna()
    ds_scores = game_df[game_df['console'] == 'DS']['critic_score'].dropna()
    three_ds_scores = game_df[game_df['console'] == '3DS']['critic_score'].dropna()
    switch_scores = game_df[game_df['console'] == 'NS']['critic_score'].dropna()

    #Running Test:
    h_stat, p_value = stats.kruskal(gba_scores, ds_scores, three_ds_scores, switch_scores)


    st.markdown("<h5 style='text-align: center;'><u>Data Used:</u></h>", unsafe_allow_html=True)
    left, center, right = st.columns([0.5,3,0.5])
    with center: 
        copy_game_df = game_df[['console', 'game', 'critic_score']].reset_index(drop=True)
        updated_copy_game_df = copy_game_df[copy_game_df['console'] != 'GB'].dropna().reset_index(drop=True)
        st.dataframe(updated_copy_game_df, height = 400, hide_index=True)

    st.divider()
    
    
    st.markdown("<h5 style='text-align: center;'><u>Test Result:</u></h>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center;'><b>Kruskal-Wallis H-test:</b> " + str(round(h_stat, 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'><b>p-value:</b> " + str(round(p_value, 4)) + "</h5>", unsafe_allow_html=True)
    st.markdown("**Conclusion:** The p-value of ~0. 0001 would reject our null hypothesis. This means there is enough statistical evidence to prove that <strong>critic scores have shifted significantly from the GBA to Switch generations.</strong>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4 style='text-align: center;'><b>Further Analysis: </b></h4>", unsafe_allow_html=True)

    st.markdown("<h5><u>Summary:</u></h5>", unsafe_allow_html=True)
    st.markdown("To see which consoles differed the most, let's run a <strong>Dunn's Test</strong> to observe which pairs of handheld's p-values differ the most.", unsafe_allow_html=True)

    #Establishing a valid df:
    valid_df = game_df[game_df['console'].isin(['GBA', 'DS', '3DS', 'NS'])].dropna(subset=['critic_score'])

    # Run Dunn's test with Bonferroniadjustment
    dunn_matrix = sp.posthoc_dunn(
        valid_df, 
        val_col='critic_score', 
        group_col='console', 
        p_adjust='bonferroni'
    )

    st.markdown("<h5 style='text-align: center;'><u>Dunn's Test Result:</u></h>", unsafe_allow_html=True)

    left,center,right = st.columns([0.5, 3, 0.5])
    with center:
        st.dataframe(dunn_matrix, hide_index=True)

    st.markdown("**Final Conclusion:** The Dunn's test results show that there is a <strong>significant difference in the scores between the 3DS and Switch.</strong>", unsafe_allow_html=True)


    st.divider()
    st.markdown("<h5 style='text-align: center;'><u>Conclusion:</u></h>", unsafe_allow_html=True)
    st.markdown("Because we rejected the null hypothesis, we can conclude that <strong>critic scores have shifted overtime with console generations. More specifically, with the 3DS to Nintendo Switch era.</strong> This could be because the quality/rating for one generation of games was much higher than the other.", unsafe_allow_html=True)

st.write("")


col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("Return Home", type="primary", use_container_width=True):
        st.switch_page("Home.py")