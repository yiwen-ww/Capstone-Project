#!/usr/bin/env python
# coding: utf-8

# # Capstone Project - The Battle of Neighborhoods (Week 2)
# In this week, you will continue working on your capstone project. Please remember by the end of this week, you will need to submit the following:
# 
# #### A full report consisting of all of the following components :
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>
# 
# 1.  <a href="#item1">Introduction</a>
# 
# 2.  <a href="#item2">Data and the source of the data.</a>
# 
# 3.  <a href="#item3">Methodology section which represents the main component of the report where you discuss and describe any exploratory data analysis that you did, any inferential statistical testing that you performed, if any, and what machine learnings were used and why.
# </a>
# 
# 4.  <a href="#item4">Results section where you discuss the results.</a>
# 
# 5.  <a href="#item5">Discussion section where you discuss any observations you noted and any recommendations you can make based on the results.</a>  
# 
# 
# 
# 

# ##  Introduction 

# We explored New York and Toronto as well as clustering in the previous exercise.
# In New York, if someone is looking to open a restaurant, where would you recommend that they open it? 
# 
# This project will be of interest to those who want to open a restaurant or begin any other business which related to the similar factors like restaurant in New York.

# Before we get the data and start exploring it, let's download all the dependencies that we will need.
# 
# ## Data and the source of the data.
# 
# ### New York city geographical coordinates dataset
# 
# Load and explore the data
# 
# From the Wikipage given the List of postal codes of Canada :
# 
# wikilikn： https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M
# 
# For your convenience, I downloaded the files and placed it on the server, so you can simply run a `wget` command and access the data. So let's go ahead and do that.
# 

# In[74]:


#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
#!conda install -c anaconda beautiful-soup --yes
#pip install wordcloud
#!conda install -c conda-forge wordcloud==1.4.1 --yes
#!conda install -c anaconda seaborn -y
#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab

#!conda install -c conda-forge wget --yes


# In[96]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

import csv # implements classes to read and write tabular data in CSV form

#!conda install -c anaconda beautiful-soup --yes
from bs4 import BeautifulSoup  # package for parsing HTML and XML documents

from PIL import Image # converting images into arrays

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
# install wordcloud
#!conda install -c conda-forge wordcloud==1.4.1 --yes

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker

# notice: installing seaborn might takes a few minutes
#!conda install -c anaconda seaborn -y
import seaborn as sns

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

# import k-means from clustering stage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



print('Libraries imported.')


# downloaded the files and placed it on the server,run a wget command and access the data. 

# In[97]:


get_ipython().system("wget -q -O 'newyork_data.json' https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/labs/newyork_data.json")
print('Data downloaded!')


# In[7]:


#load the data.
with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)


# In[8]:


neighborhoods_data = newyork_data['features']


# In[9]:


neighborhoods_data[0]


# In[10]:


#Tranform the data into a pandas dataframe
# define the dataframe columns
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiate the dataframe
neighborhoods = pd.DataFrame(columns=column_names)


# In[11]:


neighborhoods


# In[12]:


#Then let's loop through the data and fill the dataframe one row at a time.
for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    neighborhoods = neighborhoods.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)


# In[13]:


neighborhoods.head()


# In[14]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)


# #### Use geopy library to get the latitude and longitude values of New York City.

# In[15]:


address = 'New York City, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# In[102]:


#Create a map of New York with neighborhoods superimposed on top
# create map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork


# In[103]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('Number of Neighborhood for each Borough in New York City')
#On x-axis
plt.xlabel('Borough', fontsize = 15)
#On y-axis
plt.ylabel('No.of Neighborhood', fontsize=15)
#giving a bar plot
neighborhoods.groupby('Borough')['Neighborhood'].count().plot(kind='bar')
#legend
plt.legend()
#displays the plot
plt.show()


# ###  Web scrapping of Population and Demographics data of New York city from Wikipedia
# 
# #### POPULATION DATA
# 
# Web scrapping of Population data from wikipedia page - https://en.wikipedia.org/wiki/New_York_City
# 
# 

# In[274]:


website_url = requests.get('https://en.wikipedia.org/wiki/Demographics_of_New_York_City').text
soup = BeautifulSoup(website_url,'lxml')
table = soup.find('table',{'class':'wikitable sortable'})
#print(soup.prettify())

headers = [header.text for header in table.find_all('th')]

table_rows = table.find_all('tr')        
rows = []
for row in table_rows:
   td = row.find_all('td')
   row = [row.text for row in td]
   rows.append(row)

with open('gener_info.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerow(headers)
   writer.writerows(row for row in rows if row)


# In[275]:


# load data from csv
Pop_data=pd.read_csv('gener_info.csv')
Pop_data.drop(Pop_data.columns[[7,8,9,10,11]], axis=1,inplace=True)
print('Data downloaded!')
Pop_data


# In[276]:


#Remove whitespaces and rename columns
Pop_data.columns = Pop_data.columns.str.replace(' ', '')
Pop_data.columns = Pop_data.columns.str.replace('\'','')
Pop_data.rename(columns={'Borough':'person_mi2','Density\n':'sq_km','Landarea\n':'sq_mi'}, inplace=True)
Pop_data


# In[277]:


Pop_data.rename(columns = {'NewYorkCitysfiveboroughsvte\n' : 'Borough',
                           'Jurisdiction\n':'County',
                           'Population\n':'population_2019', 
                           'sq_mi':'square_miles',
                           'sq_km':'square_km',
                           'GDP\n':'GDP_2012',
                           'persons/mi2':'persons_mi2',
                           'persons/km2\n':'persons_km2'
                          }, inplace=True)
Pop_data=Pop_data.drop(columns=['persons_mi2','persons_km2'])
#Pop_data.drop(labels=,axis=1,inplace=True)
Pop_data


# In[278]:


colname=Pop_data.columns.values.tolist()
print(colname)


# In[279]:


for i in colname:
    Pop_data[i]=Pop_data[i].replace(to_replace='\n', value='', regex=True)

Pop_data


# In[280]:


#Pop_data.loc[5:,].shift(periods=1, axis="columns")
Pop_data.loc[5:,:] = Pop_data.loc[2:,:].shift(1,axis=1)
Pop_data
#Pop_data


# In[281]:


# drop nan and lasz row ,because there is nothing useful 
Pop_data = Pop_data.dropna()
#Pop_data=Pop_data.drop(index =7,axis=0)
Pop_data


# 

# ####  DEMOGRAPHICS DATA
# 
# We will web scrap Demographics data from wikipedia page - https://en.wikipedia.org/wiki/Demographic_history_of_New_York_City

# In[302]:


website_url2 = requests.get('https://en.wikipedia.org/wiki/Demographic_history_of_New_York_City').text
soup2 = BeautifulSoup(website_url2,"html.parser")
tables=soup.find_all('tbody')
#tables
#soup2
table = soup.find('table',{'class':'wikitable sortable collapsible'})
#print(soup.prettify())

headers = [header.text for header in table.find_all('th')]

table_rows = table.find_all('tr')        
rows = []
for row in table_rows:
   td = row.find_all('td')
   row = [row.text for row in td]
   rows.append(row)

with open('NYC_DEMO.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerow(headers)
   writer.writerows(row for row in rows if row)


# In[311]:


Demo_data=pd.read_csv('NYC_DEMO.csv')
Demo_data=Demo_data[['Year','Population']]
#Demo_data


# #####  New York City ,Bronx,Brooklyn,Manhattan,Queens,Staten Island

# In[387]:


#Bronx
table_1=Demo_data[0:11]
table_1.rename(columns={'Population':'NeY_population'}, inplace=True)
#print(table_1)
#
table_2=Demo_data[12:23]
table_2.rename(columns={'Population':'Bronx_population'}, inplace=True)

table_3=Demo_data[24:35]
table_3.rename(columns={'Population':'Brooklyn_population'}, inplace=True)
#
table_4=Demo_data[36:47]
table_4.rename(columns={'Population':'Manhattan_population'}, inplace=True)
#
table_5=Demo_data[48:59]
table_5.rename(columns={'Population':'Queens_population'}, inplace=True)
#
table_6=Demo_data[60:71]
table_6.rename(columns={'Population':'stateni_population'}, inplace=True)
#table_1
popu_data=table_1.merge(table_2,on='Year',how='left')
popu_data=popu_data.merge(table_3,on='Year',how='left')
popu_data=popu_data.merge(table_4,on='Year',how='left')
popu_data=popu_data.merge(table_5,on='Year',how='left')                     
popu_data=popu_data.merge(table_6,on='Year',how='left')
popu_data = popu_data.stack().str.replace(',', '').unstack()
popu_data


# In[388]:


popu_data['Year']=pd.to_numeric(popu_data['Year'])
popu_data['NeY_population']= pd.to_numeric(popu_data['NeY_population'])
popu_data['Bronx_population']= pd.to_numeric(popu_data['Bronx_population'])
popu_data['Brooklyn_population']= pd.to_numeric(popu_data['Brooklyn_population'])
popu_data['Manhattan_population']= pd.to_numeric(popu_data['Manhattan_population'])
popu_data['Queens_population']= pd.to_numeric(popu_data['Queens_population'])
popu_data['stateni_population']= pd.to_numeric(popu_data['stateni_population'])

#popu_data.info()


# In[389]:


### plot population of each borough
plt.figure(figsize=(9,5), dpi = 100)
plt.title('population of each borough in 1900-2000')

plt.plot(popu_data['Year'], popu_data['NeY_population'], color="blue",label='new York')
plt.plot(popu_data['Year'], popu_data['Bronx_population'], color="red",label='the bronx')
plt.plot(popu_data['Year'], popu_data['Brooklyn_population'], color="green",label='brooklyn')
plt.plot(popu_data['Year'], popu_data['Manhattan_population'], color="grey",label='manhattan')
plt.plot(popu_data['Year'], popu_data['Queens_population'], color="black", label='queens')
plt.plot(popu_data['Year'], popu_data['stateni_population'], color="y",label='staten island' )

plt.xlabel('Year')
plt.ylabel('Population')

plt.legend()
plt.show()


# #### Summary
# 1. Brooklyn has the largest population in 2019
# 
# 2. Queens had the highest GDP in 2012 and most neighbours .
# 
# 3. Manhattan has the highest population density
# 
# 4. Between 1900 and 2000, Queens has the highest population growth rate, and Berkeley has the largest population most of the time.
# 
# Conclusion: Based on the above summary, Queens has better investment potential. 

# ###  Farmers Market dataset
# 
# Website-https://www.grownyc.org/greenmarketco/foodbox

# In[502]:


# Data from website - https://data.cityofnewyork.us/dataset/DOHMH-Farmers-Markets-and-Food-Boxes/8vwk-6iz2
import os
os.getcwd()
os.listdir(os.getcwd())


# In[505]:


FM_NYC= pd.read_csv("DOHMH_Farmers_Markets.csv")


# In[506]:


FM_NYC.head()


# ## A Segmenting and Clustering Neighborhoods 

# In[515]:


#neighborhoods
neighborhoods.head()


# In[516]:


neighborhoods['Borough'].value_counts().to_frame()


# In[518]:


neighborhoods.shape


# In[519]:


neighborhoods.isnull().sum()


# In[594]:


BM_Geo = neighborhoods.loc[(neighborhoods['Borough'] == 'Brooklyn')|(neighborhoods['Borough'] == 'Manhattan')|(neighborhoods['Borough'] == 'Bronx')|(neighborhoods['Borough'] == 'Queens')|(neighborhoods['Borough'] == 'Staten Island')]
BM_Geo = BM_Geo.reset_index(drop=True)
BM_Geo.head()


# In[595]:


BM_Geo.shape


# In[596]:


import time
start_time = time.time()

address = 'New York City, NY'

geolocator = Nominatim(user_agent="Jupyter")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))

print("--- %s seconds ---" % round((time.time() - start_time), 2))


# In[592]:


# create map
map_bm = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
rainbow = ['r','y','b','c','m']

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(BQS_merged['Latitude'], BQS_merged['Longitude'], BQS_merged['Neighborhood'], BQS_merged['Cluster_Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:


BQS_venues = getNearbyVenues(names=BQS_Geo['Neighborhood'],
                                  latitudes=BQS_Geo['Latitude'],
                                  longitudes=BQS_Geo['Longitude'],
                                  LIMIT=200)

print('The "BQS_venues" dataframe has {} venues and {} unique venue types.'.format(
      len(BQS_venues['Venue Category']),
      len(BQS_venues['Venue Category'].unique())))

BQS_venues.to_csv('BQS_venues.csv', sep=',', encoding='UTF8')
BQS_venues.head()

