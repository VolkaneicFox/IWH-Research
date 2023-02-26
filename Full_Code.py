# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:06:00 2022

@author: SUKANYA
"""
#%% PART 0: Paths and Packages
dirpath = "D:\\WorkStuff\\IWH-Halle\\Paper1\\"
collpath = dirpath+"datacollected"

#standard imports
import pandas as pd; import numpy as np; import scipy as sc
import sklearn.utils

#mapping imports
import geopandas as gpd

#plotting and panel import
import matplotlib.pyplot as plt 
import panel as pn

#imports for spatial clustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

#%% PART 1: Gathering Climate Data from the CDC

from urllib.request import Request, urlopen
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests, os
import seaborn as sns

def keyfromval(val,dic):
    for item in dic:
        if val in [int(x) for x in dic[item]]:
            c = item
            break
    return c

def newstr(a,b):
    return str(a)+"-"+str(b)

def concat_cols(col_a, col_b,sep="-"):
    col_a,col_b = list(col_a),list(col_b)
    return [str(col_a[i])+sep+str(col_b[i]) for i in range(len(col_a))]

page_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"


station_list = "KL_Tageswerte_Beschreibung_Stationen.txt"   
    
page = Request(page_url)
html_page = urlopen(page)

soup = BeautifulSoup(html_page,"html.parser")

allfiles = soup.get_text()
catalogue = [i.split(" ")[0] for i in allfiles.split("\n") if "zip" in i]

station_page = requests.get(page_url+station_list)
with open(collpath+"\\climate data\\"+"Station_IDs.csv","wb") as op_file:
    op_file.write(station_page.content)
op_file.close()

with open(collpath+"\\climate data\\Station_IDs.csv","r") as op_file:
    txt = op_file.readlines()
op_file.close()


station_dict = {}
for line in txt:
    feats = [x for x in line.split("|") if x!=""]
    s_name, s_id = feats[-1], feats[0]
    if s_id.isdigit():
        if s_name in station_dict.keys():
            station_dict[s_name].append(s_id)
    
        else:
            station_dict[s_name] = []
            station_dict[s_name].append(s_id)
regional_path = "D:\\WorkStuff\\IWH-Halle\\Paper1\\data_temp_store"
for region in station_dict.keys():
    region = region.strip()
    with open(regional_path+region+".txt","w") as r:
        pass
    r.close()
    

from zipfile import ZipFile
from progressbar import ProgressBar

pbar = ProgressBar()


for item in pbar(catalogue):
    file_address = page_url+item
    r = requests.get(file_address)
    with open(item,"wb") as output_file:
        output_file.write(r.content)
    output_file.close()
    
    
    with ZipFile(item,"r") as zip:
        files = zip.namelist()
        for member in files:
            if "produkt" in member:
                give_data = member
                zip.extract(member,"D:\WorkStuff\IWH-Halle\Paper1\data_temp_store")
            if "Geographie" in member:
                give_name = member
                zip.extract(member,"D:\WorkStuff\IWH-Halle\Paper1\data_temp_store")

    with open("D:\WorkStuff\IWH-Halle\Paper1\data_temp_store\\"+give_name,"r") as name_giver:
        txt = name_giver.readlines()
    name_giver.close()
    
    id_temp = int(txt[1].split(";")[0])
    region = keyfromval(id_temp,station_dict).strip()+".txt"
    give_data = "D:\WorkStuff\IWH-Halle\Paper1\data_temp_store\\"+give_data
    
    with open(regional_path+region,"a") as outfile:
        with open(give_data) as infile:
            y = infile.read()

        outfile.write(y)
    
    outfile.close()
    infile.close()
    os.remove("D:\WorkStuff\IWH-Halle\Paper1\data_temp_store\\"+give_name)
    os.remove(give_data)
    os.remove(item)

## Raw data compiled and stored ##
region_dict = station_dict


header_dict = {"STATIONS_ID":"STATIONS_ID","MESS_DATUM":"DATE","QN_3":"quality auto control",
               "FX":"MAX WIND GUST (m/s)","FM":"MEAN WIND GUST (m/s)",
               "QN_4":"QN_4","RSK":"PRECIPITATION HEIGHT (mm)",
               "RSKF":"PRECIPITATION FORM","SDK":"SUN HOURS",
               "SHK_TAG":"SNOW DEPTH (cm)","NM":"DAILY MEAN CLOUD COVER (1/8)",
               "VPM": "DAILY MEAN VAPOUR PRESSURE (hPa)",
               "PM":"DAILY MEAN PRESSURE","TMK":"DAILY MEAN TEMPERATURE",
               "UPM": "DAILY MEAN RELATIVE HUMIDITY","TXK":"DAILY MAX TEMP",
               "TNK":"DAILY MIN TEMP","TGK":"DAILY MIN 5cm"}
precipitation_dict = {"0":"No Precipitation","1":"Only Rain",
                      "4":"Unidentified Form of Precipitation","6":"Only Rain or Liquid Precipitation",
                      "7":"Only Snow or Solid Precipitation","8":"Rain and Snow","9":"ERROR"}    


temp_path =  "D:\\WorkStuff\\IWH-Halle\\Paper1\\data_temp_store\\"

for region in region_dict.keys():
    region = region.strip()
    file_name = temp_path+region+".txt"
    with open(file_name,"r") as tempfile:
        temp = tempfile.readlines()
    tempfile.close()
    for i in header_dict:
        temp[0].replace(i,header_dict[i])
        
#%% PART 2: Creating the List of Relevant operational Stations

'''
Operational stations post 1995. The start date is from 1991. The data is 
daily data. 
'''
idspath= "\\climate data\\ReadyData\\stations.txt"
with open(collpath+idspath, "r") as file:
    infile = file.readlines()
file.close()
stations91 = [infile[0]]+[i for i in infile[1:] if int(i.split("|")[2])>19910000 & int(i.split("|")[1])>=19910000 ]
with open(collpath+"\\climate data\\ReadyData\\stations91.txt","w") as file:
    file.write('\n'.join(stations91))
file.close()

#%% PART 3: Importing Station Data and Parsing Date
def num2date(date):
    date = str(date)
    return pd.to_datetime(date[0:4]+"-"+date[4:6]+"-"+date[6:]).date()
stations91 = pd.read_csv(collpath+"\\climate data\\ReadyData\\stations95.txt",sep="|")

#stationheaders = ['Stations_id', 'von_datum', 'bis_datum', 'Stationshoehe', 'geoBreite',
#       'geoLaenge', 'Stationsname', 'Bundesland']

bundes_dict = dict.fromkeys(list(set(stations91.Bundesland))) #making the dictionary of stations
for bund in bundes_dict:
    bundes_dict[bund] = list(stations91.Stations_id[stations91.Bundesland==bund])

stations91[["von_datum","bis_datum"]]=stations91[["von_datum","bis_datum"]].applymap(num2date)

#%% PART 5: Clustering the Stations for each Year based on Location Data
'''
Dimensionality reduction precedes clustering. The actual data set has daily data
over thirty years for a variable number of stations for each year. Thus, the number
of stations need to be reduced for each year.

The chosen algorithm is ** DBSCAN SPATIAL CLUSTERING ** , as shown here: 
    https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/

The output will include the centroid of each cluster for each year and then 
this will be added to a new series of columns to dummydf, as [1993c,1994c...] 
which will indicate the centroid for each cluster for each year. Outliers will 
also be considered.
'''


#%% PART 4: Mapping the Stations over time
'''
creating a map of the stations over time, from 1991 and then update the stations

'''


dummyyears = np.arange(1993,2023,step=1)
dummydf = stations91
N = dummydf.shape[0]
dummydf[list(dummyyears)] = 0
dummydf[[str(i)+'cid' for i in list(dummyyears)]] = np.nan
'''
NOTE: to have columns with cluster ids initialsed to np.nan
'''
stationheaders = ['Stations_id', 'von_datum', 'bis_datum', 'Stationshoehe', 'geoBreite',
       'geoLaenge', 'Stationsname', 'Bundesland']


for ix in range(dummydf.shape[0]):
    years = np.arange(dummydf.loc[ix,"von_datum"].year,dummydf.loc[ix,"bis_datum"].year+1)
    dummydf.loc[ix,list(years)] = 1

def get_epsilon_list(df,yy):
    cdf = dummydf[['geoLaenge','geoBreite']+[yy]][dummydf[yy]==1]
    X= cdf[['geoLaenge','geoBreite']]
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances,indices = nbrs.kneighbors(X)
    epsilons = np.sort(distances[:,1],axis=0)
    return list(epsilons)

eps_dict = dict.fromkeys(list(dummyyears))
for yy in dummyyears:
    eps_dict[yy] = get_epsilon_list(dummydf, yy) #dictionary of epsilon values for each year
    
sklearn.utils.check_random_state(1000)
def DBCluster(df,yy,eps):
    params = [eps/6371,5]
    cdf = df[["Stations_id","geoLaenge","geoBreite",yy]][df[yy]==1]
     #attempted optimate params
    db = DBSCAN(eps=params[0],min_samples=params[1],algorithm="ball_tree",metric="haversine").fit(np.radians(cdf[["geoLaenge","geoBreite"]]))
    core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    cdf["Cluster_id"] = db.labels_
    return cdf

DB_dict = {}
for yy in dummyyears:
    #print(yy)
    DB_dict[yy]=DBCluster(dummydf[dummydf[yy]==1],yy,75)
DB_dict_1 = DB_dict
for yy in dummyyears:
    df = DB_dict[yy]
    nonoise_df = df[df["Cluster_id"]>=0]
    large_clusters = [ix for ix in set(nonoise_df["Cluster_id"]) if 
                      nonoise_df[nonoise_df["Cluster_id"]==ix].shape[0]>0.3*df.shape[0]]
    for ix in large_clusters:
        cluster_select = nonoise_df[nonoise_df["Cluster_id"]==ix]
        db = DBSCAN(eps=50/6371,min_samples=5,algorithm="ball_tree",metric="haversine").fit(np.radians(cluster_select[["geoLaenge","geoBreite"]]))
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_select["Cluster_id"+str(ix)] = db.labels_
        df = pd.merge(df,cluster_select[["Stations_id","Cluster_id"+str(ix)]],on=["Stations_id"],how="outer")
    
    DB_dict_1[yy] = df

   
def joinCluster(x):
    f = ""
    for i in x:
        if i == -1:
            f=-1
        elif  np.isnan(i)==False: 
            f =f + str(int(i))
    return int(f)


DB_dict_2 = {}
for yy in DB_dict_1:
    df = DB_dict_1[yy].reset_index()    
    clus_cols = [x for x in list(df.columns) if "Cluster" in str(x)]
    for row in range(df.shape[0]):
        ids = df.loc[row,clus_cols]
        df.loc[row,"F"] = joinCluster(ids)
    DB_dict_2[yy] = df
        
''' updating code: if there is too much noise, then increase round1 epsilon'''
DB_dict_noisy = {}
DB_dict={}
for yy in dummyyears:
    #print(yy)
    clusdf=DBCluster(dummydf[dummydf[yy]==1],yy,75)
    if clusdf[clusdf["Cluster_id"]==-1].shape[0] > 0.35*clusdf.shape[0]:
        DB_dict_noisy[yy] = clusdf
    else:
        DB_dict[yy] = clusdf


DB_dict_1 = {}
for yy in DB_dict :
    df = DB_dict[yy]
    nonoise_df = df[df["Cluster_id"]>=0]
    large_clusters = [ix for ix in set(nonoise_df["Cluster_id"]) if
                      nonoise_df[nonoise_df["Cluster_id"]==ix].shape[0]>0.3*df.shape[0]]
    for ix in large_clusters:
        cluster_select = nonoise_df[nonoise_df["Cluster_id"]==ix]
        db = DBSCAN(eps=50/6371,min_samples=5,algorithm="ball_tree",metric="haversine").fit(np.radians(cluster_select[["geoLaenge","geoBreite"]]))
        core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_select["Cluster_id"+str(ix)] = db.labels_
        df = pd.merge(df,cluster_select[["Stations_id","Cluster_id"+str(ix)]],on=["Stations_id"],how="outer")
    
    DB_dict_1[yy] = df

   
def joinCluster(x):
    f = ""
    for i in x:
        if i == -1:
            f="-1"
        elif  np.isnan(i)==False: 
            f =f + str(int(i))+"."
    return f


DB_dict_2 = {}
for yy in DB_dict_1:
    df = DB_dict_1[yy].reset_index()    
    clus_cols = [x for x in list(df.columns) if "Cluster" in str(x)]
    for row in range(df.shape[0]):
        ids = df.loc[row,clus_cols]
        df.loc[row,"F"] = joinCluster(ids)
    DB_dict_2[yy] = df
        
DB_dict_noisy1 = {}
for yy in DB_dict_2:
    clusdf = DB_dict_2[yy]
    if clusdf[clusdf["F"]==-1].shape[0] > 0.35*clusdf.shape[0]:
        DB_dict_noisy1[yy] = clusdf

for yy in DB_dict_noisy1:
    df = pd.DataFrame()
    clusdf = DB_dict_noisy1[yy]
    cdf = clusdf[clusdf["F"]==-1]
    db = DBSCAN(eps=100/6371,min_samples=5,algorithm="ball_tree",metric="haversine").fit(np.radians(cdf[["geoLaenge","geoBreite"]]))
    core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    cdf["Cluster_id"+str(-1)] = db.labels_
    df = pd.merge(clusdf,cdf[["Stations_id","Cluster_id"+str(-1)]],on=["Stations_id"],how="outer")
    DB_dict_noisy1[yy] = df


'''
For every year, this can generate the cluster ID of every weather station, then to find the centroid 
of the cluster for every year. This will reduce the number of stations we have to monitor each year.
'''



'''
Optimal Paramters for DBScan: The epsilon parameter for DBScan needs to be optimized. Each year 
will have its own epsilon and min_samples. For now, the min_samples is the natural log of the number
of operational stations for that year. 
The optimal epsilon is derived from the nearest neighbour algorithm. The output will be a dictionary
with keys = years, and the epsilon will be the value. The input will  be the data of each year with 
the geo-coordinates and station ID, and the min_samples, as this is the k-neighbours allowed. 
'''

    
'''
eps <-- the smaller the epsilon, the tighter the neighbourhood; it is the maximum distance
        per 6371km that two points can be from each other
min_samples <-- the minimum number of neighbours a point must have to be a 
                core_point
'''

shpfile = dirpath+"\\mapdata\\vg2500_bld.shp"
DEmap = gpd.read_file(shpfile)

import seaborn as sns
#sns.set_style("darkgrid")

def stationsplot(yy):
    df = DB_dict_2[yy]
    fig,ax = plt.subplots(figsize=(15,15))
    DEmap.plot(ax=ax,color="grey",alpha=0.3)
    sns.color_palette("tab10")
    sns.scatterplot(x="geoLaenge",y="geoBreite",data=df[df["F"]!=-1],hue="F")
    sns.scatterplot(x="geoLaenge",y="geoBreite",data=df[df["F"]==-1],marker="x",color="black")
    plt.title("Weather Stations in " +str(yy))
    #plt.show()
    plt.close()
    #add the cluster station for each year
    #ax.scatter()
    return fig

figs_dict = {}
for yy in DB_dict:
    figs_dict[yy]=stationsplot(yy)
'''
The data on weather stations provides the date that each station was operational. The objective 
is to create a slider widget that shows the number of stations throughout Germany, every year. 
The slider will be the year. Create a dataframe with the coloumns - StationID, Station_Name, 
Station_Lat, Station_Long and [1991, 1992, 1993,..., 2021]. The rows will remain the stations. 
Then there is a dummy for every year that a station has been operational.

#ISSUE: Station set up at the end of the year
'''
#stations95 = stations95[stations95["von_datum"]>=pd.to_datetime("1991-01-01")]



    
    
pn.extension()
years = pn.widgets.IntSlider(name="YEAR",value=1995,start=1993,end=2022,step=1)
interact = pn.bind(stationsplot,yy=years)

first_map = pn.Column(years,interact)
first_map.show()

#%% Deriving the histories for station clusters

df2022 = DB_dict_2[2022].copy()
df2022["2022_Cluster"] = df2022["F"]
df2022.drop(["F","index",2022,"Cluster_id","Cluster_id3"],axis=1,inplace=True)

for yy in DB_dict_2:
    if yy==2022:
        continue
    else:
        tdf = DB_dict_2[yy].copy()
        tdf[str(yy)+"_Cluster"] = tdf["F"]
        tdf = tdf[["Stations_id",str(yy)+"_Cluster"]]
        df2022 = pd.merge(df2022,tdf[["Stations_id",str(yy)+"_Cluster"]],on="Stations_id",how="outer")

for yy in DB_dict_noisy:
    tdf = DB_dict_noisy[yy].copy()
    tdf[str(yy)+"_Cluster"] = tdf["Cluster_id"]
    df2022 = pd.merge(df2022,tdf[["Stations_id",str(yy)+"_Cluster"]],on="Stations_id",how="outer")
    
for yy in DB_dict_noisy1:
    tdf = DB_dict_noisy1[yy].copy()
    tdf[str(yy)+"_Cluster"] = np.nan
    clus_cols = [y for y in list(tdf.columns) if "cluster" in str(y).lower()]
    for row in range(tdf.shape[0]):
        tdf.loc[row,str(yy)+"_Cluster"] = joinCluster(tdf.loc[row,clus_cols+["F"]])
    df2022 = pd.merge(df2022,tdf[["Stations_id",str(yy)+"_Cluster"]],on="Stations_id",how="outer")

colss = ["Stations_id","geoBreite","geoLaenge"]+[str(2022-i)+"_Cluster" for i in range(30)]
dfhistory = df2022[colss]

finalhistory = pd.merge(dfhistory,stations91[["Stations_id","geoLaenge","geoBreite"]],on="Stations_id",how="outer")
'''
beginning with 2022, the stations belong to clusters identified in column F
moving to 2021, the stations in 2021 are matched to the clusters for the stations
in 2022. 

keep columns: Stations_id, geoLaenge, geoBreite, Year_clusterID

The three dictionaries we are combining:
    1. DB_dict_2 - The final cluster is in F
    2. DB_dict_noisy - The final cluster is in Cluster_id
    3. DB_dict_noisy1 - ....pending.... 
for DB_dict_2, the final cluster is in F
'''

for yy in [1993,1995,2000,2005,2010,2015,2020,2022]:
    fig,ax = plt.subplots(figsize=(15,15))
    DEmap.plot(ax=ax,color="grey",alpha=0.3)
    sns.color_palette("tab10")
    sns.scatterplot(x="geoLaenge_y",y="geoBreite_y",data=finalhistory,hue=str(yy)+"_Cluster")
    sns.scatterplot(x="geoLaenge_y",y="geoBreite_y",data=finalhistory[finalhistory[str(yy)+"_Cluster"]==-1],marker="x",color="black")
    plt.title("Weather Stations in " +str(yy))

#%% MERGING WEATHER DATA WITH THE STATION HISTORY

import pandas as pd

station_history = pd.read_csv(collpath+"\\climate data\\ReadyData\\clusterHistory.csv")

"""
creating the cluster history: beginning with the clusters in 2022, for each cluster get the station
with the longest history. 


"""
clusters_cols = [str(1993+y)+"_Cluster" for y in range(30)]
station_history["Obs_count"] = station_history[clusters_cols].count(axis=1)

clusterv2 = station_history[["Stations_id","geoLaenge_y","geoBreite_y"]]
db_eps50 = DBSCAN(eps=50/6371,min_samples=5,algorithm="ball_tree",metric="haversine").fit(np.radians(clusterv2[["geoLaenge_y","geoBreite_y"]]))
core_samples_mask = np.zeros_like(db_eps50.labels_,dtype=bool)
core_samples_mask[db_eps50.core_sample_indices_] = True
clusterv2["Cluster_id"] = db_eps50.labels_

df=(clusterv2.groupby(by="Cluster_id").count()["Stations_id"]>=0.10*clusterv2.shape[0]).reset_index()
clusters = df[df["Stations_id"]==True]["Cluster_id"]
clusters = [x for x in list(clusters) if x>=0]

clusterv2["Cluster_Final"] = clusterv2["Cluster_id"]

largecluster = clusterv2[clusterv2["Cluster_id"].isin(clusters)]
db_eps30 = DBSCAN(eps=35/6371,min_samples=5,algorithm="ball_tree",metric="haversine").fit(np.radians(largecluster[["geoLaenge_y","geoBreite_y"]]))
core_samples_mask = np.zeros_like(db_eps30.labels_,dtype=bool)
core_samples_mask[db_eps30.core_sample_indices_] = True
largecluster["New_Cluster"] = db_eps30.labels_

largecluster["Cluster_id"] = largecluster["Cluster_id"]+(largecluster["New_Cluster"]-1)*0.01


clusterv2 = pd.merge(clusterv2,largecluster[["Stations_id","Cluster_id"]],on="Stations_id",how="outer")
clusterv2["Cluster_id_y"].fillna(clusterv2["Cluster_id_x"],inplace=True)
clusterv2 = clusterv2[["Stations_id","geoLaenge_y","geoBreite_y","Cluster_id_y"]]

dummies = station_history[clusters_cols+["Stations_id","geoLaenge_y","geoBreite_y"]]
