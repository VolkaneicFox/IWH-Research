# IWH-Research
Datasets used in papers written in IWH Halle (Saale)

[DATA_FILE]AMR_coords: The labour market regions in DE with their capital/major city and the respective geo-coordinates

[DATA_FILE]stations91v2.csv: The directory of weather stations with the station ID and coordinates of every weather station. 

[DATA_FILE]Stations_AMR.csv: The directory of weather stations mapped to the respective AMR with key and AMR name

The weather stations are matched via location proximity to the major city in each AMR and then the AMR_key is assigned to each weather station.

------------------------------------------------------------------------------------------------------------------------
MethodologyV1: 
--------------

[DATA_FILE]AMR_panel_dims: The number of observations under temperature, precipitation and wind, for each AMR

Version#1 : Unbalanced panels, the original data from 19910101

There are several stations in each AMR. -> Take the daily mean over all stations within one AMR -> generates the daily 

temperature for each AMR

----------------

[DATA_FILE]winddfv1.csv [2311972 rows x 4 columns]: The unbalanced panel data of AMR against wind data.

[DATA_FILE]precdfv1.csv [1930156 rows x 4 columns]: The unbalanced panel data of AMR against precipitation data.

[DATA_FILE]temperaturedfv1.csv [1794578 rows x 7 columns]: The unbalanced panel data of AMR against temperature data.

------------------------------------------------------------------------------------------------------------------------
