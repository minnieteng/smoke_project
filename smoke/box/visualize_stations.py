import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
# mpl_toolkits.__path__.append('/home/peterw/ml_env/lib/python3.6/site-packages/mpl_toolkits')
from mpl_toolkits.basemap import Basemap

if __name__ == "__main__":
    # Load bc_air_monitoring_stations.csv and PM25 as a dataframe
    stations_df = pd.read_csv('../pm25_parser/bc_air_monitoring_stations.csv',
                            header=0)
    pm25_df = pd.read_csv('../pm25_parser/PM25.csv',
                            header=0)
    

    # The columns we care about in each dataframe
    cols_stations = ['EMS_ID', 'LAT', 'LONG']
    cols_pm25 = ['DATE_PST', 'DATE', 'TIME', 'EMS_ID', 'RAW_VALUE']

    stations_df = stations_df[cols_stations]
    pm25_df = pm25_df[cols_pm25]

    # Keep track of which stations have observations this year
    all_ems_ids = set(pm25_df['EMS_ID'].to_numpy())

    stations_df_relevant = stations_df[stations_df['EMS_ID'].isin(all_ems_ids)]

    # Set the dimension of the figure
    my_dpi=96
    plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)

    # Make the background map
    # m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)  # full map
    m = Basemap(llcrnrlon=-140, llcrnrlat=47, urcrnrlon=-115, urcrnrlat=60)
    m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m.fillcontinents(color='grey', alpha=0.3)
    m.drawcoastlines(linewidth=0.1, color="white")

    # Add a point per position
    m.scatter(stations_df_relevant['LONG'], stations_df_relevant['LAT'],
                alpha=0.4, cmap="Set1")

    plt.savefig('monitoring_stations.png')


    # Get the station which is furthest north
    lats = stations_df_relevant['LAT'].to_numpy()
    lons = stations_df_relevant['LONG'].to_numpy()
    arg_max = np.argmax(lats)
    print('Northern-most Station: %.5f, %.5f' % (lats[arg_max], lons[arg_max]))
