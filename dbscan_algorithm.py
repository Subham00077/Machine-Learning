
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('weather_stations.csv')
df.head()


df.dropna(inplace=True)



df.info()


features = df[[
    'Data.Temperature.Avg Temp',
    'Data.Temperature.Max Temp',
    'Data.Temperature.Min Temp',
    'Data.Wind.Speed',
    'Data.Precipitation'
]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



dbscan = DBSCAN(eps=1.3, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)




df = df.loc[features.index]  
df['Cluster'] = clusters




plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['Data.Temperature.Avg Temp'],
    y=df['Data.Temperature.Max Temp'],
    hue=df['Cluster'],
    palette='tab10',
    style=(df['Cluster'] == -1),
    s=100
)
plt.title('DBSCAN Clustering of Weather Stations')
plt.xlabel('Average Temperature')
plt.ylabel('Max Temperature')
plt.legend(title='Cluster')
plt.show()





outliers = df[df['Cluster'] == -1]
plt.figure(figsize=(10, 6))
plt.scatter(
    outliers['Data.Temperature.Avg Temp'],
    outliers['Data.Temperature.Max Temp'],
    c='red',
    label='Outliers',
    s=100
)
plt.xlabel('Average Temperature')
plt.ylabel('Max Temperature')
plt.title('Outlier Weather Stations Detected by DBSCAN')
plt.legend()
plt.show()


