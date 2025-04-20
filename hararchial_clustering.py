import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering




df = pd.read_csv('Vehicle_dataset.csv')


df.replace('$null$', np.nan, inplace=True)
df.dropna(inplace=True)



cols_to_convert = ['engine_s', 'horsepow', 'wheelbas', 'width', 'length',
                   'curb_wgt', 'fuel_cap', 'mpg']
df[cols_to_convert] = df[cols_to_convert].astype(float)





features = df[cols_to_convert]



scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



linkage_methods = ['single', 'complete', 'average']
labels = df['manufact'] + ' ' + df['model']



for method in linkage_methods:
    plt.figure(figsize=(12, 5))
    Z = linkage(scaled_features, method=method)
    dendrogram(Z, labels=labels.values, leaf_rotation=90)
    plt.title(f'Dendrogram - {method.capitalize()} Linkage')
    plt.xlabel('Car Model')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
















