import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PCAfunctions import *

def create_mean_teams(df):
    lista_equipo = df['Home'].unique()
    df_mean_teams = pd.DataFrame()
    for i in lista_equipo:
        df_equipo = df[df['Home']==i]
        home_columns = [col for col in df.columns if col.endswith('_Home')] + ['ORtg'] + ['DRtg']
        df_equipo = df_equipo[home_columns]
        mean_values = df_equipo.mean()
        df_mean = mean_values.to_frame().T
        df_mean['Team'] = i 
        df_mean_teams = pd.concat([df_mean_teams, df_mean], axis=0)
    column_order = ['Team'] + [col for col in df_mean_teams.columns if col != 'Team']
    df_mean_teams = df_mean_teams.reindex(columns=column_order)
    # Ordenar las filas por la columna 'Team' en orden alfabético
    df_mean_teams = df_mean_teams.sort_values(by='Team')
    df_mean_teams.reset_index(drop=True, inplace=True)
    return df_mean_teams

def compact_data(df_mean):
    df_diff = pd.DataFrame()
    for col in df_mean.columns:
        if col.startswith('Tm_'):
            ag_col = 'Ag_' + col.split('_')[1] + '_Home'
            if ag_col in df_mean.columns: 
                short_name = col[3:-5] + '_Diff'
                df_diff[short_name] = df_mean[col] - df_mean[ag_col]
    df_diff = pd.concat([df_diff, df_mean.filter(like='Diff')], axis=1)
    # Incluir las columnas 'Times_W' y 'Times_L'
    df_diff['ORtg'] = df_mean['ORtg']
    df_diff['DRtg'] = df_mean['DRtg']
    df_diff['Team'] = df_mean['Team']
    return df_diff

def pca_dimension(df):
    df_num = df._get_numeric_data()
    data = StandardScaler().fit_transform(df_num)
    data = pd.DataFrame(data, columns=df_num.columns)
    
    pca = PCA(n_components=3).fit(data)
    data_pca = pca.transform(data)
    data_pca = pd.DataFrame(data_pca, columns=['dim1', 'dim2', 'dim3'])
    data_pca['Team']=df['Team']
    
    dt_components=pd.DataFrame(pca.components_, columns=data.columns)
    
    return data, data_pca, dt_components, pca


def cluster_3d_graphic(data):
    algoritmo = KMeans(n_clusters = 3, init = 'k-means++',
                       max_iter = 300, n_init = 10)
    algoritmo.fit(data)
    centroides, etiquetas = algoritmo.cluster_centers_, algoritmo.labels_
    
    modelo_pca = PCA(n_components = 3)
    modelo_pca.fit(data)
    pca = modelo_pca.transform(data)
    
    dt_components=pd.DataFrame(modelo_pca.components_, columns=data.columns)
    dt_components
    
    centroides_pca = modelo_pca.transform(centroides)
    colores = ['blue', 'red', 'green']
    colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]
    plt.scatter(pca[:, 0], pca[:, 1], c = colores_cluster,
                marker = 'o',alpha = 0.7)
    plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                marker = 'x', s = 100, linewidths = 3, c = colores)
    xvector = modelo_pca.components_[0] * max(pca[:,0])
    yvector = modelo_pca.components_[1] * max(pca[:,1])
    columnas = data.columns
    for i in range(len(columnas)):
        plt.arrow(0, 0, xvector[i], yvector[i], color = 'black',
                  width = 0.0005, head_width = 0.02, alpha = 0.75)
        plt.text(xvector[i], yvector[i], list(columnas)[i], color='black',
                 alpha=1)

    plt.show()
    return etiquetas



