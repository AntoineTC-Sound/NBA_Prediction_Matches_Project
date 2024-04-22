import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def mapa_correlacion(df):
    corr = df.corr()
    # Configurando el tamaño del gráfico
    plt.figure(figsize=(20, 15))
    # Creando la heatmap (mapa de calor)
    sns.heatmap(corr,
                annot=True,      # Para mostrar los valores de las correlaciones
                cmap='coolwarm', # Colores de la matriz, puedes cambiarlo por otro como 'viridis'
                fmt=".1f",       # Formato de los números dentro de los cuadros
                linewidths=.05)  # Espacio entre cuadros
    # Mostrando el gráfico
    plt.show()
    

    
def make_diff(df):   
    columns_home = [col for col in df.columns if '_Home' in col] 

    df_home = df[columns_home]

    df_home.columns = [col.replace('_Home', '') for col in df_home.columns]

    df_home = df_home._get_numeric_data()


    columns_away = [col for col in df.columns if '_Away' in col] 

    df_away = df[columns_away]

    df_away.columns = [col.replace('_Away', '') for col in df_away.columns]

    df_away = df_away._get_numeric_data()

    df_diff = df_home - df_away
    return df_diff



def print_most_correlated(df, n):
    corr = df.corr()
    correlated_vars = []

    for col in corr.columns:
        correlated = corr.index[corr[col].abs() >= n].tolist()
        correlated.remove(col)  # Eliminar la variable en sí misma de la lista
        for var in correlated:
            if (var, col) not in correlated_vars and (col, var) not in correlated_vars:
                correlated_vars.append((col, var))

    # Mostrar las variables correlacionadas
    print(f"Variables correlacionadas con un |correlación| >= {n}:")
    for var1, var2 in correlated_vars:
        print(f"{var1} está correlacionado con {var2} (correlación = {corr.loc[var1, var2]:.2f})")
        
        
def cross_correlation(df, n):
    columns_home = [col for col in df.columns if '_Home' in col] 
    df_home = df[columns_home]
    df_home.columns = [col.replace('_Home', '') for col in df_home.columns]
    df_home = df_home._get_numeric_data()


    columns_away = [col for col in df.columns if '_Away' in col] 
    df_away = df[columns_away]
    df_away.columns = [col.replace('_Away', '') for col in df_away.columns]
    df_away = df_away._get_numeric_data()  
    
    corr_cross = df_home.corrwith(df_away)
    corr_cross_filtered = corr_cross[(corr_cross >= n) | (corr_cross <= -n)]

    corr_cross_filtered.plot(kind='bar', figsize=(10, 6), title='Correlaciones Cruzadas Más Significativas')
    plt.xlabel('Estadísticas')
    plt.ylabel('Correlación')
    plt.show()
    

def total_players_stats(df):
    df['PTS_Total_Home'] = df['PTS_P1_Home'] + df['PTS_P2_Home']
    df['TRB_Total_Home'] = df['TRB_P1_Home'] + df['TRB_P2_Home']
    df['AST_Total_Home'] = df['AST_P1_Home'] + df['AST_P2_Home']
    df['PTS_Total_Away'] = df['PTS_P1_Away'] + df['PTS_P2_Away']
    df['TRB_Total_Away'] = df['TRB_P1_Away'] + df['TRB_P2_Away']
    df['AST_Total_Away'] = df['AST_P1_Away'] + df['AST_P2_Away']
    return df


def avg_p1_p2(df):
    for stat in ['TS%', 'TRB', 'AST', 'PTS', '+/-']:
        df[f'{stat}_Avg_Home'] = (df[f'{stat}_P1_Home'] + df[f'{stat}_P2_Home']) / 2
        df[f'{stat}_Avg_Away'] = (df[f'{stat}_P1_Away'] + df[f'{stat}_P2_Away']) / 2
    return df


def porcentual_diff_players(df):
    for stat in ['TS%', 'TRB', 'AST', 'PTS', '+/-']:
        df[f'{stat}_Diff_Home'] = (df[f'{stat}_P1_Home'] - df[f'{stat}_P2_Home']) / df[f'{stat}_P1_Home']
        df[f'{stat}_Diff_Away'] = (df[f'{stat}_P1_Away'] - df[f'{stat}_P2_Away']) / df[f'{stat}_P1_Away']
    return df




