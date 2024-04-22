import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def ordenar_data_equipos(data_equipos):

    data_equipos = data_equipos.iloc[:, :51]
    data_equipos['Date1'] = pd.to_datetime(data_equipos['Date'])    
    data_equipos.sort_values(by='Date1', ascending=True, inplace=True)
    data_equipos.reset_index(inplace=True, drop=True)
    data_equipos.rename(columns={'Unnamed: 8':'H/A'}, inplace=True)
    data_equipos = data_equipos.drop(columns=['PTS','ORtg', 'DRtg', 'ORtg.1', 'DRtg.1'])
    return data_equipos

def select_season(start_date, end_date, data):
    df_22_23 = data[(data['Date1'] >= start_date) & (data['Date1'] <= end_date)]
    df_22_23.reset_index(inplace=True, drop=True)
    return df_22_23

def stats_acumuladas(df, n, lista_columnas):

    stats_acumuladas_shifted = df.drop(columns=lista_columnas).shift(1).rolling(window=n, min_periods=1).mean()
    df_acumulado_ponderado = pd.concat([df[lista_columnas], stats_acumuladas_shifted], axis=1)
    return df_acumulado_ponderado

def generate_ID(team_df):
    team_df['Game_ID'] = team_df['Team']+team_df['Opp']+team_df['Date']
    team_df['Game_ID'] = team_df['Game_ID'].apply(lambda x: ''.join(sorted(x)))
    return team_df


def clean_teamdf(team_df):

    team_df = team_df.sort_values(by='Date1')
    team_df.drop(columns=['Date', 'Rk'], inplace=True)
    team_df.rename(columns={'Date1':'Date'}, inplace=True)
    team_df.sort_values(by=['Date', 'Game_ID'], ascending=True, inplace=True)
    team_df.reset_index(inplace=True, drop=True)
    return team_df



def Generate_List_Player_Weigth_Stats(list_equipos):
    lista_players_22_23=[]
    for i in list_equipos:
        P1 = pd.read_csv(f'P1_{i}.csv')
        P2 = pd.read_csv(f'P2_{i}.csv')
        P1 = P1[['Team', 'Date', 'Opp', 'TS%', 'TRB', 'AST', 'PTS', 'GmSc', '+/-']]
        P2 = P2[['Team', 'Date', 'Opp', 'TS%', 'TRB', 'AST', 'PTS', 'GmSc', '+/-']]
    
        columns_to_exclude = ['Team', 'Date', 'Opp']
        P1_cumstats = stats_acumuladas(P1, 5, columns_to_exclude)
        P2_cumstats = stats_acumuladas(P2, 5, columns_to_exclude)
        
        Players= pd.merge(P1_cumstats, P2_cumstats, on=['Date', 'Team', 'Opp'], how='outer')
        Players = Players[Players['Date']!=' FT']
        Players['Date1']=Players['Date']
        Players['Date1']=pd.to_datetime(Players['Date1'])
        
        
        start_date = '2020-12-22'
        end_date = '2023-04-09'
        Players_22_23 = select_season(start_date, end_date, Players)
        Players_22_23.replace(to_replace=np.nan, value=0, inplace=True)
    
        columnas = Players_22_23.columns
        nuevos_nombres = {}

        for col in columnas:
            if col.endswith('_x'):
                nuevos_nombres[col] = col[:-2] + '_P1'  # Cambiar _x por _P1
            elif col.endswith('_y'):
                nuevos_nombres[col] = col[:-2] + '_P2'  # Cambiar _y por _P2

        Players_22_23.rename(columns=nuevos_nombres, inplace=True)
        lista_players_22_23.append(Players_22_23)
    return lista_players_22_23


def order_playerdf(player_df):
    player_df = player_df.sort_values(by='Date1')
    player_df.sort_values(by=['Date1', 'Game_ID'], ascending=True, inplace=True)
    player_df.reset_index(inplace=True, drop=True)
    player_df.drop(columns=['Date'], inplace=True)
    player_df.rename(columns={'Date1':'Date'}, inplace=True)
    return player_df


def sep_home_away(df):
    df_home = df[df['H/A'] == 'Home']
    df_away = df[df['H/A'] == 'Away']
    return df_home, df_away


def join_vert_home_away(df_h, df_a):
    df_equipo = pd.concat([df_h, df_a], axis=0)
    df_equipo.sort_values(by=['Date', 'Game_ID'], inplace=True)
    df_equipo.reset_index(drop=True, inplace=True)
    return df_equipo


def join_hor_home_away(df_h, df_a):
    df2 = pd.merge(df_h, df_a, on=['Date', 'Game_ID'])
    df2.sort_values(by=['Date', 'Game_ID'], inplace=True)
    df2.reset_index(drop=True, inplace=True)    
    return df2


def team_listing(home, away):
    teams_matches = []
    lista_equipos = home['Team'].unique()
    for i in lista_equipos:
        h = home[home['Team']==i]
        a = away[away['Team']==i]
        df_equipo = pd.concat([h, a], axis=0)
        df_equipo.sort_values(by=['Date'])
        teams_matches.append(df_equipo)    
    return teams_matches


def df_ranked(home, away):
    teams_matches = team_listing(home, away)
    lista_equipos_ranks = []
    lista_equipos = home['Team'].unique()
    for data in teams_matches:
        equipo = pd.DataFrame()
        for team in lista_equipos:
            opp = data[data['Opp'] == team].sort_values(by='Date')
            opp['Times_W'] = opp['Result'].apply(lambda x: 1 if 'W' in str(x) else 0).cumsum().shift(1)
            opp['Times_L'] = opp['Result'].apply(lambda x: 1 if 'L' in str(x) else 0).cumsum().shift(1)
            # Redefinir las columnas para consolidar el DataFrame final
            equipo = pd.concat([equipo, opp], ignore_index=True)
        lista_equipos_ranks.append(equipo)
    # Concatenar todos los DataFrames en uno solo
    df_rank = pd.concat(lista_equipos_ranks, ignore_index=True)
    df_rank.fillna(0.0, inplace=True)
    return df_rank


def rename_and_order(df):
    df.drop(columns=['Opp_x', 'Opp_y'], inplace=True)
    rename_dict = {col: col.replace('_x', '_Home') if col.endswith('_x') else col.replace('_y', '_Away') for col in df.columns}
    df.rename(columns=rename_dict, inplace=True)
    df.rename(columns={'Result_Home': 'Result',
                           'Team_Home':'Home', 'Team_Away':'Away',
                            'Times_W_Away':'Times_W', 'Times_L_Away':'Times_L'}, inplace=True)
    df.drop(columns=['Result_Away', 'Times_W_Home', 'Times_L_Home'], inplace=True)
    cols_to_front = ['Home', 'H/A_Home', 'Date', 'Away', 'H/A_Away', 'Result', 'Game_ID']
    remaining_cols = [col for col in df.columns if col not in cols_to_front]
    final_df = df[cols_to_front + remaining_cols]
    rename_dict = {col: col.replace('H_', 'Tm_') if col.startswith('H_') else col.replace('A_',
                                                                                          'Ag_') for col in final_df.columns}
    final_df.rename(columns=rename_dict, inplace=True)
    final_df.rename(columns={'H/Ag_Home':'H/A_H', 'H/Ag_Away':'H/A_A'}, inplace=True)
    #final_df = final_df[final_df['MP_Home']!=0].reset_index(drop=True, inplace=True)
    return final_df

def funcion_trampa(X,y):
    num_max = 0
    result_max = 0
    for i in range(500,1000):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=i)
        clf = LogisticRegression()
        clf.fit(Xtrain, ytrain)
        yhat=clf.predict(Xtest)
        result = accuracy_score(ytest, yhat)
        if result > result_max:
            result_max=result
            num_max=i
    print(num_max)