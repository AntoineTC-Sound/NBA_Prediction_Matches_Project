import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier




def bayes(prob_inicial, prob_model):
    
    prob_bayes = (prob_model * prob_inicial) / ((prob_model * prob_inicial) + ((1 - prob_model) * (1 - prob_inicial)))
    return prob_bayes



def NaiveBayes(X, y, df):
    prob_inicial_W = df['Prob_W']
    prob_inicial_L = df['Prob_L']
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=200, shuffle=False)

    prob_inicial_Wtest = prob_inicial_W[-200:]
    prob_inicial_Ltest = prob_inicial_L[-200:]
    clf = LogisticRegression()
    clf.fit(Xtrain, ytrain)
    prob_model = clf.predict_proba(Xtest)
    
    prob_inicial = []
    for i in zip(prob_inicial_Wtest, prob_inicial_Ltest):
        prob_inicial.append(i)
        
    prob_inicial = np.array(prob_inicial)
    prob_bayes = pd.DataFrame(bayes(prob_inicial, prob_model))
    predict_bayes=prob_bayes.iloc[:,0].apply(lambda x: 1 if x >0.5 else 0)  
    return accuracy_score(ytest, predict_bayes)
    
def GNB(Xtrain, Xtest, ytrain, ytest):
    clf = GaussianNB()
    clf.fit(Xtrain, ytrain)
    yhat = clf.predict(Xtest)
    test_acc = accuracy_score(ytest, yhat)    
    yhat_train = clf.predict(Xtrain)
    train_acc = accuracy_score(ytrain, yhat_train)
    return test_acc, train_acc

    

def LogicRegression(Xtrain, Xtest, ytrain, ytest):
    clf = LogisticRegression()
    clf.fit(Xtrain, ytrain)
    yhat = clf.predict(Xtest)
    test_acc = accuracy_score(ytest, yhat)  
    yhat_train = clf.predict(Xtrain)
    train_acc = accuracy_score(ytrain, yhat_train)
    return test_acc, train_acc


def XGB(Xtrain, Xtest, ytrain, ytest):
    clf = XGBClassifier()

    params={'base_score': [0.3],
     'colsample_bylevel': [1],
     'colsample_bytree': [0.7],
     'gamma': [0.1],
     'learning_rate': [0.2],
     'max_depth': [1],
     'min_child_weight': [3],
     'n_estimators': [100],
     'n_jobs': [-1],
     'random_state': [0],
     'reg_alpha': [0.3],
     'reg_lambda': [0.25],
     'scale_pos_weight': [1],
     'subsample': [0.6]} # ratio de muestras por cada arbol

    scoring = ['accuracy', 'roc_auc']
    n_cv=3 
    grid_solver = GridSearchCV(estimator = clf,
                       param_grid = params, 
                       scoring = scoring,
                       cv = n_cv,
                       refit = 'accuracy',
                       verbose = 1)
    model = grid_solver.fit(Xtrain,ytrain)
    best_model=model.best_estimator_
    
    yhat=model.predict(Xtest)
    test=accuracy_score(ytest, yhat)
    
    yhat=model.predict(Xtrain)
    train=accuracy_score(ytrain, yhat)
    return test, train






def RandomForest(Xtrain, Xtest, ytrain, ytest):
    clf = RandomForestClassifier()
    clf.get_params()
    params={'bootstrap': [True],
     'ccp_alpha': [0.1],
     'class_weight': [None],
     'criterion': ['gini'],
     'max_depth': [1],
     'max_features': ['sqrt'],
     'max_leaf_nodes': [2],
     'max_samples': [2],
     'min_impurity_decrease': [0.1],
     'min_samples_leaf': [4],
     'min_samples_split': [4],
     'min_weight_fraction_leaf': [0.05],
     'n_estimators': [100],
     'n_jobs': [None],
     'oob_score': [False],
     'random_state': [None],
     'verbose': [0],
     'warm_start': [False]}
    scoring = ['accuracy', 'roc_auc']
    n_cv=3 
    grid_solver = GridSearchCV(estimator = clf,
                       param_grid = params, 
                       scoring = scoring,
                       cv = n_cv,
                       refit = 'accuracy',
                       verbose = 1)
    model = grid_solver.fit(Xtrain,ytrain)
    yhat=model.predict(Xtest)
    test = accuracy_score(ytest, yhat)
    
    yhat=model.predict(Xtrain)
    train = accuracy_score(ytrain, yhat)
    return test,train


def CatBoost(Xtrain, Xtest, ytrain, ytest):
    clf = CatBoostClassifier()

    # Definir los parámetros para GridSearchCV
    params = {
        'learning_rate': [0.01],  # Reducimos la tasa de aprendizaje para un aprendizaje más lento y conservador
        'depth': [4],  # Limitamos la profundidad de los árboles para un modelo más simple
        'l2_leaf_reg': [1],  # Aplicamos una regularización L2 más fuerte
        'iterations': [100],  # Limitamos el número de iteraciones para evitar la complejidad excesiva
        'random_state': [42]
    }

    grid_solver = GridSearchCV(estimator=clf, param_grid=params, scoring='accuracy', cv=3, refit=True, verbose=0)
    grid_solver.fit(Xtrain, ytrain)
    best_model = grid_solver.best_estimator_

    ypred = best_model.predict(Xtest)
    test = accuracy_score(ytest, ypred)
    
    yhat = best_model.predict(Xtrain)
    train = accuracy_score(ytrain, yhat)
    
    return test, train


def NeuralNet(Xtrain, Xtest, ytrain, ytest):
    params = {
        'hidden_layer_sizes': [(100,), (50, 50)],  # Tamaños de las capas ocultas
        'activation': ['relu', 'tanh'],  # Funciones de activación
        'solver': ['adam'],  # Solucionador de optimización
        'alpha': [0.0001, 0.001],  # Término de regularización L2
        'learning_rate': ['constant', 'adaptive'],  # Tasa de aprendizaje
        'max_iter': [200],  # Número máximo de iteraciones
        'random_state': [42]
    }

    mlp_clf = MLPClassifier()
    grid_solver_mlp = GridSearchCV(estimator=mlp_clf, param_grid=params, 
                                   scoring='accuracy', cv=3, refit=True, verbose=1)

    grid_solver_mlp.fit(Xtrain, ytrain)
    best_model = grid_solver_mlp.best_estimator_

    yhat = best_model.predict(Xtest)
    test = accuracy_score(ytest, yhat)
    
    yhat = best_model.predict(Xtrain)
    train = accuracy_score(ytrain, yhat)
    return test, train
    

    
def get_importances(Xtrain, Xtest, ytrain, ytest, n):
    clf = XGBClassifier()

    params={'base_score': [0.3],
     'colsample_bylevel': [1],
     'colsample_bytree': [0.7],
     'gamma': [0.1],
     'learning_rate': [0.2],
     'max_depth': [1],
     'min_child_weight': [3],
     'n_estimators': [100],
     'n_jobs': [-1],
     'random_state': [0],
     'reg_alpha': [0.3],
     'reg_lambda': [0.25],
     'scale_pos_weight': [1],
     'subsample': [0.6]} # ratio de muestras por cada arbol

    scoring = ['accuracy', 'roc_auc']
    n_cv=3 
    grid_solver = GridSearchCV(estimator = clf,
                       param_grid = params, 
                       scoring = scoring,
                       cv = n_cv,
                       refit = 'accuracy',
                       verbose = 1)
    model = grid_solver.fit(Xtrain,ytrain)
    
    best_model=model.best_estimator_
    importances=pd.DataFrame([Xtest.columns,best_model.feature_importances_], index=["feature","importance"]).T
    print(importances.sort_values("importance", ascending = False).head(n))
    
    
    
def model_xgb(Xtrain, ytrain):
    clf = XGBClassifier()
    params={'base_score': [0.3],
     'colsample_bylevel': [1],
     'colsample_bytree': [0.7],
     'gamma': [0.1],
     'learning_rate': [0.2],
     'max_depth': [1],
     'min_child_weight': [3],
     'n_estimators': [100],
     'n_jobs': [-1],
     'random_state': [0],
     'reg_alpha': [0.3],
     'reg_lambda': [0.25],
     'scale_pos_weight': [1],
     'subsample': [0.6]} 
    scoring = ['accuracy', 'roc_auc']
    n_cv=3 
    grid_solver = GridSearchCV(estimator = clf,
                       param_grid = params, 
                       scoring = scoring,
                       cv = n_cv,
                       refit = 'accuracy',
                       verbose = 1)
    xgb = grid_solver.fit(Xtrain,ytrain)
    best_xgb=xgb.best_estimator_
    return best_xgb


def model_catboost(Xtrain, ytrain):
    clf = CatBoostClassifier()
    params = {
        'learning_rate': [0.01],  
        'depth': [4],  
        'l2_leaf_reg': [1],  
        'iterations': [100], 
        'random_state': [42]
    }
    grid_solver = GridSearchCV(estimator=clf, 
                               param_grid=params, scoring='accuracy', cv=3, refit=True, verbose=0)
    grid_solver.fit(Xtrain, ytrain)
    catboost = grid_solver.best_estimator_
    return catboost

def model_neuralnet(Xtrain, ytrain):
    params = {
        'hidden_layer_sizes': [(100,), (50, 50)],  # Tamaños de las capas ocultas
        'activation': ['relu', 'tanh'],  # Funciones de activación
        'solver': ['adam'],  # Solucionador de optimización
        'alpha': [0.0001, 0.001],  # Término de regularización L2
        'learning_rate': ['constant', 'adaptive'],  # Tasa de aprendizaje
        'max_iter': [200],  # Número máximo de iteraciones
        'random_state': [42]
    }

    mlp_clf = MLPClassifier()
    grid_solver_mlp = GridSearchCV(estimator=mlp_clf, param_grid=params, 
                                   scoring='accuracy', cv=3, refit=True, verbose=1)

    grid_solver_mlp.fit(Xtrain, ytrain)
    best_model = grid_solver_mlp.best_estimator_
    return best_model


def voting_3bests(Xtrain, Xtest, ytrain, ytest):    
    neural = model_neuralnet(Xtrain, ytrain)
    xgb = model_xgb(Xtrain, ytrain)
    catboost = model_catboost(Xtrain, ytrain)

    pred_neural = neural.predict(Xtest)
    pred_xgb = xgb.predict(Xtest)
    pred_catboost = catboost.predict(Xtest)

    accuracy_neural = 0.6
    accuracy_xgb = 0.7
    accuracy_catboost = 0.65

    total_accuracy = accuracy_neural + accuracy_xgb + accuracy_catboost

    weight_neural = accuracy_neural / total_accuracy
    weight_xgb = accuracy_xgb / total_accuracy
    weight_catboost = accuracy_catboost / total_accuracy

    voting_clf = VotingClassifier(
        estimators=[
            ('neural', neural),
            ('xgb', xgb),
            ('catboost', catboost)
        ],
        voting='soft', 
        weights=[weight_neural, weight_xgb, weight_catboost]
    )

    voting_clf.fit(Xtrain, ytrain)

    pred_voting_test = voting_clf.predict(Xtest)
    test = accuracy_score(ytest, pred_voting_test)
    
    pred_voting = voting_clf.predict(Xtrain)
    train = accuracy_score(ytrain, pred_voting)
    
    return test, train, pred_voting_test


    
def accuracy_teams(X, y, df):
    X['Team'] = df['Home']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=200, random_state=0) 
    Team_test = Xtest['Team']
    Team_train = Xtrain['Team']
    Xtest.drop(columns=['Team'], inplace = True)
    Xtrain.drop(columns=['Team'], inplace = True)
    
    xgb = model_xgb(Xtrain, ytrain)
    yhat = xgb.predict(Xtest)
    
    df_preds = pd.DataFrame()
    df_preds['Home'] = Team_test
    df_preds['Result'] = ytest
    df_preds['Predict']= yhat
    
    lista_equipos = df['Home'].unique()
    d = {}
    for i in lista_equipos:
        df_equipo = df_preds[df_preds['Home']==i]
        d[i]=[round(accuracy_score(df_equipo['Result'], df_equipo['Predict']),2)]
    accuracy_teams = pd.DataFrame(d)
    sns.barplot(accuracy_teams)

    plt.xticks(rotation=90)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()
    print(d)
    return df_preds, accuracy_teams
    
    
    
    
    
    
    
    
    