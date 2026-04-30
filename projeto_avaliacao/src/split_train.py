import pandas as pd                                             # manipular dados (dataframe)
import numpy as np                                              # cálculos
import wandb                                                    # rastrear, visualizar e gerenciar experimentos de ML
import kagglehub                                                # acessar datasets do Kaggle
import shutil                                                   # serve para mexer com arquivos e pastas
import os                                                       # para lidar com caminhos de arquivos e diretórios
import seaborn as sns                                           # gráficos
import random                                                   # para gerar números aleatórios
import matplotlib.pyplot as plt                                 # gráficos
import joblib                                                   # para salvar e carregar modelos
import torch                                                    # para trabalhar com tensores e deep learning
import torch.nn as nn                                           # para construir redes neurais
import torch.optim as optim                                     # para otimizar os pesos da rede neural
import statsmodels.api as sm                                    # para análise estatística


from torch.utils.data import DataLoader, TensorDataset          # para criar dataloaders a partir de tensores
from pyparsing import col                                       # para lidar com colunas em parsing de texto
from dotenv import load_dotenv                                  # para carregar variáveis de ambiente de um arquivo .env
from sklearn.model_selection import train_test_split            # para dividir os dados em treino e teste
from sklearn.impute import SimpleImputer                        # para tratar valores faltantes
from sklearn.feature_selection import mutual_info_regression    # para calcular a importância das features usando mutual information
from sklearn.ensemble import RandomForestRegressor              # modelo de regressão baseado em árvores
from sklearn.linear_model import LinearRegression               # modelo de regressão linear
from sklearn.metrics import mean_squared_error                  # para calcular o erro do modelo
from sklearn.ensemble import RandomForestRegressor              # modelo de regressão baseado em árvores
from sklearn.linear_model import LinearRegression               # modelo de regressão linear
from sklearn.metrics import mean_squared_error                  # para calcular o erro do modelo
from sklearn.preprocessing import MinMaxScaler                  # para normalizar os dados
from sklearn.feature_selection import mutual_info_classif       # para calcular a importância das features usando mutual information
from sklearn.ensemble import RandomForestClassifier             # modelo de classificação baseado em árvores
from sklearn.preprocessing import StandardScaler                # para padronizar os dados
from sklearn.feature_selection import SelectKBest, f_classif    # para selecionar as melhores features usando o teste f_classif
from sklearn.pipeline import Pipeline                           # para criar um pipeline de pré-processamento e modelagem
from sklearn.linear_model import LogisticRegression             # modelo de regressão logística para classificação
from sklearn.inspection import permutation_importance           # para calcular a importância das features usando permutation importance
from statsmodels.stats.outliers_influence import variance_inflation_factor  # para calcular o VIF (Variance Inflation Factor) e detectar multicolinearidade

# Configurar semente para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

config = {
    "data": {
        "target_col": "class",
        "test_size": 0.2,
        "random_state": 42,
        "imputation_strategy": "median",
        "missing_threshold": 0.5
    },
    "model": {
        "hidden_sizes": [64, 32],
        "output_dim": 6,
        "dropout": 0.2
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 150,
        "early_stopping_patience": 10
        
    }
}

def split_train_test(
    df,
    target_col,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    return train_df, test_df
from scipy.stats import ks_2samp, chi2_contingency
def compare_distributions(train_df, test_df, columns):
    results = {}
    for col in columns:
        train_vals = train_df[col].dropna()
        test_vals  = test_df[col].dropna()

        if train_df[col].dtype in ['int64', 'float64']:
            # KS test para variáveis numéricas contínuas
            ks_stat, p_value = ks_2samp(train_vals, test_vals)
            results[col] = {
                'test'     : 'KS',
                'statistic': round(ks_stat, 4),
                'p_value'  : round(p_value, 4)
            }
        else:
            # Chi2 para variáveis categóricas
            all_cats    = sorted(set(train_vals).union(set(test_vals)))
            train_counts = [train_vals.value_counts().get(cat, 0) for cat in all_cats]
            test_counts  = [test_vals.value_counts().get(cat,  0) for cat in all_cats]
            chi2, p_value, _, _ = chi2_contingency([train_counts, test_counts])
            results[col] = {
                'test'     : 'Chi2',
                'statistic': round(chi2, 4),
                'p_value'  : round(p_value, 4)
            }

    return results
train_df, test_df = split_train_test(
    df_out,
    target_col=config["data"]["target_col"],
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"]
)

feature_cols = [
    c for c in train_df.columns
    if c != config["data"]["target_col"]
]

# Usar train_df e test_df que foram retornados
print(f"Treino: {len(train_df)} amostras")
print(f"Teste:  {len(test_df)} amostras")

comp_results = compare_distributions(train_df, test_df, feature_cols)

# Mostrar resultado
results_df = pd.DataFrame(comp_results).T
print(results_df)