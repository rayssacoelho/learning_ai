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


path = kagglehub.dataset_download("olcaybolat1/dermatology-dataset-classification") 
print("Path para os arquivos do dataset:", path)
# Carregar os dados do arquivo .data
# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# caminho que o kagglehub retornou
source_path = path  

# destino dentro do seu projeto
dest_path = os.path.join(os.getcwd(), "data")

# copia tudo pra pasta data
shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

print("Dataset copiado para:", dest_path)

df_raw = pd.read_csv( # lê o arquivo CSV
     "data/dermatology_database_1.csv",
    low_memory=False # evita avisos de tipos de dados mistos
)
print(f"Shape: {df_raw.shape}") # mostra o número de linhas e colunas
print(f"Columns: {list(df_raw.columns[:5])}...") # mostra os nomes das primeiras 5 colunas
df_raw.head() # mostra as primeiras 5 linhas do dataframe