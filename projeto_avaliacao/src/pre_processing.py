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

# Remoção de duplicatas e tratamento de valores faltantes

# CORREÇÃO ESSENCIAL: troca '?' por NaN antes de tudo
df_raw['age'] = df_raw['age'].replace('?', np.nan) # substitui os valores '?' por NaN para que possam ser tratados como valores faltantes
df_raw['age'] = pd.to_numeric(df_raw['age'])  # converte para número

def remove_duplicates(df): # remove linhas duplicadas do dataframe
    before = len(df)        # salva número original de linhas
    df = df.drop_duplicates() # remove todas as linhas duplicadas do dataframe
    print(f"Removed {before - len(df)} duplicates") # imprime quantas linhas foram removidas
    return df # retorna o dataframe sem duplicatas

def handle_missing_values(df, strategy=config["data"]["imputation_strategy"], threshold=config["data"]["missing_threshold"]): # trata valores faltantes
    missing_frac = df.isnull().median() # calcula a fração de valores faltantes por coluna
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist() # identifica colunas com mais de 50% de valores faltantes
    df = df.drop(columns=cols_to_drop) # remove essas colunas do dataframe
    print(f"Dropped columns: {cols_to_drop}") # imprime quais colunas foram removidas 
    numeric_cols = df.select_dtypes(include=[np.number]).columns # seleciona apenas as colunas numéricas
    imputer = SimpleImputer(strategy=strategy) # cria um imputer para preencher os valores faltantes usando a estratégia escolhida (mediana)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols]) # preenche os valores faltantes nas colunas numéricas usando a estratégia escolhida (média, mediana, etc.)
    cat_cols = df.select_dtypes(include=['object']).columns # seleciona apenas as colunas do tipo objeto
    df[cat_cols] = df[cat_cols].fillna('missing') # preenche os valores faltantes com 'missing'
    return df

#executando
df_clean = remove_duplicates(df_raw) # remove linhas duplicadas
df_clean = handle_missing_values(df_clean, strategy=config["data"]["imputation_strategy"], threshold=config["data"]["missing_threshold"]) # trata valores faltantes
print(f"Dataset no formato limpo: {df_clean.shape}") # mostra o número de linhas e colunas do dataset limpo

# confirma que não sobrou nulo
print(f"Nulos restantes: {df_clean.isnull().sum().sum()}")

# Tratamento de outliers
def detect_and_treat_outliers_iqr(df_clean, columns, strategy='capping', fator=1.5): # função para detectar e tratar outliers usando o método do intervalo interquartil (IQR)
    # Detecta e trata outliers via IQR.
    # Estratégias: 'capping' | 'mediana' | 'remover', serve para substituir os outliers pelos limites ou pela mediana, ou remover as linhas com outliers
    df_out = df_clean.copy() # para não modificar o original

    for col in columns:
        Q1 = df_out[col].quantile(0.25) # primeiro quartil
        Q3 = df_out[col].quantile(0.75) # terceiro quartil
        IQR = Q3 - Q1 # intervalo interquartil
        li = Q1 - fator * IQR   # limite inferior
        ls = Q3 + fator * IQR   # limite superior

        outliers = ((df_out[col] < li) | (df_out[col] > ls)).sum() # conta quantos outliers existem nessa coluna comparando com os limites se são menores que o limite inferior ou maiores que o limite superior
        print(f"[{col}] Q1={Q1:.1f} | Q3={Q3:.1f} | Limites: [{li:.1f}, {ls:.1f}] | Outliers: {outliers}") # imprime os quartis, limites e número de outliers encontrados
        
        if strategy == 'capping': # substitui os outliers pelos limites
            df_out[col] = df_out[col].clip(lower=li, upper=ls)   # substitui pelos limites

        elif strategy == 'median': # substitui os outliers pela mediana
            mediana = df_out[col].median()
            df_out[col] = df_out[col].where((df_out[col] >= li) & (df_out[col] <= ls), mediana) # substitui os outliers pela mediana usando where para manter os valores dentro dos limites e substituir os outliers

        elif strategy == 'remove': # remove as linhas com outliers
            df_out = df_out[(df_out[col] >= li) & (df_out[col] <= ls)] # mantém apenas as linhas onde os valores estão dentro dos limites, removendo as linhas com outliers

    return df_out
   

# ── executa ───────────────────────────────────────────────────
# só 'age' tem escala contínua; as demais são ordinais (0-3)
colunas_alvo = ['age'] # define quais colunas serão tratadas para outliers (apenas 'age' nesse caso, pois as outras são ordinais com poucos valores possíveis)

df_out = detect_and_treat_outliers_iqr(df_clean, columns=colunas_alvo, strategy='capping') # trata os outliers usando o método do intervalo interquartil (IQR) e a estratégia de capping (substituir pelos limites)

print(f"\nShape após tratamento: {df_clean.shape}") # mostra o número de linhas e colunas do dataset após o tratamento de outliers
print(f"Age — min: {df_clean['age'].min():.0f} | max: {df_clean['age'].max():.0f}") # mostra o valor mínimo e máximo da coluna 'age' após o tratamento de outliers para confirmar que os valores extremos foram tratados

# Normalização e padronização
# ── separa target antes de tudo ──────────────────────────
X = df_out.drop(columns=[config["data"]["target_col"]]) # separa as features (todas as colunas exceto 'class') e salva em X
y = df_out[config["data"]["target_col"]] # separa a coluna 'class' como target e salva em y

# ── aplica MinMaxScaler ──────────────────────────────────
scaler = MinMaxScaler() # cria um objeto MinMaxScaler para normalizar os dados, transformando os valores para uma escala entre 0 e 1
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # aplica o scaler aos dados de X e cria um novo dataframe com os valores normalizados, mantendo os mesmos nomes de colunas

# ── comparação antes × depois ────────────────────────────
print("ANTES da normalização:")
print(X[['age', 'itching', 'scaling']].describe().round(2)) # mostra estatísticas básicas para as colunas 'age', 'itching' e 'scaling' antes da normalização, arredondando os valores para 2 casas decimais

print("\nDEPOIS da normalização:")
print(X_scaled[['age', 'itching', 'scaling']].describe().round(2)) # mostra estatísticas básicas para as colunas 'age', 'itching' e 'scaling' depois da normalização, arredondando os valores para 2 casas decimais, confirmando que os valores foram transformados para uma escala entre 0 e 1