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

# Correlacion Matrix
corr_matrix = df_out.drop(columns=[config["data"]["target_col"]]).corr()
# ── Guardar como corr_series (com .abs()) ────────
corr_series = df_out.drop(columns=[config["data"]["target_col"]]) \
                    .corrwith(df_out[config["data"]["target_col"]]) \
                    .abs() \
                    .sort_values(ascending=False)

print("Top 10 features com maior correlação absoluta:")
print(corr_series.head(10))

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top 10 features by Mutual Information:")
print(mi_series.head(10))

# Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=config["data"]["random_state"])
rf.fit(X, y)
rf_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 features by Random Forest Importance:")
print(rf_series.head(10))

# Combination and Normalization
def normalize(s):
    """Normaliza uma Series para escala 0-1 (Min-Max)"""
    return (s - s.min()) / (s.max() - s.min())

corr_norm = normalize(corr_series)
mi_norm   = normalize(mi_series)
rf_norm   = normalize(rf_series)

# Score final = média dos 3 métodos normalizados
combined        = (corr_norm + mi_norm + rf_norm) / 3
combined_sorted = combined.sort_values(ascending=False)

print("TOP 15 FEATURES — SCORE COMBINADO")
print(combined_sorted.head(15).round(3))


# VIF (Variance Inflation Factor) para detectar multicolinearidade
def calculate_vif(df_features):
    """Calcula o VIF para cada feature do dataframe."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_features.values, i)
        for i in range(df_features.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

# Pegar as top 15 pelo score combinado para analisar VIF
top15_features = combined_sorted.head(15).index.tolist()
X_top15 = X[top15_features]

vif_df = calculate_vif(X_top15)
high_vif_features = vif_df[vif_df['VIF'] > 10]['feature'].tolist()

print(f"\nVIF calculado para as top 15 features:")
print(vif_df.to_string(index=False))
print(f"\nFeatures com VIF > 10 (serão removidas): {high_vif_features}")

# Candidates Finais (removendo as de VIF alto)
candidates = combined_sorted.head(10).index.tolist()
final_features = [f for f in candidates if f not in high_vif_features]

print("\n" + "="*55)
print("FEATURES FINAIS SELECIONADAS PARA A MLP")
print("="*55)
for i, f in enumerate(final_features, 1):
    print(f"  {i:2d}. {f}  (score: {combined_sorted[f]:.3f})")
print(f"\nTotal: {len(final_features)} features")