import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ── Arquitetura da MLP ────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=6, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))   # estabiliza treino
            layers.append(nn.ReLU())         # ativações não lineares
            layers.append(nn.Dropout(dropout)) # reativado — importante para regularização
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def prepare_dataloaders(train_df, test_df, target_col, batch_size):

    X_train = train_df.drop(columns=[target_col]).values
    X_test  = test_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    y_test  = test_df[target_col].values

    # Normalizar apenas as features
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ✅ y como long 1D e classes 0-5
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train - 1, dtype=torch.long)  # 1-6 → 0-5
    y_test_t  = torch.tensor(y_test  - 1, dtype=torch.long)

    # Verificação de shapes
    print(f"X_train shape: {X_train_t.shape}")          # [292, 10]
    print(f"y_train shape: {y_train_t.shape}")          # [292]
    print(f"Classes únicas: {y_train_t.unique()}")      # [0,1,2,3,4,5]

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, scaler