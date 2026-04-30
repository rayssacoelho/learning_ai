model = MLP(
    input_dim    = len(final_features),
    hidden_sizes = config["model"]["hidden_sizes"],
    output_dim   = config["model"]["output_dim"],
    dropout      = config["model"]["dropout"]
).to(device)

model.load_state_dict(torch.load("best_model_final.pt", weights_only=True))
model.eval() # coloca o modelo em modo de avaliação, desativando dropout e batchnorm para garantir previsões consistentes durante a avaliação
print("✅ Modelo carregado!")

all_preds  = [] # lista para armazenar as previsões do modelo
all_labels = [] # lista para armazenar os rótulos verdadeiros
all_probs  = [] # lista para armazenar as probabilidades previstas para cada classe, que serão usadas para calcular métricas como ROC AUC

with torch.no_grad():
    for X_batch, y_batch in test_loader: # itera sobre os batches do dataloader de teste, onde X_batch são as features e y_batch são os rótulos verdadeiros para aquele batch
        X_batch = X_batch.to(device) # move o batch de features para o mesmo dispositivo do modelo (GPU ou CPU)
        output  = model(X_batch) # obtém as saídas do modelo para o batch atual, que são os logits (valores antes da função softmax) para cada classe
        probs   = torch.softmax(output, dim=1)  # probabilidades por classe
        preds   = output.argmax(dim=1) # obtém as previsões do modelo para o batch atual, que são os índices das classes com a maior probabilidade prevista

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)   # converte a lista de previsões para um array numpy para facilitar o cálculo das métricas
all_labels = np.array(all_labels)  # converte a lista de rótulos verdadeiros para um array numpy para facilitar o cálculo das métricas
all_probs  = np.array(all_probs)

# Nomes das classes (0-5 → Classe 1-6)
class_names = [f"Classe {i+1}" for i in range(6)]

# Métricas de avaliação
acc       = accuracy_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

try:
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
except:
    auc = 0.0

print("\n" + "=" * 50)
print("MÉTRICAS FINAIS — MODELO MLP")
print("=" * 50)
print(f"Acurácia  : {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score  : {f1:.4f}")
print(f"Precisão  : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"AUC-ROC   : {auc:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)}")