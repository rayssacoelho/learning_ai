def train_model(config, train_loader, test_loader, input_dim):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = MLP(
        input_dim,
        config['model']['hidden_sizes'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_loss = float('inf')
    patience  = config['training']['early_stopping_patience']
    counter   = 0

    wandb.watch(model, log="all")

    for epoch in range(config['training']['epochs']):

        # ── Treino ────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)  # CrossEntropy espera [N,6] e [N]
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validação ─────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output   = model(X_batch)
                loss     = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)


                pred     = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()

        val_loss /= len(test_loader.dataset)
        acc       = correct / len(test_loader.dataset)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1:3d}] | Train Loss: {train_loss:.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {acc:.3f}")

        wandb.log({
            "epoch"     : epoch + 1,
            "train_loss": train_loss,
            "val_loss"  : val_loss,
            "val_acc"   : acc,
            "lr"        : optimizer.param_groups[0]['lr']
        })

        # ── Early Stopping ────────────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            counter   = 0
            torch.save(model.state_dict(), "best_model.pt")
            model_artifact = wandb.Artifact("trained_model", type="model")
            model_artifact.add_file("best_model.pt")
            wandb.log_artifact(model_artifact)
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping na época {epoch+1}")
                break

    # ✅ weights_only=True evita warning em PyTorch novo
    #model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    return model

# Integration the Sweep with train function
def train():
    wandb.init()

    sweep_params = wandb.config

    local_config = config.copy()

    local_config["model"]["hidden_sizes"] = sweep_params.hidden_sizes
    local_config["model"]["dropout"] = sweep_params.dropout
    local_config["training"]["learning_rate"] = sweep_params.learning_rate
    local_config["training"]["batch_size"] = sweep_params.batch_size

    train_df_mlp = train_df[final_features + [config["data"]["target_col"]]]
    test_df_mlp  = test_df[final_features + [config["data"]["target_col"]]]

    train_loader, test_loader, scaler = prepare_dataloaders(
        train_df_mlp,
        test_df_mlp,
        config["data"]["target_col"],
        local_config["training"]["batch_size"]
    )

    input_dim = train_loader.dataset.tensors[0].shape[1]

    model = train_model(
        local_config,
        train_loader,
        test_loader,
        input_dim
    )

    wandb.finish()


sweep_id = wandb.sweep(
    sweep_config,
    project="mlops-project-dermatology"
)

wandb.agent(
    sweep_id, 
    function=train, 
    count=10
)

# Better parameter with Sweep
import wandb
api = wandb.Api()

# Busca todos os runs do sweep e pega o melhor por val_acc
sweep = api.sweep(f"raycoelho-ufrn/mlops-project-dermatology/{sweep_id}")
best_run = sweep.best_run(order="val_acc")

print("=" * 50)
print(f"Melhor run: {best_run.name}")
print(f"val_acc:    {best_run.summary.get('val_acc', 'N/A'):.4f}")
print(f"val_loss:   {best_run.summary.get('val_loss', 'N/A'):.4f}")
print("\nMelhores hiperparâmetros:")
best_params = best_run.config
for k, v in best_params.items():
    print(f"  {k}: {v}")