print("MLP BASELINE")

mlp = HestonMLP().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in mlp.parameters() if p.requires_grad):,}")

mlp, mlp_history = run_training(
    mlp, train_loader, val_loader, DEVICE,
    save_path="heston_mlp_best.pt",
)

criterion = nn.MSELoss()
_, mlp_test_pred, mlp_test_true = evaluate(mlp, test_loader, criterion, DEVICE)
mae_mlp, r2_mlp = compute_metrics(mlp_test_pred, mlp_test_true)
print_metrics(mae_mlp, r2_mlp, label="MLP Test")
