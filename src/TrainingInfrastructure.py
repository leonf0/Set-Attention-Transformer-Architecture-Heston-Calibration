def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item() * len(X_batch)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(all_pred), np.concatenate(all_true)


def compute_metrics(pred_transformed: np.ndarray, true_transformed: np.ndarray):
    """MAE and R² on the original (physical) scale."""
    pred = inverse_transform_labels(pred_transformed)
    true = inverse_transform_labels(true_transformed)

    mae    = np.abs(pred - true).mean(axis=0)
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    r2     = 1.0 - ss_res / ss_tot

    return mae, r2


def print_metrics(mae, r2, label=""):
    print(f"\n{'='*52}")
    print(f"  {label} Metrics (original scale)")
    print(f"{'='*52}")
    print(f"  {'Param':<10} {'MAE':>12} {'R²':>12}")
    print(f"  {'-'*36}")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:<10} {mae[i]:>12.5f} {r2[i]:>12.4f}")
    print(f"  {'-'*36}")
    print(f"  {'Mean':<10} {mae.mean():>12.5f} {r2.mean():>12.4f}")


def run_training(model, train_loader, val_loader, device,
                 lr=3e-4, max_epochs=100, patience=10, save_path="best.pt"):
    """Generic training loop — shared by MLP and Set Attention."""
    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5, verbose=True,
    )

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_loss": []}

    print(f"Training for up to {max_epochs} epochs (patience={patience})…")
    for epoch in range(1, max_epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train={train_loss:.6f}  val={val_loss:.6f} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(save_path))
    return model, history

SEED       = 42
N_SAMPLES  = 50_000
BATCH_SIZE = 512
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

print(f"\nGenerating {N_SAMPLES} IV surfaces…")
t0 = time.time()
X_raw, y_raw = generate_dataset(n_samples=N_SAMPLES, seed=SEED)
print(f"Done in {time.time()-t0:.1f}s  |  X: {X_raw.shape}  y: {y_raw.shape}")

y_transformed = transform_labels(y_raw).astype(np.float32)
X_raw = X_raw.astype(np.float32)

dataset = HestonSurfaceDataset(X_raw, y_transformed)
n       = len(dataset)
n_test  = int(n * TEST_FRAC)
n_val   = int(n * VAL_FRAC)
n_train = n - n_val - n_test

train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Split — train: {n_train}  val: {n_val}  test: {n_test}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
