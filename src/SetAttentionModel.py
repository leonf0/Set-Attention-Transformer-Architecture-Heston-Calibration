class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        out, _ = self.attn(Q, K, V)
        return self.norm(Q + out)


class SAB(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha  = MultiheadAttention(d_model, n_heads, dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        H = self.mha(X, X, X)
        return self.norm(H + self.ff(H))

class PMA(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, d_ff, dropout=0.1):
        super().__init__()
        self.S    = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.mha  = MultiheadAttention(d_model, n_heads, dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        S = self.S.expand(X.size(0), -1, -1)  
        H = self.mha(S, X, X)               
        return self.norm(H + self.ff(H))


class HestonSetAttention(nn.Module):
    def __init__(self, input_dim=3, d_model=64, n_heads=4, d_ff=128,
                 n_sab_layers=2, n_seeds=4, output_dim=5, dropout=0.1):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )

        self.encoder = nn.ModuleList([
            SAB(d_model, n_heads, d_ff, dropout) for _ in range(n_sab_layers)
        ])

        self.pool = PMA(d_model, n_heads, n_seeds, d_ff, dropout)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_seeds * d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, X):
        E = self.embed(X)          
        for sab in self.encoder:
            E = sab(E)             
        P = self.pool(E)            
