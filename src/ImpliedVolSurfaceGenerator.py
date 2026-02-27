taus = np.array([
    7   / 365,
    14  / 365,
    30  / 365,
    60  / 365,
    91  / 365,
    182 / 365,
    273 / 365,
    365 / 365,
])

log_moneyness = np.linspace(-0.4, 0.2, 15)

logm_grid, tau_grid = np.meshgrid(log_moneyness, taus, indexing="ij")
grid_points = np.stack([logm_grid.ravel(), tau_grid.ravel()], axis=1)

def sample_params(n_samples, seed=42, reject_feller=True):

    bounds_lo = np.array([0.01, 0.5, 0.01, 0.1, -0.95])
    bounds_hi = np.array([0.25, 8.0, 0.25, 1.0, -0.10])

    if reject_feller:
        oversample_factor = 3
        sampler = qmc.LatinHypercube(d=5, seed=seed)
        raw = sampler.random(n=n_samples * oversample_factor)
        scaled = qmc.scale(raw, bounds_lo, bounds_hi)

        v0    = scaled[:, 0]
        kappa = scaled[:, 1]
        theta = scaled[:, 2]
        sigma = scaled[:, 3]
        rho   = scaled[:, 4]

        feller_mask = (2 * kappa * theta) > (sigma ** 2)
        params = scaled[feller_mask][:n_samples]
        feller_flags = np.ones(len(params), dtype=bool)

        if len(params) < n_samples:
            raise ValueError(
                f"Only {len(params)} Feller-satisfying samples from "
                f"{n_samples * oversample_factor} draws. Increase oversample_factor."
            )

        return params, feller_flags

    else:
        sampler = qmc.LatinHypercube(d=5, seed=seed)
        raw = sampler.random(n=n_samples)
        scaled = qmc.scale(raw, bounds_lo, bounds_hi)

        v0    = scaled[:, 0]
        kappa = scaled[:, 1]
        theta = scaled[:, 2]
        sigma = scaled[:, 3]

        feller_flags = (2 * kappa * theta) > (sigma ** 2)

        return scaled, feller_flags

# Heston Characteristic Function (Rotated / Lord-Kahl Formulation)

def heston_cf(u, tau, v0, kappa, theta, sigma, rho, r=0.0, q=0.0):

    xi = kappa - rho * sigma * 1j * u
    d  = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))

    g = (xi + d) / (xi - d)

    exp_d_tau = np.exp(d * tau)

    C = (
        (r - q) * 1j * u * tau
        + (kappa * theta / sigma**2)
        * ((xi + d) * tau - 2.0 * np.log((1.0 - g * exp_d_tau) / (1.0 - g)))
    )

    D = ((xi + d) / sigma**2) * (
        (1.0 - exp_d_tau) / (1.0 - g * exp_d_tau)
    )

    return np.exp(C + D * v0)

# Carr-Madan FFT Option Pricing

def carr_madan_call_price(S, K_arr, tau, v0, kappa, theta, sigma, rho,
                          r=0.0, q=0.0, N=4096, d_u=0.01, alpha=1.5):

    d_k = 2.0 * np.pi / (N * d_u)
    beta = np.log(S) - (N / 2) * d_k

    u_arr     = np.arange(N) * d_u
    k_arr_fft = beta + np.arange(N) * d_k

    cf_arg = u_arr - (alpha + 1) * 1j

    phi_return = heston_cf(cf_arg, tau, v0, kappa, theta, sigma, rho, r, q)
    phi = np.exp(1j * cf_arg * np.log(S)) * phi_return

    denominator = (
        alpha**2 + alpha
        - u_arr**2
        + 1j * u_arr * (2.0 * alpha + 1.0)
    )

    psi = np.exp(-r * tau) * phi / denominator

    weights = np.ones(N)
    weights[0]      = 1.0 / 3.0
    weights[-1]     = 1.0 / 3.0
    weights[1:-1:2] = 4.0 / 3.0
    weights[2:-2:2] = 2.0 / 3.0

    x = np.exp(1j * u_arr * (-beta)) * psi * weights * d_u
    fft_result = np.fft.fft(x).real

    call_prices_fft = (np.exp(-alpha * k_arr_fft) / np.pi) * fft_result

    log_K_arr = np.log(K_arr)
    call_prices = np.interp(log_K_arr, k_arr_fft, call_prices_fft)

    return np.maximum(call_prices, 0.0)

# Black-Scholes and IV Inversion

def black_scholes_call(S, K, tau, r, sigma_bs):

    if tau <= 0 or sigma_bs <= 0:
        return max(S - K * np.exp(-r * tau), 0.0)

    F  = S * np.exp(r * tau)
    d1 = (np.log(F / K) + 0.5 * sigma_bs**2 * tau) / (sigma_bs * np.sqrt(tau))
    d2 = d1 - sigma_bs * np.sqrt(tau)

    return np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol_bisection(S, K, tau, r, market_price,
                          lo=1e-6, hi=5.0, tol=1e-7, max_iter=200):
  
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if market_price < intrinsic - 1e-8:
        return np.nan

    price_lo = black_scholes_call(S, K, tau, r, lo)
    price_hi = black_scholes_call(S, K, tau, r, hi)

    if market_price < price_lo or market_price > price_hi:
        return np.nan

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price_mid = black_scholes_call(S, K, tau, r, mid)

        if abs(price_mid - market_price) < tol:
            return mid

        if price_mid < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0

# IV Surface Builder

def black_scholes_put(S, K, tau, r, sigma_bs):

    if tau <= 0 or sigma_bs <= 0:
        return max(K * np.exp(-r * tau) - S, 0.0)

    F  = S * np.exp(r * tau)
    d1 = (np.log(F / K) + 0.5 * sigma_bs**2 * tau) / (sigma_bs * np.sqrt(tau))
    d2 = d1 - sigma_bs * np.sqrt(tau)

    return np.exp(-r * tau) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol_bisection_put(S, K, tau, r, market_price,
                              lo=1e-6, hi=5.0, tol=1e-7, max_iter=200):

    intrinsic = max(K * np.exp(-r * tau) - S, 0.0)
    if market_price < intrinsic - 1e-8:
        return np.nan

    price_lo = black_scholes_put(S, K, tau, r, lo)
    price_hi = black_scholes_put(S, K, tau, r, hi)

    if market_price < price_lo or market_price > price_hi:
        return np.nan

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price_mid = black_scholes_put(S, K, tau, r, mid)

        if abs(price_mid - market_price) < tol:
            return mid

        if price_mid < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0

def build_iv_surface(params, S=100.0, r=0.0,
                     log_moneyness_grid=log_moneyness, tau_arr=taus):

    v0, kappa, theta, sigma, rho = params

    feature_rows = []
    n_failures = 0

    for tau in tau_arr:
        K_arr = S * np.exp(log_moneyness_grid)

        call_prices = carr_madan_call_price(
            S, K_arr, tau, v0, kappa, theta, sigma, rho, r=r
        )

        for log_m, K, cp in zip(log_moneyness_grid, K_arr, call_prices):
            if log_m < 0:
                # K < S: use put (OTM) via put-call parity
                put_price = cp - S + K * np.exp(-r * tau)
                put_price = max(put_price, 0.0)
                iv = implied_vol_bisection_put(S, K, tau, r, put_price)
            else:
                # K >= S: use call (OTM) directly
                iv = implied_vol_bisection(S, K, tau, r, cp)

            if np.isnan(iv):
                n_failures += 1

            feature_rows.append([log_m, np.sqrt(tau), iv])

    return np.array(feature_rows), n_failures

# Batch Surface Generation 

def generate_training_data(n_samples=100_000, seed=42, reject_feller=True,
                           noise_std=0.0, S=100.0, r=0.0):

    params_all, feller_flags = sample_params(n_samples, seed=seed,
                                             reject_feller=reject_feller)

    n_grid = len(log_moneyness) * len(taus)
    features = np.empty((n_samples, n_grid, 3))
    total_failures = 0

    rng = np.random.default_rng(seed + 1) 

    for i in range(n_samples):
        surface, n_fail = build_iv_surface(params_all[i], S=S, r=r)
        total_failures += n_fail

        # Optionally add noise to IV values
        if noise_std > 0:
            noise = rng.normal(0, noise_std, size=surface.shape[0])
            surface[:, 2] = surface[:, 2] * (1.0 + noise)
            surface[:, 2] = np.maximum(surface[:, 2], 1e-4)  

        features[i] = surface

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_samples} surfaces "
                  f"({total_failures} IV failures so far)")

    meta = {
        "n_samples": n_samples,
        "n_grid_points": n_grid,
        "total_iv_failures": total_failures,
        "failure_rate": total_failures / (n_samples * n_grid),
        "noise_std": noise_std,
        "feller_violation_rate": 1.0 - feller_flags.mean(),
    }

    return features, params_all, feller_flags, meta

# Label Transforms 

def transform_labels(params):

    transformed = np.empty_like(params)
    transformed[:, 0] = np.log(params[:, 0])        
    transformed[:, 1] = np.log(params[:, 1])        
    transformed[:, 2] = np.log(params[:, 2])        
    transformed[:, 3] = np.log(params[:, 3])        
    transformed[:, 4] = np.arctanh(params[:, 4]) 

    return transformed


def inverse_transform_labels(transformed):

    params = np.empty_like(transformed)
    params[:, 0] = np.exp(transformed[:, 0])
    params[:, 1] = np.exp(transformed[:, 1])
    params[:, 2] = np.exp(transformed[:, 2])
    params[:, 3] = np.exp(transformed[:, 3])
    params[:, 4] = np.tanh(transformed[:, 4])
    return params

def generate_dataset(n_samples=100_000, seed=42):

    params_arr, feller_flags = sample_params(n_samples, seed=seed)

    X_list = []
    y_list = []
    skipped = 0

    for i, params in enumerate(params_arr):
        if i % 1000 == 0:
            print(f"Generating sample {i}/{n_samples}")

        surface, n_fail = build_iv_surface(
            params, log_moneyness_grid=log_moneyness, tau_arr=taus
        )

        if np.any(np.isnan(surface[:, 2])):
            skipped += 1
            continue
          
        # Check IV surface is arbitrage-free
        ivs = surface[:, 2]
        if np.any(ivs < 0.01) or np.any(ivs > 3.0):
            skipped += 1
            continue

        X_list.append(surface)
        y_list.append(params)

    print(f"Skipped {skipped} samples ({100*skipped/n_samples:.1f}%)")

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y
