import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import sys
import matplotlib.gridspec as gridspec

try:
    from kernels import KernelRidgeRegressor
    from uncertainty import Committee, DropoutNet
    from active_learning import active_learning_loop
except ImportError:
    print("Could not import student code. Ensure kernels.py, uncertainty.py, and active_learning.py are implemented.")
    sys.exit(1)

DOMAIN_LIMIT = 10.0
RES = 180
POOL_RES = 120
CMAP = "coolwarm"
MAX_STEPS = 200 


def ground_truth(X):
    """
    The 2D Hidden Ripple Function.
    """
    x = X[:, 0]
    y = X[:, 1]
    
    term1 = 0.5 * np.sin(0.3 * x) * np.cos(0.3 * y)
    dist_sq = (x - 5)**2 + (y - 5)**2
    term2 = 4.0 * np.exp(-1.5 * dist_sq) * np.sin(4.0 * x) * np.sin(4.0 * y)
    
    return term1 + term2


def normalize(X):
    """
    Normalize coordinates to [0,1].
    Crucial for Neural Network stability!
    """
    return X / DOMAIN_LIMIT


class FastDropoutWrapper:
    """
    Keeps student DropoutNet untouched but makes the demo interactive:
    - fewer MC passes (n_passes)
    - fewer epochs per active-learning step (epochs_first, epochs_step)
    """
    def __init__(self, net, n_passes=8, epochs_first=80, epochs_step=20, lr=0.01):
        self.net = net
        self.n_passes = n_passes
        self.epochs_first = epochs_first
        self.epochs_step = epochs_step
        self.lr = lr
        self._fit_calls = 0

    def fit(self, X, y):
        epochs = self.epochs_first if self._fit_calls == 0 else self.epochs_step
        self._fit_calls += 1
        self.net.fit(X, y, epochs=epochs, lr=self.lr)
        return self

    def predict_uncertainty(self, X):
        return self.net.predict_uncertainty(X, n_passes=self.n_passes)


def dropout_predict_mean(net, X):
    """Single forward pass with dropout OFF (fast baseline prediction)."""
    net.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        out = net(X_t).cpu().numpy().squeeze(-1)
    return out


def random_learning_loop(model, X_pool, y_pool_oracle, X_init, y_init, n_steps=10, batch_size=1, rng=None):
    """Baseline: identical loop, but selects points uniformly at random."""
    if rng is None:
        rng = np.random.default_rng(0)

    X_train = X_init.copy()
    y_train = y_init.copy()
    pool_mask = np.ones(len(X_pool), dtype=bool)

    for _ in range(n_steps):
        if not pool_mask.any():
            break

        model.fit(X_train, y_train)

        available = np.where(pool_mask)[0]
        k = min(batch_size, len(available))
        pick = rng.choice(available, size=k, replace=False)

        X_new = X_pool[pick]
        y_new = y_pool_oracle[pick]

        X_train = np.vstack([X_train, X_new])
        y_train = np.concatenate([y_train, y_new])
        pool_mask[pick] = False

        yield model, X_train, pick


def on_key(event, next_step_func, timer):
    if event.key == 'n':
        next_step_func()
    elif event.key == 'a':
        if timer[0] is None:
            timer[0] = event.canvas.new_timer(interval=700)
            timer[0].add_callback(next_step_func)
            timer[0].start()
            print("Auto-play started (press 'a' again to stop).")
        else:
            timer[0].stop()
            timer[0] = None
            print("Auto-play stopped.")


def visualize(mode, compare_random=False):
    np.random.seed(42)

    res = RES if mode == 'committee' else 140
    pool_res = POOL_RES if mode == 'committee' else 80

    # 1. Domain
    lin = np.linspace(0, DOMAIN_LIMIT, res)
    gx, gy = np.meshgrid(lin, lin)
    X_domain = np.c_[gx.ravel(), gy.ravel()]
    y_domain = ground_truth(X_domain)
    Z_gt = y_domain.reshape(res, res)

    # 2. Candidate Pool
    lin_pool = np.linspace(0, DOMAIN_LIMIT, pool_res)
    px, py = np.meshgrid(lin_pool, lin_pool)
    X_pool = np.c_[px.ravel(), py.ravel()]
    y_pool_oracle = ground_truth(X_pool)

    # 3. Initial Labeled Set
    grid_indices = np.linspace(0, pool_res - 1, 8, dtype=int)
    ix, iy = np.meshgrid(grid_indices, grid_indices)
    initial_idx = (iy * pool_res + ix).ravel()

    X_init = X_pool[initial_idx]
    y_init = y_pool_oracle[initial_idx]

    # Remove initial labeled points from candidate pool
    mask_pool = np.ones(len(X_pool), dtype=bool)
    mask_pool[initial_idx] = False
    X_pool_cand = X_pool[mask_pool]
    y_pool_cand = y_pool_oracle[mask_pool]

    # --- SETUP MODEL ---
    if mode == 'committee':
        base = KernelRidgeRegressor(gamma=4.0, alpha_reg=1e-6)
        model_active = Committee(base, n_estimators=10)
        title_str = "KRR Committee (Bagging)"
    else:
        net_a = DropoutNet(input_dim=2, hidden_dim=128, dropout_rate=0.1)
        model_active = FastDropoutWrapper(net_a, n_passes=8, epochs_first=80, epochs_step=20, lr=0.01)
        title_str = "MC Dropout Network"

    use_norm = (mode != 'committee')
    preprocess = normalize if use_norm else (lambda X: X)

    X_pool_model = preprocess(X_pool_cand)
    X_init_model = preprocess(X_init)

    max_steps = 80 if mode == 'committee' else MAX_STEPS
    batch_size = 1 if mode == 'committee' else 3

    gen_active = active_learning_loop(
        model_active, X_pool_model, y_pool_cand, X_init_model, y_init,
        n_steps=max_steps, batch_size=batch_size
    )

    if compare_random:
        if mode == 'committee':
            base_r = KernelRidgeRegressor(gamma=4.0, alpha_reg=1e-6)
            model_rand = Committee(base_r, n_estimators=10)
        else:
            net_r = DropoutNet(input_dim=2, hidden_dim=128, dropout_rate=0.1)
            model_rand = FastDropoutWrapper(net_r, n_passes=8, epochs_first=80, epochs_step=20, lr=0.01)

        rng = np.random.default_rng(123)
        gen_rand = random_learning_loop(
            model_rand, X_pool_model, y_pool_cand, X_init_model, y_init,
            n_steps=max_steps, batch_size=batch_size, rng=rng
        )

    # --- PLOT SETUP ---
    fig = plt.figure(figsize=(18, 10) if compare_random else (15, 10))
    if compare_random:
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
        ax_pred_a = fig.add_subplot(gs[0, 0])
        ax_pred_r = fig.add_subplot(gs[0, 1])
        ax_gt = fig.add_subplot(gs[0, 2])
        ax_unc_a = fig.add_subplot(gs[1, 0])
        ax_samp_r = fig.add_subplot(gs[1, 1])
        ax_metrics = fig.add_subplot(gs[1, 2])
    else:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        ax_pred_a = fig.add_subplot(gs[0, 0])
        ax_unc_a = fig.add_subplot(gs[0, 1])
        ax_gt = fig.add_subplot(gs[1, 0])
        ax_metrics = fig.add_subplot(gs[1, 1])

    extent = [0, DOMAIN_LIMIT, 0, DOMAIN_LIMIT]
    mse_hist_a = []
    mse_hist_r = []
    timer = [None]

    print(f"Starting Visualization: {title_str}")
    print(f"Initialization: 4x4 Probe ({len(X_init)} points)")

    def split_points(X_train, n_init, n_new):
        X_grid = X_train[:n_init]
        if len(X_train) > n_init + n_new:
            X_hist = X_train[n_init:-n_new]
        else:
            X_hist = np.empty((0, 2))
        X_new = X_train[-n_new:] if n_new > 0 else np.empty((0, 2))
        return X_grid, X_hist, X_new

    def next_step():
        try:
            model_fitted_a, X_train_model_a, new_idx_a = next(gen_active)

            mu_a, std_a = model_fitted_a.predict_uncertainty(preprocess(X_domain))
            X_train_a = (X_train_model_a * DOMAIN_LIMIT) if use_norm else X_train_model_a

            mse_a = np.mean((mu_a - y_domain) ** 2)
            mse_hist_a.append(mse_a)

            if compare_random:
                model_fitted_r, X_train_model_r, new_idx_r = next(gen_rand)

                if mode == 'committee':
                    mu_r, _std_r = model_fitted_r.predict_uncertainty(preprocess(X_domain))
                else:
                    mu_r = dropout_predict_mean(model_fitted_r.net, preprocess(X_domain))

                X_train_r = (X_train_model_r * DOMAIN_LIMIT) if use_norm else X_train_model_r
                mse_r = np.mean((mu_r - y_domain) ** 2)
                mse_hist_r.append(mse_r)

            # Reshape for viz
            Z_pred_a = mu_a.reshape(res, res)
            Z_unc_a = std_a.reshape(res, res)
            if compare_random:
                Z_pred_r = mu_r.reshape(res, res)

            # --- Plotting ---
            ax_pred_a.clear(); ax_gt.clear()
            if compare_random:
                ax_pred_r.clear()

            ax_pred_a.imshow(Z_pred_a, extent=extent, origin='lower', cmap=CMAP, vmin=-3, vmax=3)
            ax_pred_a.set_title(f"Active Prediction (Step {len(mse_hist_a)})")
            ax_pred_a.set_ylabel("y")

            if compare_random:
                ax_pred_r.imshow(Z_pred_r, extent=extent, origin='lower', cmap=CMAP, vmin=-3, vmax=3)
                ax_pred_r.set_title("Random Prediction")

            ax_gt.imshow(Z_gt, extent=extent, origin='lower', cmap=CMAP, vmin=-3, vmax=3)
            ax_gt.set_title("Ground Truth (Hidden Ripple)")
            ax_gt.set_xlabel("x")
            ax_gt.set_ylabel("y")

            ax_unc_a.clear(); ax_metrics.clear()
            if compare_random:
                ax_samp_r.clear()

            # Active uncertainty & samples
            ax_unc_a.imshow(Z_unc_a, extent=extent, origin='lower', cmap='magma', alpha=0.9, vmin=0)
            n_init = len(X_init)
            Xa_grid, Xa_hist, Xa_new = split_points(X_train_a, n_init, len(new_idx_a))
            ax_unc_a.scatter(Xa_grid[:, 0], Xa_grid[:, 1], c='cyan', s=30, marker='s', edgecolors='black', label='Initial Grid')
            if len(Xa_hist) > 0:
                ax_unc_a.scatter(Xa_hist[:, 0], Xa_hist[:, 1], c='white', s=15, alpha=0.6, label='History')
            if len(Xa_new) > 0:
                ax_unc_a.scatter(Xa_new[:, 0], Xa_new[:, 1], c='lime', s=250, marker='*', edgecolors='black', label='New Probe')
            ax_unc_a.set_xlim(0, DOMAIN_LIMIT)
            ax_unc_a.set_ylim(0, DOMAIN_LIMIT)
            ax_unc_a.set_title("Active: Uncertainty & Samples")
            ax_unc_a.legend(loc='upper right', fontsize='small')

            if compare_random:
                ax_samp_r.imshow(Z_gt, extent=extent, origin='lower', cmap=CMAP, vmin=-3, vmax=3, alpha=0.6)
                Xr_grid, Xr_hist, Xr_new = split_points(X_train_r, n_init, len(new_idx_r))
                ax_samp_r.scatter(Xr_grid[:, 0], Xr_grid[:, 1], c='cyan', s=30, marker='s', edgecolors='black', label='Initial Grid')
                if len(Xr_hist) > 0:
                    ax_samp_r.scatter(Xr_hist[:, 0], Xr_hist[:, 1], c='purple', s=15, alpha=0.6, label='History')
                if len(Xr_new) > 0:
                    ax_samp_r.scatter(Xr_new[:, 0], Xr_new[:, 1], c='lime', s=250, marker='*', edgecolors='black', label='New Probe')
                ax_samp_r.set_xlim(0, DOMAIN_LIMIT)
                ax_samp_r.set_ylim(0, DOMAIN_LIMIT)
                ax_samp_r.set_title("Random: Sample Locations")
                ax_samp_r.legend(loc='upper right', fontsize='small')

            # Metrics
            ax_metrics.plot(range(len(mse_hist_a)), mse_hist_a, 'r-o', linewidth=2, label='Active')
            if compare_random:
                ax_metrics.plot(range(len(mse_hist_r)), mse_hist_r, 'k-o', linewidth=2, label='Random')
                ax_metrics.legend()
            ax_metrics.set_title("Error (MSE)")
            ax_metrics.set_xlabel("Iterations")
            ax_metrics.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            fig.canvas.draw()

        except StopIteration:
            print("Loop Finished.")
            if timer[0]:
                timer[0].stop()
                timer[0] = None

    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, next_step, timer))
    next_step()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['committee', 'dropout'], default='committee')
    parser.add_argument('--compare-random', action='store_true', help='Show random sampler baseline side-by-side.')
    args = parser.parse_args()
    visualize(args.mode, compare_random=args.compare_random)
