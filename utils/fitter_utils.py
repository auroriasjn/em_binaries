import numpy as np
from analyzer import BinaryMixtureFitter


def bootstrap_mixture_weights(fitter, theta, n_boot=200, random_state=None):
    rng = np.random.default_rng(random_state)
    weights = []

    df = fitter.get_data()
    N = len(df)

    for _ in range(n_boot):
        # resample indices
        idx = rng.integers(0, N, size=N)
        boot_data = df.iloc[idx].reset_index(drop=True)

        boot_fitter = BinaryMixtureFitter(
            boot_data,
            fB=fitter.get_binary_fraction(),
            mass_ratio=fitter.mass_ratio,
            q_range=(fitter.qmin, fitter.qmax),
            field_weight=(fitter.mixture_weights[2] if fitter.use_field else 0.0),
            use_field=fitter.use_field,
            tau=fitter.tau,
        )

        boot_fitter.fit(theta, n_iterations=40)
        weights.append(boot_fitter.get_mixture_weights())

    weights = np.vstack(weights)
    mean = weights.mean(axis=0)
    std  = weights.std(axis=0)

    lower = np.percentile(weights, 16, axis=0)
    upper = np.percentile(weights, 84, axis=0)

    return {
        "samples": weights,
        "mean": mean,
        "std": std,
        "ci68": np.stack([lower, upper], axis=1),
    }
