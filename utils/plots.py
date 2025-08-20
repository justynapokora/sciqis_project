import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider


def plot_fidelity_and_entropy(fidelities: np.ndarray,
                              entropies_states: np.ndarray,
                              entropies_noisy: np.ndarray,
                              n_rounds: int,
                              num_of_qubits: int):
    """
    Plot fidelities and entropies across circuit rounds.

    Args:
        fidelities:       (n+1,) array of fidelities (ideal vs noisy).
        entropies_states: (n+1,) array of entropies for ideal states (≈0).
        entropies_noisy:  (n+1,) array of entropies for noisy states.
        n_rounds:         Number of circuit rounds (int).
        num_of_qubits:    Number of qubits (for max entropy line).
    """
    rounds = np.arange(n_rounds + 1)  # 0..n (including initial)
    max_entropy = num_of_qubits  # in bits

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: Fidelity ---
    axes[0].plot(rounds, fidelities, label="Fidelity", color="tab:blue")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title("Fidelity vs Rounds")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True)

    # --- Right: Entropy ---
    axes[1].plot(rounds, entropies_states, label="Ideal (pure)", color="tab:green", linestyle="--")
    axes[1].plot(rounds, entropies_noisy, label="Noisy", color="tab:red")
    axes[1].axhline(max_entropy, color="tab:gray", linestyle=":", label=f"Max entropy ({max_entropy} bits)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Entropy (bits)")
    axes[1].set_title("Von Neumann Entropy vs Rounds")
    axes[1].set_ylim(0, max_entropy + 0.05)
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Circuit Performance over Rounds", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_fidelities_by_noise(fidelities_by_noise: dict[str, np.ndarray],
                             n_rounds: int,
                             title: str = "Fidelity vs Rounds"):
    """
    Plot fidelities (one curve per noise type) on a single figure.

    Args:
        fidelities_by_noise: dict mapping label -> (n+1,) array of fidelities
        n_rounds: number of rounds
        title: plot title
    """
    rounds = np.arange(n_rounds + 1)

    plt.figure(figsize=(8, 5))
    for label, fids in fidelities_by_noise.items():
        plt.plot(rounds, fids, label=label)
    plt.xlabel("Round")
    plt.ylabel("Fidelity")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_entropies_by_noise(entropies_noisy_by_noise: dict[str, np.ndarray],
                            n_rounds: int,
                            num_of_qubits: int,
                            title: str = "Von Neumann Entropy vs Rounds"):
    """
    Plot entropies (one curve per noise type) on a single figure,
    including ideal baseline (0) and maximal entropy line.

    Args:
        entropies_noisy_by_noise: dict mapping label -> (n+1,) array of entropies
        n_rounds: number of rounds
        num_of_qubits: number of qubits (for max entropy line)
        title: plot title
    """
    rounds = np.arange(n_rounds + 1)
    max_entropy = num_of_qubits  # in bits

    plt.figure(figsize=(8, 5))
    for label, ents in entropies_noisy_by_noise.items():
        plt.plot(rounds, ents, label=label)

    # Add ideal entropy baseline (always zero)
    plt.axhline(0, color="black", linestyle="--", linewidth=1, label="ideal (pure)")

    # Add maximal entropy line
    plt.axhline(max_entropy, color="gray", linestyle=":", linewidth=1.2,
                label=f"max entropy ({max_entropy} bits)")

    plt.xlabel("Round")
    plt.ylabel("Entropy (bits)")
    plt.title(title)
    plt.ylim(0, max_entropy + 0.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- Density matrices ----------
def prepare_density_matrices(states: np.ndarray, noisy_states: np.ndarray):
    """Precompute ideal/noisy density matrices and their differences for all rounds."""
    rhos_ideal = np.array([s @ s.conj().T for s in states], dtype=complex)
    rhos_noisy = np.asarray(noisy_states, dtype=complex)
    deltas = rhos_noisy - rhos_ideal
    return {
        "rhos_ideal": rhos_ideal,
        "rhos_noisy": rhos_noisy,
        "deltas": deltas,
        "n_rounds": states.shape[0] - 1
    }


def plot_density_matrices_bundle(rho_ideal: np.ndarray,
                                 rho_noisy: np.ndarray,
                                 delta: np.ndarray,
                                 round_idx: int,
                                 show_magnitudes: bool = False):
    """
    Plot heatmaps comparing ideal vs noisy density matrices for one round,
    with computational basis labels on axes.

    Layouts:
      show_magnitudes=False  -> 2 rows × 3 cols:
        Row 1: Re(ρ_ideal), Re(ρ_noisy), Re(Δρ)
        Row 2: Im(ρ_ideal), Im(ρ_noisy), Im(Δρ)

      show_magnitudes=True   -> 3 rows × 3 cols (adds magnitudes):
        Row 3: |ρ_ideal|, |ρ_noisy|, |Δρ|
    """
    dim = rho_ideal.shape[0]
    n_qubits = int(np.log2(dim))
    labels = [format(i, f"0{n_qubits}b") for i in range(dim)]

    if show_magnitudes:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    re, im, ab = np.real, np.imag, np.abs

    def _imshow(ax, mat, title, vmin, vmax, cmap):
        imshow = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        plt.colorbar(imshow, ax=ax)

    # --- Row 1: Real parts ---
    _imshow(axes[0, 0], re(rho_ideal), "Re(ρ_ideal)", -1, 1, "bwr")
    _imshow(axes[0, 1], re(rho_noisy), "Re(ρ_noisy)", -1, 1, "bwr")
    _imshow(axes[0, 2], re(delta), "Re(Δρ)", -1, 1, "bwr")

    # --- Row 2: Imag parts ---
    _imshow(axes[1, 0], im(rho_ideal), "Im(ρ_ideal)", -1, 1, "bwr")
    _imshow(axes[1, 1], im(rho_noisy), "Im(ρ_noisy)", -1, 1, "bwr")
    _imshow(axes[1, 2], im(delta), "Im(Δρ)", -1, 1, "bwr")

    # --- Row 3: Magnitudes (optional) ---
    if show_magnitudes:
        _imshow(axes[2, 0], ab(rho_ideal), "|ρ_ideal|", 0, 1, "viridis")
        _imshow(axes[2, 1], ab(rho_noisy), "|ρ_noisy|", 0, 1, "viridis")
        _imshow(axes[2, 2], ab(delta), "|Δρ|", 0, 1, "viridis")

    fig.suptitle(f"Density Matrices — Round {round_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()


def interactive_density_matrix_plot(states: np.ndarray,
                                    noisy_states: np.ndarray,
                                    show_magnitudes: bool = False):
    """
    Interactive visualization across rounds, with precomputation.

    Args:
        states:       (n+1, d, 1) ideal pure state vectors
        noisy_states: (n+1, d, d) noisy density matrices
        show_magnitudes: whether to include magnitude row (default: False)
    """
    precomp = prepare_density_matrices(states, noisy_states)
    n_rounds = precomp["n_rounds"]

    def _plot(round_idx: int):
        plot_density_matrices_bundle(
            precomp["rhos_ideal"][round_idx],
            precomp["rhos_noisy"][round_idx],
            precomp["deltas"][round_idx],
            round_idx,
            show_magnitudes=show_magnitudes
        )

    interact(
        _plot,
        round_idx=IntSlider(min=0, max=n_rounds, step=1, value=0, description="Round")
    )
