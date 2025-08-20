import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from ipywidgets import interact, IntSlider

from utils.circuit_metrics import haar_fidelity_pdf, haar_bin_masses, pairwise_fidelities_consecutive, \
    pairwise_uhlmann_fidelities_consecutive, frame_potentials_from_samples


# ---------- Fidelity and entropy ----------
def plot_fidelity_and_entropy(fidelities: np.ndarray,
                              entropies_states: np.ndarray,
                              entropies_noisy: np.ndarray,
                              n_rounds: int,
                              num_of_qubits: int):
    """
    Plot fidelities and entropies across circuit rounds.
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
    """
    Precompute ideal/noisy density matrices and their differences for all rounds.
    """
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


# ---------- Bloch sphere ----------
def _bloch_vectors_from_states(states_1c: np.ndarray) -> np.ndarray:
    """
    Convert (S,2,1) or (S,2) pure statevectors to Bloch vectors (S,3).
    """
    psi = states_1c.reshape(states_1c.shape[0], 2)  # (S,2)
    a = psi[:, 0]
    b = psi[:, 1]

    # Normalize defensively
    norms = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
    norms[norms == 0] = 1.0
    a = a / norms
    b = b / norms

    # Bloch components
    rx = 2 * np.real(np.conjugate(a) * b)
    ry = 2 * np.imag(np.conjugate(a) * b)
    rz = np.abs(a) ** 2 - np.abs(b) ** 2
    return np.column_stack([rx, ry, rz])


def _bloch_vectors_from_density(noisy_1c: np.ndarray) -> np.ndarray:
    """
    Convert (S,2,2) density matrices to Bloch vectors (S,3):
      r_x = 2 Re(rho01), r_y = 2 Im(rho01), r_z = rho00 - rho11
    Assumes trace≈1; renormalizes trace if needed.
    """
    rho = np.asarray(noisy_1c, dtype=complex)
    if rho.ndim != 3 or rho.shape[1:] != (2, 2):
        raise ValueError(f"Expected (S,2,2) density matrices, got {rho.shape}")

    # Hermitize & trace-normalize (numerical hygiene)
    rho = 0.5 * (rho + np.conjugate(np.swapaxes(rho, 1, 2)))
    tr = np.trace(rho, axis1=1, axis2=2).reshape(-1, 1, 1)
    tr[np.isclose(tr, 0)] = 1.0
    rho = rho / tr

    rho00 = rho[:, 0, 0]
    rho11 = rho[:, 1, 1]
    rho01 = rho[:, 0, 1]

    rx = 2.0 * np.real(rho01)
    ry = 2.0 * np.imag(rho01)
    rz = np.real(rho00 - rho11)
    return np.column_stack([rx, ry, rz])


def _draw_unit_sphere(ax):
    """
    Wireframe + axes + labels for a Bloch sphere.
    """
    # --- sphere wireframe (radius = 1) ---
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.5, alpha=0.25)

    # --- axis lines slightly longer than the sphere ---
    axis_pad = 0.08
    L = 1.0 + float(axis_pad)
    ax.plot([-L, L], [0, 0], [0, 0], linewidth=1, alpha=0.5)  # X
    ax.plot([0, 0], [-L, L], [0, 0], linewidth=1, alpha=0.5)  # Y
    ax.plot([0, 0], [0, 0], [-L, L], linewidth=1, alpha=0.5)  # Z

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)

    # centered axis labels (with white outline for readability)
    label_kwargs = dict(ha="center", va="center", fontsize=10, color="black",
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    label_pos = 0.75
    ax.text(label_pos, 0.0, 0.0, "X", zorder=10, clip_on=False, **label_kwargs)
    ax.text(0.0, label_pos, 0.0, "Y", zorder=10, clip_on=False, **label_kwargs)
    ax.text(0.0, 0.0, label_pos, "Z", zorder=10, clip_on=False, **label_kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])


def plot_bloch_spheres_for_states(states: np.ndarray,
                                  noisy_states: np.ndarray | None = None,
                                  circuit_labels: list[str] | None = None,
                                  max_points_per_circuit: int = 2000,
                                  stride: int | None = None,
                                  point_size: float = 8.0,
                                  alpha: float = 0.8,
                                  color_ideal: str = "C0",
                                  color_noisy: str = "C3",
                                  marker_ideal: str = "o",
                                  marker_noisy: str = "^"):
    """
    Plot Bloch spheres for multiple circuits.
    If `noisy_states` is provided, draws 3 rows: Ideal / Noisy / Both (overlay).
    """
    num_circuits, samples, dim = states.shape[0], states.shape[1], states.shape[2]
    if dim != 2:
        raise ValueError(f"Single-qubit only (dim=2). Got dim={dim}.")
    have_noisy = noisy_states is not None

    if have_noisy:
        if noisy_states.shape[0] != num_circuits or noisy_states.shape[1] != samples:
            raise ValueError("states and noisy_states must align on (num_circuits, samples).")
        if noisy_states.shape[2:] != (2, 2):
            raise ValueError(f"noisy_states must be (C,S,2,2); got {noisy_states.shape}.")

    if circuit_labels is None:
        circuit_labels = [f"Circuit {i + 1}" for i in range(num_circuits)]

    # Subsample indices (shared for ideal & noisy)
    base_idx = np.arange(samples)
    if stride and stride > 1:
        base_idx = base_idx[::stride]
    if base_idx.size > max_points_per_circuit:
        base_idx = np.linspace(0, base_idx.size - 1, max_points_per_circuit, dtype=int)

    # Precompute Bloch vectors once per circuit (avoid recomputation across rows)
    bloch_i_list = [_bloch_vectors_from_states(states[ci, base_idx]) for ci in range(num_circuits)]
    bloch_n_list = [_bloch_vectors_from_density(noisy_states[ci, base_idx]) for ci in range(num_circuits)] if have_noisy else None

    rows = 3 if have_noisy else 1
    fig = plt.figure(figsize=(4.8 * num_circuits, 4.3 * rows))

    axes = []
    for r in range(rows):
        for c in range(num_circuits):
            ax = fig.add_subplot(rows, num_circuits, r * num_circuits + c + 1, projection='3d')
            _draw_unit_sphere(ax)
            axes.append(ax)

    # Row 1: Ideal
    for ci in range(num_circuits):
        ax = axes[ci]
        bi = bloch_i_list[ci]
        ax.scatter(bi[:, 0], bi[:, 1], bi[:, 2],
                   s=point_size, alpha=alpha, depthshade=False, color=color_ideal, marker=marker_ideal)
        ax.set_title(f"{circuit_labels[ci]} — ideal")

    if have_noisy:
        # Row 2: Noisy
        offset = num_circuits
        for ci in range(num_circuits):
            ax = axes[offset + ci]
            bn = bloch_n_list[ci]
            ax.scatter(bn[:, 0], bn[:, 1], bn[:, 2],
                       s=point_size, alpha=alpha, depthshade=False, color=color_noisy, marker=marker_noisy)
            ax.set_title(f"{circuit_labels[ci]} — noisy")

        # Row 3: Both (overlay)
        offset = 2 * num_circuits
        for ci in range(num_circuits):
            ax = axes[offset + ci]
            bi = bloch_i_list[ci]
            bn = bloch_n_list[ci]
            ax.scatter(bi[:, 0], bi[:, 1], bi[:, 2],
                       s=point_size, alpha=alpha, depthshade=False, color=color_ideal, marker=marker_ideal, label="ideal")
            ax.scatter(bn[:, 0], bn[:, 1], bn[:, 2],
                       s=point_size, alpha=alpha, depthshade=False, color=color_noisy, marker=marker_noisy, label="noisy")
            ax.set_title(f"{circuit_labels[ci]} — both")
            if ci == 0:
                ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# ---------- Fidelity histogram ----------
def plot_fidelity_histograms_vs_haar(states: np.ndarray,
                                     noisy_states: np.ndarray | None = None,
                                     circuit_labels: list[str] | None = None,
                                     num_bins: int = 75,
                                     y_mode: str = "probability",  # "probability" or "density"
                                     log_base: str = "e"  # "e" or "2" for bits
                                     ):
    """
    If noisy_states is provided, plots 3 rows per circuit:
      Row 1: ideal vs Haar  (bars + Haar overlay)  → KL(ideal‖Haar)
      Row 2: noisy vs Haar  (bars + Haar overlay)  → KL(noisy‖Haar)
      Row 3: ideal vs noisy (overlay only; no Haar)
    """
    C, S, dim = states.shape[0], states.shape[1], states.shape[2]
    if circuit_labels is None:
        circuit_labels = [f"Circuit {i + 1}" for i in range(C)]
    have_noisy = noisy_states is not None

    # Build pairwise fidelities
    fids_ideal = [pairwise_fidelities_consecutive(states[c]) for c in range(C)]
    fids_noisy = [pairwise_uhlmann_fidelities_consecutive(noisy_states[c]) for c in range(C)] if have_noisy else [                                                                                                     None] * C

    # Shared bins and Haar bin masses
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    q = haar_bin_masses(bin_edges, dim)
    eps = 1e-12
    q = np.clip(q, eps, 1.0);
    q /= q.sum()

    ln_conv = 1.0 if log_base == "e" else (1.0 / np.log(2.0))
    rows = 3 if have_noisy else 1
    cols = C
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)

    KLs_ideal = np.zeros(C, dtype=float)
    KLs_noisy = np.zeros(C, dtype=float) if have_noisy else None

    # ----- Row 1: ideal vs Haar -----
    for c in range(C):
        ax = axes[0, c]
        fids = fids_ideal[c]

        # per-bin probabilities p for KL
        counts, _ = np.histogram(fids, bins=bin_edges)
        p = counts.astype(float);
        p /= max(p.sum(), 1.0)
        mask = p > 0
        KL_ideal = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))) * ln_conv
        KLs_ideal[c] = KL_ideal

        if y_mode.lower() == "probability":
            weights = np.full_like(fids, 1.0 / fids.size, dtype=float)
            ax.hist(fids, bins=bin_edges, weights=weights, alpha=0.65, label="ideal")
            try:
                ax.stairs(q, bin_edges, color="r", linestyle="--", linewidth=2, label="Haar")
            except AttributeError:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.step(centers, q, where="mid", color="r", linestyle="--", linewidth=2, label="Haar")
            ax.set_ylabel("Probability")
        else:
            ax.hist(fids, bins=bin_edges, density=True, alpha=0.65, label="ideal")
            Fgrid = np.linspace(0, 1, 400)
            ax.plot(Fgrid, haar_fidelity_pdf(Fgrid, dim), 'r--', lw=2, label="Haar PDF")
            ax.set_ylabel("Density")

        ax.set_xlim(0, 1)
        ax.set_title(f"{circuit_labels[c]} — KL(ideal‖Haar) = {KL_ideal:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if have_noisy:
        # ----- Row 2: noisy vs Haar -----
        for c in range(C):
            ax = axes[1, c]
            fids = fids_noisy[c]

            counts, _ = np.histogram(fids, bins=bin_edges)
            p = counts.astype(float)
            p /= max(p.sum(), 1.0)
            mask = p > 0
            KL_noisy = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))) * ln_conv
            KLs_noisy[c] = KL_noisy

            if y_mode.lower() == "probability":
                weights = np.full_like(fids, 1.0 / fids.size, dtype=float)
                ax.hist(fids, bins=bin_edges, weights=weights, alpha=0.65, color="C1", label="noisy")
                try:
                    ax.stairs(q, bin_edges, color="r", linestyle="--", linewidth=2, label="Haar")
                except AttributeError:
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.step(centers, q, where="mid", color="r", linestyle="--", linewidth=2, label="Haar")
                ax.set_ylabel("Probability")
            else:
                ax.hist(fids, bins=bin_edges, density=True, alpha=0.65, color="C1", label="noisy")
                Fgrid = np.linspace(0, 1, 400)
                ax.plot(Fgrid, haar_fidelity_pdf(Fgrid, dim), 'r--', lw=2, label="Haar PDF")
                ax.set_ylabel("Density")

            ax.set_xlim(0, 1)
            ax.set_title(f"{circuit_labels[c]} — KL(noisy‖Haar) = {KL_noisy:.3f}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # ----- Row 3: ideal vs noisy (overlay, no Haar) -----
        for c in range(C):
            ax = axes[2, c]
            fids_i = fids_ideal[c]
            fids_n = fids_noisy[c]

            # convert both to per-bin probabilities for a clean apples-to-apples overlay
            counts_i, _ = np.histogram(fids_i, bins=bin_edges);
            p_i = counts_i.astype(float);
            p_i /= max(p_i.sum(), 1.0)
            counts_n, _ = np.histogram(fids_n, bins=bin_edges);
            p_n = counts_n.astype(float);
            p_n /= max(p_n.sum(), 1.0)

            try:
                ax.stairs(p_i, bin_edges, color="C0", linewidth=2, label="ideal")
                ax.stairs(p_n, bin_edges, color="C1", linewidth=2, label="noisy")
            except AttributeError:
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.step(centers, p_i, where="mid", color="C0", linewidth=2, label="ideal")
                ax.step(centers, p_n, where="mid", color="C1", linewidth=2, label="noisy")

            ax.set_xlim(0, 1)
            ax.set_xlabel(r"Fidelity $F = |\langle \psi | \phi \rangle|^2$")
            ax.set_title(f"{circuit_labels[c]} — ideal vs noisy")
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.legend()

    if not have_noisy:
        for c in range(C):
            axes[0, c].set_xlabel(r"Fidelity $F = |\langle \psi | \phi \rangle|^2$")

    fig.suptitle("Pairwise Fidelity Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

    return KLs_ideal, KLs_noisy, bin_edges


# ---------- Frame potentials ----------
def plot_frame_potentials(states: np.ndarray,
                          noisy_states: np.ndarray | None = None,
                          circuit_labels: list[str] | None = None,
                          t_max: int = 4):
    """
    Plot frame potentials (moments 1..t_max) per circuit, showing:
      - ideal (pure) estimates,
      - noisy (mixed, via Uhlmann) estimates if provided,
      - Haar (Welch bound) reference.
    """
    FP_ideal, FP_noisy, FP_haar = frame_potentials_from_samples(states, noisy_states, t_max=t_max)

    C = FP_ideal.shape[0]
    if circuit_labels is None:
        circuit_labels = [f"Circuit {i + 1}" for i in range(C)]

    # Layout: one column per circuit
    x = np.arange(1, t_max + 1)
    fig, axes = plt.subplots(1, C, figsize=(4.6 * C, 3.6), squeeze=False)
    axes = axes[0]

    for c in range(C):
        ax = axes[c]
        ax.plot(x, FP_ideal[c], marker='o', linestyle='-', label='ideal')
        if FP_noisy is not None:
            ax.plot(x, FP_noisy[c], marker='s', linestyle='--', label='noisy')

        # Haar baseline (same for all circuits of same dim)
        ax.plot(x, FP_haar, linestyle=':', color='purple', label='Haar')

        ax.set_xticks(x)
        ax.set_xlabel("moment $t$")
        ax.set_title(circuit_labels[c])
        ax.grid(True, alpha=0.3)

        # Y label on the first subplot
        if c == 0:
            ax.set_ylabel(r"frame potential $\mathcal{F}^{(t)}=\mathbb{E}[F^t]$")

        # Only show a legend on the first subplot to save space
        if c == 0:
            ax.legend()

    fig.suptitle("Frame potentials (moments 1–4) per circuit", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------- Expressibility and entangling capability  ----------
def _shorten_labels(labels):
    """
    Get Ciruit ids from labels
    """
    out = []
    for s in labels:
        t = s
        if "Circuit" in t:
            t = t.split("Circuit", 1)[1]
        out.append(t.strip())
    return out


def _default_cycle_colors(L: int):
    """
    Define plot colors
    """
    # Use Matplotlib's default color cycle, repeated as needed
    cycle = plt.rcParams.get('axes.prop_cycle', None)
    if cycle is None:
        # Fallback to C0..C9
        colors = [f"C{i}" for i in range(L)]
    else:
        base = cycle.by_key().get('color', [])
        colors = [base[i % len(base)] for i in range(L)] if base else [f"C{i}" for i in range(L)]
    return colors


def _layer_markers(L: int):
    """
    Define plot markers
    """
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', '<', '>']
    return [markers[i % len(markers)] for i in range(L)]


def plot_expressibility_layers_by_circuit(expressibilities: np.ndarray,
                                          expressibilities_noisy: np.ndarray,
                                          labels: list[str],
                                          title: str = "Expressibility (KL to Haar) vs Circuit",
                                          log_scale: bool = True,
                                          jitter: float = 0.10):
    """
    Scatter-only (no connecting lines) expressibility per circuit & layer.
    - Circuits sorted by L=1 ideal KL ascending.
    - Optional log y-scale.
    - Original look: default color cycle + per-layer markers.
    - Legend below; no overlap.
    """
    KL_i = np.asarray(expressibilities, dtype=float)
    KL_n = np.asarray(expressibilities_noisy, dtype=float)
    C, L = KL_i.shape

    short = _shorten_labels(labels)
    sort_idx = np.argsort(KL_i[:, 0])
    KL_i, KL_n = KL_i[sort_idx], KL_n[sort_idx]
    short = [short[i] for i in sort_idx]

    x = np.arange(C)
    colors = _default_cycle_colors(L)
    # swap 2nd and 3rd colors once
    if L >= 3:
        colors[1], colors[2] = colors[2], colors[1]
    markers = _layer_markers(L)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.9 * C + 5), 5), sharey=True)
    ax_i, ax_n = axes

    eps = 1e-12 if log_scale else 0.0

    for l in range(L):
        x_off = x + (l - (L - 1) / 2) * jitter
        lbl = f"L={l+1}"
        mk = markers[l]
        col = colors[l]

        ax_i.scatter(x_off, KL_i[:, l] + eps, s=46, marker=mk, color=col, alpha=0.95, label=lbl)
        ax_n.scatter(x_off, KL_n[:, l] + eps, s=46, marker=mk, color=col, alpha=0.95, label=lbl)

    for ax, subtitle in zip(axes, ["ideal", "noisy"]):
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=0)
        ax.set_xlabel("Circuit")
        ax.set_title(subtitle, pad=6)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')

    axes[0].set_ylabel("KL divergence to Haar")

    # Single legend below both plots
    handles, labelsL = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labelsL, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(L, 8), frameon=False)

    fig.suptitle(title, fontsize=14, y=0.98)
    # Reserve space at bottom for the legend and a bit at top for the suptitle.
    fig.subplots_adjust(bottom=0.20, top=0.90, wspace=0.10)
    plt.show()


def plot_entangling_capability_layers_by_circuit(entangling_abilities: np.ndarray,
                                                 entangling_abilities_noisy: np.ndarray,
                                                 labels: list[str],
                                                 num_qubits: int,
                                                 title: str = "Entangling capability (Meyer–Wallach Q) vs Circuit",
                                                 jitter: float = 0.10):
    """
    Scatter-only entangling capability per circuit & layer.
    - Circuits sorted by L=1 ideal Q ascending.
    - Haar baseline for pure states.
    - Original look: default color cycle + per-layer markers.
    - Legend below; no overlap.
    """
    Q_i = np.asarray(entangling_abilities, dtype=float)
    Q_n = np.asarray(entangling_abilities_noisy, dtype=float)
    C, L = Q_i.shape

    short = _shorten_labels(labels)
    sort_idx = np.argsort(Q_i[:, 0])
    Q_i, Q_n = Q_i[sort_idx], Q_n[sort_idx]
    short = [short[i] for i in sort_idx]

    N = 2 ** num_qubits
    Q_haar = (N - 2) / (N + 1)

    x = np.arange(C)
    colors = _default_cycle_colors(L)
    if L >= 3:
        colors[1], colors[2] = colors[2], colors[1]
    markers = _layer_markers(L)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.9 * C + 5), 5), sharey=True)
    ax_i, ax_n = axes

    for l in range(L):
        x_off = x + (l - (L - 1) / 2) * jitter
        lbl = f"L={l+1}"
        mk = markers[l]
        col = colors[l]

        ax_i.scatter(x_off, Q_i[:, l], s=46, marker=mk, color=col, alpha=0.95, label=lbl)
        ax_n.scatter(x_off, Q_n[:, l], s=46, marker=mk, color=col, alpha=0.95, label=lbl)

    for ax, subtitle in zip(axes, ["ideal", "noisy"]):
        ax.axhline(Q_haar, linestyle=':', linewidth=1.8, label=r"Haar baseline $\mathbb{E}[Q]$")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=0)
        ax.set_xlabel("Circuit")
        ax.set_title(subtitle, pad=6)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Entangling capability $\overline{Q}$")
    axes[0].set_ylim(0, 1.02)

    handles, labelsL = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labelsL, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(L + 1, 9), frameon=False)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(bottom=0.20, top=0.90, wspace=0.10)
    plt.show()


def plot_expressibility_vs_layers_for_circuit(
    KL_i_row: np.ndarray,
    KL_n_row: np.ndarray,
    circuit_label: str = "",
    log_scale: bool = True
):
    """
    Plot expressibility for a single circuit vs layers.
    - X axis: layer index (1..L)
    - Y axis: KL divergence to Haar
    - Points use the SAME per-layer markers & default colors (with 2nd/3rd color swapped).
    - Lines connect layers: solid black (ideal) and dashed gray (noisy).
    """
    # Ensure 1D arrays
    KL_i_row = np.asarray(KL_i_row, dtype=float).ravel()
    KL_n_row = np.asarray(KL_n_row, dtype=float).ravel()
    assert KL_i_row.shape == KL_n_row.shape, "ideal and noisy rows must have same length"
    L = KL_i_row.size

    # Reuse your helpers
    colors = _default_cycle_colors(L)
    # Apply your hard swap of 2nd and 3rd colors globally
    if L >= 3:
        colors[1], colors[2] = colors[2], colors[1]
    markers = _layer_markers(L)

    x = np.arange(1, L + 1)
    eps = 1e-12 if log_scale else 0.0

    fig, ax = plt.subplots(figsize=(7, 4.6))

    # Connecting lines
    ax.plot(x, KL_i_row + eps, color="black", linewidth=1.6, label="ideal")
    ax.plot(x, KL_n_row + eps, color="gray", linestyle="--", linewidth=1.6, label="noisy")

    # Per-layer markers (same shapes/colors as your panel plots)
    for l in range(L):
        col = colors[l]
        mk  = markers[l]
        # ideal = filled marker
        ax.scatter(x[l], KL_i_row[l] + eps, s=60, marker=mk, color=col, alpha=0.95, zorder=3)
        # noisy = hollow marker (same color/shape)
        ax.scatter(x[l], KL_n_row[l] + eps, s=60, marker=mk, facecolors="none",
                   edgecolors=col, linewidths=1.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={k}" for k in x])
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence to Haar")
    if circuit_label:
        ax.set_title(f"Expressibility vs Layer — {circuit_label}", pad=8)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")

    # Simple legend for ideal/noisy lines
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    plt.show()
