"""Compare exported P(k) spectra from AxiCAMB and axionHMcode."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d


FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)
K_MIN = 1e-2
K_MAX = 5.0
PK_MIN = 1e-3

MODEL_SPECS = [
    ("pk_lin_lcdm", "k_lin_lcdm", "Linear LCDM", "#999999"),
    ("pk_lin_ax", "k_lin_ax", "Linear axion", "#d95f02"),
    ("pk_nl_ax_basic", "k_nl_ax_basic", "Non-linear axion basic", "#e7298a"),
    ("pk_nl_ax_dome", "k_nl_ax_dome", "Non-linear axion DOME", "#66a61e"),
    ("pk_nl_ax_naive", "k_nl_ax_naive", "Non-linear axion naive CAMB", "#e6ab02"),
]

COMPONENT_SPECS = [
    ("pk_lin_ax", "k_lin_ax", "Total linear axion", "#d95f02"),
    ("pk_lin_cold_ax", "k_lin_cold_ax", "Cold component", "#1b9e77"),
    ("pk_lin_axion_component", "k_lin_axion_component", "Axion component", "#7570b3"),
]


def load_export(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def check_redshifts(axicamb_data, axhm_data):
    z_axicamb = np.asarray(axicamb_data["z"], dtype=float)
    z_axhm = np.asarray(axhm_data["z"], dtype=float)
    if z_axicamb.shape != z_axhm.shape or not np.allclose(z_axicamb, z_axhm):
        raise ValueError(
            f"Mismatched redshift arrays: {z_axicamb.tolist()} vs {z_axhm.tolist()}"
        )
    return z_axicamb


def make_interp(k, pk):
    valid = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two positive finite samples for interpolation")
    return interp1d(
        np.log(k[valid]),
        np.log(pk[valid]),
        kind="linear",
        bounds_error=True,
    )


def overlapping_ratio(k1, pk1, k2, pk2, npts=500):
    kmin = max(np.min(k1), np.min(k2))
    kmax = min(np.max(k1), np.max(k2))
    if not np.isfinite(kmin) or not np.isfinite(kmax) or kmin <= 0 or kmax <= kmin:
        raise ValueError("No overlapping positive k-range for comparison")

    k_common = np.geomspace(kmin, kmax, npts)
    interp1 = make_interp(k1, pk1)
    interp2 = make_interp(k2, pk2)
    pk1_common = np.exp(interp1(np.log(k_common)))
    pk2_common = np.exp(interp2(np.log(k_common)))
    return k_common, pk1_common / pk2_common


def build_title(axicamb_data, axhm_data):
    m_ax = float(axicamb_data["m_ax"])
    f_ax = float(axicamb_data["f_ax"])
    source_a = str(np.asarray(axicamb_data["source"]).item())
    source_b = str(np.asarray(axhm_data["source"]).item())
    return f"P(k) comparison: AxiCAMB vs axionCAMB, m_ax={m_ax:.2e}, f_ax={f_ax:.3g}"


def plot_comparison(axicamb_data, axhm_data, output, model_specs=None):
    if model_specs is None:
        model_specs = MODEL_SPECS
    z_arr = check_redshifts(axicamb_data, axhm_data)
    ncols = len(z_arr)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(5.2 * ncols, 8.2),
        squeeze=False,
        sharex="col",
    )

    for col, z in enumerate(z_arr):
        ax_top = axes[0, col]
        ax_ratio = axes[1, col]

        for pk_key, k_key, label, color in model_specs:
            k_axicamb = np.asarray(axicamb_data[k_key], dtype=float)
            pk_axicamb = np.asarray(axicamb_data[pk_key][col], dtype=float)
            k_axhm = np.asarray(axhm_data[k_key], dtype=float)
            pk_axhm = np.asarray(axhm_data[pk_key][col], dtype=float)

            ax_top.plot(
                k_axicamb,
                pk_axicamb,
                color=color,
                lw=2.4,
                alpha=0.7,
                solid_capstyle="round",
            )
            ax_top.plot(
                k_axhm,
                pk_axhm,
                color=color,
                lw=1.6,
                ls=(0, (6, 3)),
                zorder=3,
            )

            k_ratio, ratio = overlapping_ratio(k_axicamb, pk_axicamb, k_axhm, pk_axhm)
            ax_ratio.plot(k_ratio, ratio, color=color, lw=2.0, label=label)

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_xlim(K_MIN, K_MAX)
        ax_top.set_ylim(bottom=PK_MIN)
        ax_top.set_title(f"z = {z:.2f}")
        ax_top.set_ylabel(r"$P(k)\,[h^{-3}\,\mathrm{Mpc}^3]$")
        ax_top.grid(alpha=0.25, which="both")

        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(K_MIN, K_MAX)
        ax_ratio.axhline(1.0, color="0.25", lw=1.0, ls=":")
        ax_ratio.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
        ax_ratio.set_ylabel("AxiCAMB / axionCAMB")
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.grid(alpha=0.25, which="both")

    model_handles = [
        Line2D([0], [0], color=color, lw=2.5, label=label)
        for _, _, label, color in model_specs
    ]
    style_handles = [
        Line2D([0], [0], color="0.2", lw=2.4, alpha=0.7, label="AxiCAMB"),
        Line2D([0], [0], color="0.2", lw=1.6, ls=(0, (6, 3)), label="axionCAMB"),
    ]
    fig.legend(model_handles, [h.get_label() for h in model_handles],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.legend(style_handles, [h.get_label() for h in style_handles],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.915))
    fig.suptitle(build_title(axicamb_data, axhm_data), y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=150)


def plot_linear_components(axicamb_data, axhm_data, output):
    z_arr = check_redshifts(axicamb_data, axhm_data)
    ncols = len(z_arr)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(5.2 * ncols, 8.2),
        squeeze=False,
        sharex="col",
    )

    for col, z in enumerate(z_arr):
        ax_top = axes[0, col]
        ax_ratio = axes[1, col]

        for pk_key, k_key, label, color in COMPONENT_SPECS:
            if pk_key not in axicamb_data or pk_key not in axhm_data:
                continue

            k_axicamb = np.asarray(axicamb_data[k_key], dtype=float)
            pk_axicamb = np.asarray(axicamb_data[pk_key][col], dtype=float)
            k_axhm = np.asarray(axhm_data[k_key], dtype=float)
            pk_axhm = np.asarray(axhm_data[pk_key][col], dtype=float)

            ax_top.plot(
                k_axicamb,
                pk_axicamb,
                color=color,
                lw=2.4,
                alpha=0.7,
                solid_capstyle="round",
            )
            ax_top.plot(
                k_axhm,
                pk_axhm,
                color=color,
                lw=1.6,
                ls=(0, (6, 3)),
                zorder=3,
            )

            k_ratio, ratio = overlapping_ratio(k_axicamb, pk_axicamb, k_axhm, pk_axhm)
            ax_ratio.plot(k_ratio, ratio, color=color, lw=2.0, label=label)

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_xlim(K_MIN, K_MAX)
        ax_top.set_ylim(bottom=PK_MIN)
        ax_top.set_title(f"z = {z:.2f}")
        ax_top.set_ylabel(r"$P(k)\,[h^{-3}\,\mathrm{Mpc}^3]$")
        ax_top.grid(alpha=0.25, which="both")

        ax_ratio.set_xscale("log")
        ax_ratio.set_xlim(K_MIN, K_MAX)
        ax_ratio.axhline(1.0, color="0.25", lw=1.0, ls=":")
        ax_ratio.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
        ax_ratio.set_ylabel("AxiCAMB / axionCAMB")
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.grid(alpha=0.25, which="both")

    component_handles = [
        Line2D([0], [0], color=color, lw=2.5, label=label)
        for _, _, label, color in COMPONENT_SPECS
    ]
    style_handles = [
        Line2D([0], [0], color="0.2", lw=2.4, alpha=0.7, label="AxiCAMB"),
        Line2D([0], [0], color="0.2", lw=1.6, ls=(0, (6, 3)), label="axionCAMB"),
    ]
    fig.legend(
        component_handles,
        [h.get_label() for h in component_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.legend(
        style_handles,
        [h.get_label() for h in style_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.915),
    )
    fig.suptitle(build_title(axicamb_data, axhm_data) + " (linear ingredients)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=150)


def component_output_path(output):
    root, ext = os.path.splitext(output)
    if not ext:
        ext = ".pdf"
    return root + "_components" + ext


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axicamb", type=str, required=True, help="Path to AxiCAMB .npz export")
    parser.add_argument(
        "--axionhmcode", type=str, required=True, help="Path to axionHMcode .npz export"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(FIGDIR, "pk_comparison.pdf"),
        help="Output figure path",
    )
    parser.add_argument(
        "--show_naive", action="store_true",
        help="Include naive CAMB nonlinear for axion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    axicamb_data = load_export(args.axicamb)
    axhm_data = load_export(args.axionhmcode)
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    model_specs = MODEL_SPECS if args.show_naive else [
        s for s in MODEL_SPECS if "naive" not in s[0]
    ]
    plot_comparison(axicamb_data, axhm_data, args.output, model_specs=model_specs)
    components_output = component_output_path(args.output)
    plot_linear_components(axicamb_data, axhm_data, components_output)
    print(f"Saved {args.output}")
    print(f"Saved {components_output}")


if __name__ == "__main__":
    main()
