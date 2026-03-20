"""
Default cosmological parameters for AxiCAMB vs AxiECAMB comparisons.

All comparison scripts should import from here to avoid parameter mismatches.
"""

# Planck 2018 baseline
DEFAULT = {
    'H0': 67.32,
    'ombh2': 0.022383,
    'omch2_total': 0.12011,  # total CDM+axion density
    'ns': 0.96605,
    'As': 2.10058e-9,
    'tau': 0.0543,
    'mnu': 0.06,
    'YHe': 0.245861,
    'Neff': 3.046,
    'kmax': 50.0,
}

# Axion defaults
AXION_DEFAULT = {
    'm_ax': 1e-24,
    'f_ax': 0.3,
    'use_PH': True,
    'movH_switch': 50.0,  # m/H at fluid switch (mH in AxiCAMB, movH_switch in AxiECAMB)
}


def get_axicamb_kwargs(cosmo=None, axion=None):
    """Get kwargs for axicamb_runner.run()."""
    c = {**DEFAULT, **(cosmo or {})}
    a = {**AXION_DEFAULT, **(axion or {})}
    return {
        'm_ax': a['m_ax'],
        'f_ax': a['f_ax'],
        'H0': c['H0'],
        'ombh2': c['ombh2'],
        'omch2_total': c['omch2_total'],
        'ns': c['ns'],
        'As': c['As'],
        'tau': c['tau'],
        'kmax': c['kmax'],
        'use_PH': a['use_PH'],
        'mH': a['movH_switch'],
        'mnu': c['mnu'],
    }


def get_axiecamb_kwargs(cosmo=None, axion=None):
    """Get kwargs for axiecamb_runner.run()."""
    c = {**DEFAULT, **(cosmo or {})}
    a = {**AXION_DEFAULT, **(axion or {})}
    return {
        'm_ax': a['m_ax'],
        'f_ax': a['f_ax'],
        'H0': c['H0'],
        'ombh2': c['ombh2'],
        'omdah2': c['omch2_total'],
        'ns': c['ns'],
        'As': c['As'],
        'tau': c['tau'],
        'kmax': c['kmax'],
        'YHe': c['YHe'],
        'omnuh2': c['mnu'] / 93.14 if c['mnu'] > 0 else 0.0,
        'Neff': c['Neff'] - 1.0 if c['mnu'] > 0 else c['Neff'],
        'massive_neutrinos': 1 if c['mnu'] > 0 else 0,
        'movH_switch': a['movH_switch'],
    }


def get_lcdm_kwargs(cosmo=None):
    """Get kwargs for axicamb_runner.get_lcdm_pk()."""
    c = {**DEFAULT, **(cosmo or {})}
    return {
        'H0': c['H0'],
        'ombh2': c['ombh2'],
        'omch2': c['omch2_total'],
        'ns': c['ns'],
        'As': c['As'],
        'tau': c['tau'],
        'kmax': c['kmax'],
        'mnu': c['mnu'],
    }


def add_cli_args(parser):
    """Add common cosmology CLI arguments to an argparse parser."""
    parser.add_argument('--m_ax', type=float, default=AXION_DEFAULT['m_ax'])
    parser.add_argument('--f_ax', type=float, default=AXION_DEFAULT['f_ax'])
    parser.add_argument('--mnu', type=float, default=DEFAULT['mnu'])
    parser.add_argument('--movH_switch', type=float, default=AXION_DEFAULT['movH_switch'],
                        help='m/H switch for fluid approximation (default: 50)')
    parser.add_argument('--H0', type=float, default=DEFAULT['H0'])
    parser.add_argument('--ombh2', type=float, default=DEFAULT['ombh2'])
    parser.add_argument('--omch2', type=float, default=DEFAULT['omch2_total'])
    parser.add_argument('--ns', type=float, default=DEFAULT['ns'])
    parser.add_argument('--As', type=float, default=DEFAULT['As'])
    parser.add_argument('--tau', type=float, default=DEFAULT['tau'])
    return parser


def from_args(args):
    """Build cosmo and axion dicts from parsed CLI args."""
    cosmo = {
        'H0': args.H0,
        'ombh2': args.ombh2,
        'omch2_total': args.omch2,
        'ns': args.ns,
        'As': args.As,
        'tau': args.tau,
        'mnu': args.mnu,
    }
    axion = {
        'm_ax': args.m_ax,
        'f_ax': args.f_ax,
        'movH_switch': args.movH_switch,
    }
    return cosmo, axion
