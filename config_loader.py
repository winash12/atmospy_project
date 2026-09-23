import sys
import os
from functools import wraps
import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ConfigContext:
    """
    Central repository tracking parsed configuration data tables from config.toml.
    Preserves clean isolated dictionary boundaries per section header.
    """
    def __init__(self, file_name="config.toml"):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(root_dir, file_name)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[CONFIG CRITICAL] Missing mandatory TOML layout file at: {config_path}")
            
        with open(config_path, "rb") as f:
            self._cfg = tomllib.load(f)
            
        # Keep tables strictly separated to prevent mixing parameters
        self.constants_table = self._cfg.get("constants", {})
        self.solver_table = self._cfg.get("solver_settings", {})
        self.library_table = self._cfg.get("library_settings", {})

env_config = ConfigContext()


# --- DECOUPLED DECORATOR 1: CONSTANTS HEADER ONLY ---
def inject_constants(func):
    """Injects strictly physical domain parameters from the [constants] TOML table."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        c = env_config.constants_table
        kwargs.setdefault("CP", np.float64(c.get("CP", 1004.0)))
        kwargs.setdefault("MD", np.float64(c.get("MD", 28.9644)))
        kwargs.setdefault("R", np.float64(c.get("R", 8314.41)))
        kwargs.setdefault("G", np.float64(c.get("G", 9.80665)))
        kwargs.setdefault("P0", np.float64(c.get("P0", 100000.0)))
        kwargs.setdefault("MISSING_DATA", np.float64(c.get("MISSING_DATA", -9999.0)))
        
        # Calculate precision-derived physical parameters natively
        rd = np.float64(kwargs["R"] / kwargs["MD"])
        kwargs.setdefault("KAPPA", np.float64(rd / kwargs["CP"]))
        kwargs.setdefault("kappa_val", kwargs["KAPPA"]) # Parity token alias
        kwargs.setdefault("p0_val", kwargs["P0"])      # Parity token alias
        
        return func(*args, **kwargs)
    return wrapper


# --- DECOUPLED DECORATOR 2: SOLVER SETTINGS HEADER ONLY ---
def inject_solver_settings(func):
    """Injects strictly runtime parameters from the [solver_settings] TOML table."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = env_config.solver_table
        kwargs.setdefault("MAXLVL", int(s.get("MAXLVL", 50)))
        kwargs.setdefault("DTHTA", float(s.get("DTHTA", 5.0)))
        kwargs.setdefault("EPSLN", np.float64(s.get("EPSLN", 1.0)))
        kwargs.setdefault("epsln_val", kwargs["EPSLN"]) # Parity token alias
        kwargs.setdefault("NMAX", int(s.get("NMAX", 5)))
        kwargs.setdefault("nmax_val", kwargs["NMAX"])    # Parity token alias
        kwargs.setdefault("MOORE_EPSILON", float(s.get("MOORE_EPSILON", 0.005)))
        kwargs.setdefault("STRATEGY_MODULE", str(s.get("STRATEGY_MODULE", "physics_strategies")))
        kwargs.setdefault("STRATEGY_CLASS", str(s.get("STRATEGY_CLASS", "LegacySAStrategy")))
        kwargs.setdefault("SUBTERRANEAN_PHYSICS", str(s.get("SUBTERRANEAN_PHYSICS", "MISSING_VALUE")))
        kwargs.setdefault("SUBTERRANEAN_MODULE", "coordinate_transformers")
        
        return func(*args, **kwargs)
    return wrapper


# --- DECOUPLED DECORATOR 3: LIBRARY SETTINGS HEADER ONLY ---
def inject_library_settings(func):
    """Injects strictly backend processing strings from the [library_settings] TOML table."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        l = env_config.library_table
        kwargs.setdefault("P2THTA_MODULE", str(l.get("P2THTA_MODULE", "coordinate_transformers")))
        kwargs.setdefault("P2THTA_CLASS", str(l.get("P2THTA_CLASS", "IsentropicXtensorBackend")))
        kwargs.setdefault("S2THTA_MODULE", str(l.get("S2THTA_MODULE", "coordinate_transformers")))
        kwargs.setdefault("S2THTA_CLASS", str(l.get("S2THTA_CLASS", "IsobaricVelocityNumPyBackend")))
        
        return func(*args, **kwargs)
    return wrapper
