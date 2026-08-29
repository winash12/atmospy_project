import yaml
from functools import wraps
import numpy as np

class ConfigContext:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)
            
        c = self._cfg["constants"]
        s = self._cfg["solver_settings"]
        
        self.CP = np.float64(c["CP"])
        self.MD = np.float64(c["MD"])
        self.R  = np.float64(c["R"])
        self.G  = np.float64(c["G"])
        self.P0 = np.float64(c["P0"])
        self.MISSING_DATA = np.float64(c["MISSING_DATA"])
        
        # Derived values computed down to machine epsilon
        self.RD = np.float64(self.R / self.MD)
        self.KAPPA = np.float64(self.RD / self.CP)
      
        # Pre-calculated constant factors for the loop-free ECMWF extrapolation shortcut
        self.R_GAMMA_G = (287.05 * 0.0065) / self.G 
        
        self.MAXLVL = int(s["MAXLVL"])
        self.DTHTA = float(s["DTHTA"])
        self.EPSLN = np.float64(s["EPSLN"])
        self.NMAX = int(s["NMAX"])
        self.MOORE_EPSILON = float(s["MOORE_EPSILON"])
        
        # --- FIX 1: Parse the string variables natively from your YAML file ---
        self.STRATEGY_MODULE = str(s["STRATEGY_MODULE"])
        self.STRATEGY_CLASS = str(s["STRATEGY_CLASS"])

env_config = ConfigContext()

def inject_constants(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.setdefault("KAPPA", env_config.KAPPA)
        kwargs.setdefault("P0", env_config.P0)
        kwargs.setdefault("MISSING_DATA", env_config.MISSING_DATA)
        kwargs.setdefault("MOORE_EPSILON", env_config.MOORE_EPSILON)
        kwargs.setdefault("R_GAMMA_G", env_config.R_GAMMA_G)
        kwargs.setdefault("MAXLVL", env_config.MAXLVL)
        kwargs.setdefault("DTHTA", env_config.DTHTA)
        kwargs.setdefault("EPSLN", env_config.EPSLN)
        kwargs.setdefault("NMAX", env_config.NMAX)
        # --- FIX 2: Inject them directly into your execution keyword mapping ---
        kwargs.setdefault("STRATEGY_MODULE", env_config.STRATEGY_MODULE)
        kwargs.setdefault("STRATEGY_CLASS", env_config.STRATEGY_CLASS)
        
        return func(*args, **kwargs)
    return wrapper

