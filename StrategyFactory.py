import yaml
import importlib
import numpy as np
from abc import ABC, abstractmethod

# --- The Interface ---
class InterpolationStrategy(ABC):
    @abstractmethod
    def compute(self, ssfc, psfc, spres, pthta):
        pass

# --- Strategy 1: Fortran 2023 ---
class FortranStrategy(InterpolationStrategy):
    def __init__(self, module_name):
        self.lib = importlib.import_module(module_name)

    def compute(self, ssfc, psfc, spres, pthta):
        # Zero-copy memory alignment for bitwise parity
        return self.lib.s2thta_vector(
            np.asfortranarray(ssfc),
            np.asfortranarray(psfc),
            np.asfortranarray(spres),
            np.asfortranarray(pthta)
        )

# --- Strategy 2: Vectorized Python ---
class PythonStrategy(InterpolationStrategy):
    def compute(self, ssfc, psfc, spres, pthta):
        # Your verified NumPy logic that gives 0.0 MAE
        return self._run_numpy_logic(ssfc, psfc, spres, pthta)

    def _run_numpy_logic(self, ssfc, psfc, spres, pthta):
        # [Insert your verified Python code here]
        pass

# --- The Factory (Dynamic Choice) ---
class StrategyFactory:
    @staticmethod
    def get_strategy(config_path="config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        backend = config['interpolation']['backend'].lower()
        module = config['interpolation']['module_name']

        if backend == "f23":
            print("Production Mode: Using Fortran 2023 Backend")
            return FortranStrategy(module)
        elif backend == "python":
            print("Validation Mode: Using Vectorized Python Backend")
            return PythonStrategy()
        else:
            raise ValueError(f"Backend '{backend}' not supported.")
