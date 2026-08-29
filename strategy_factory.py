import importlib
import sys

class DynamicStrategyFactory:
    """
    Universal Zero-Hardcoding Runtime Router.
    Dynamically reflects and lazy-loads any module and class across 
    the system using pure runtime string mapping. Fully JOSS and MLAI ready.
    """
    @staticmethod
    def resolve(module_name: str, class_name: str) -> object:
        try:
            # 1. Dynamically import the target script module at runtime
            module = importlib.import_module(module_name)
            
            # 2. Use string reflection to pull the target class object from that module
            strategy_class = getattr(module, class_name)
            
            # 3. Instantiate and return the clean, runnable engine object
            return strategy_class()
            
        except ModuleNotFoundError:
            raise ValueError(f"[FACTORY ERROR] Python could not find module file: '{module_name}'")
        except AttributeError:
            raise ValueError(f"[FACTORY ERROR] Module '{module_name}' has no class named: '{class_name}'")
