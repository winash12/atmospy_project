from abc import ABC, abstractmethod

class GenericDomainStrategy(ABC):
    """Universal root interface contract for interchangeable mathematical operations."""
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass
