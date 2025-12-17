#=====================================================================================
#                        Base Data Class  - NeuroRobCoRe
#=====================================================================================

from abc import ABC, abstractmethod
from pathlib import Path

#------------------------------------------------

class BaseData(ABC):
    """Base class for all data types with common functionality"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self._loaded = False
        self._validate_filepath()
    
    def _validate_filepath(self):
        """Validate that file exists"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
    
    @abstractmethod
    def load(self):
        """Load data from file"""
        pass
    
    @abstractmethod
    def get_data_attributes(self) -> list:
        """Return list of data attribute names"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if data has been loaded"""
        return self._loaded
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(file={self.filepath.name}, {status})"