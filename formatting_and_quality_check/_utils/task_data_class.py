#=================================================================================================
#                              Task Data Classes - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Metadata and Trials data classes with automatic attribute creation - flexible for any task
#  - json section becomes attribute in Metadata containing the subsection as dict
#  - csv columns become attributes in TrialsData, with event columns stored in .events DataFrame 
#    and others as numpy arrays
#
#=================================================================================================

import pandas as pd 
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
try:
    from _utils.base_data_class import BaseData # base class 
except ImportError:
    class BaseData:
        """Minimal base class fallback"""
        def __init__(self, filepath: str):
            self.filepath = Path(filepath)
            self._loaded = False

## ======================= METADATA ========================


class Metadata(BaseData):
    """
    Load metadata from JSON - each top-level JSON key becomes a class attribute containing that section's data.

    Useful methods:
    - get_data_attributes(): list of all data attribute names (section names from JSON)
    - get_raw_data(): original nested dictionary structure
    - __repr__(): summary of loaded data
    """
    
    def __init__(self, json_path: str, verbose: bool = True):
        """
        Args:
            json_path: Path to JSON file
            verbose: Whether to print loading info
        """
        super().__init__(json_path)
        
        # Internal variables 
        self._verbose = verbose
        self._raw_data = None           # Original dict
        self._attribute_names = []      # List of generated attributes
        
        # Data attributes are generated dynamically during load()

    # ----------------------------------------------------------------------------
       
    def load(self):
        """Load JSON and create dict attributes for each section"""
        if self._verbose:
            print(f"Loading metadata from {self.filepath.name}...")
        
        # Load JSON with auto-fix for trailing commas
        self._raw_data = self._load_json()
        
        # Create dict attribute for each top-level section
        for section_name, section_data in self._raw_data.items():
            setattr(self, section_name, section_data)
            self._attribute_names.append(section_name)
        
        if self._verbose:
            print(f"  ✓ Loaded {len(self._attribute_names)} sections: {self._attribute_names}")
        
        self._loaded = True
        return self
    
    # ----------------------------------------------------------------------------

    def _load_json(self) -> dict:
        """Load JSON with auto-fix for trailing commas"""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if self._verbose:
                print(f"  warning: JSON decode error, attempting to fix trailing commas...")
            with open(self.filepath, 'r') as f:
                content = f.read()
            # Remove trailing commas before closing brackets/braces
            fixed = re.sub(r',(\s*[}\]])', r'\1', content)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse JSON file: {self.filepath}") from e
            
    # ----------------------------------------------------------------------------

    def get_data_attributes(self) -> List[str]:
        """Get list of data attribute names (section names from JSON)"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return self._attribute_names
    
    # ----------------------------------------------------------------------------

    def get_raw_data(self) -> dict:
        """Get original nested dictionary structure"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return self._raw_data
    
    # ----------------------------------------------------------------------------

    def __repr__(self):
        if not self._loaded:
            return f"Metadata(file={self.filepath.name}, not loaded)"
        return f"Metadata(file={self.filepath.name}, sections={len(self._attribute_names)})"





## ====================== TRIALS DATA ========================

class TrialsData(BaseData):
    """
    Load trial data from CSV and automatically creates attributes from columns.
    
    - Specify event_columns → stored as events attribute (DataFrame)
    - All other columns → become separate attributes (numpy arrays)

    Useful methods:
    - get_data_attributes(): list of all data attribute names
    - get_trial(trial_num): get all data for a specific trial as dict
    - __repr__(): summary of loaded data
    - __len__(): number of trials

    """
    
    def __init__(self, csv_path: str, event_columns: Optional[List[str]] = None, verbose: bool = True):
        """
        Args:
            csv_path: Path to CSV file
            event_columns: List of column names to keep as events DataFrame
                          If None, try to auto-detect columns with timing keywords
            verbose: Whether to print loading info
        """
        super().__init__(csv_path)
        
        # Internal variables
        self._verbose = verbose
        self._event_columns_spec = event_columns
        self._event_column_names = []
        self._data_column_names = []
        
        # Data attributes
        self.events = None  # DataFrame with event timing
        # Other attributes created dynamically during load()
    
    # ----------------------------------------------------------------------------

    def load(self):
        """Load CSV file and create attributes from columns"""
        if self._verbose:
            print(f"Loading trials from {self.filepath.name}...")
        
        # Load CSV
        df = pd.read_csv(self.filepath)
        n_cols = len(df.columns)
        
        if self._verbose:
            print(f"  ✓ Loaded {len(df)} trials with {n_cols} columns")
        
        # Determine which columns are events
        if self._event_columns_spec is None:
            self._event_column_names = self._auto_detect_event_columns(df)
        else:
            self._event_column_names = [col for col in self._event_columns_spec if col in df.columns]
            missing = set(self._event_columns_spec) - set(self._event_column_names)
            if missing and self._verbose:
                print(f"  ⚠ Event columns not found: {missing}")
        
        if self._verbose:
            print(f"  ✓ Identified {len(self._event_column_names)} event columns")
        
        # Create events DataFrame
        self._create_events_dataframe(df)
        
        # Create numpy array attributes for all other columns
        self._data_column_names = [col for col in df.columns 
                                   if col not in self._event_column_names]
        
        for col in self._data_column_names:
            setattr(self, col, df[col].values)
        
        if self._verbose:
            print(f"  ✓ Created {len(self._data_column_names)} data attributes")
        
        self._loaded = True
        return self
    
    # ----------------------------------------------------------------------------

    def _create_events_dataframe(self, df: pd.DataFrame):
        """Create events DataFrame with trial identifier"""
        if self._event_column_names:
            # Try to find trial number column
            trial_col = None
            for col_name in ['trial_num', 'trial', 'trial_number']:
                if col_name in df.columns:
                    trial_col = col_name
                    break
            
            # Build events DataFrame
            if trial_col:
                event_cols = [trial_col] + self._event_column_names
                self.events = df[event_cols].copy()
            else:
                self.events = df[self._event_column_names].copy()
        else:
            self.events = pd.DataFrame()
    
    # ----------------------------------------------------------------------------

    def _auto_detect_event_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect event columns based on keywords in column names"""
        event_keywords = ['state', 'time', 'onset', 'offset', 'start', 'end', 'cue']
        event_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in event_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    event_cols.append(col)
        
        return event_cols
    # ----------------------------------------------------------------------------

    def _get_num_trials(self) -> int:
        """Get total number of trials"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return len(self.events) if not self.events.empty else 0
    
    # ----------------------------------------------------------------------------

    def get_data_attributes(self) -> List[str]:
        """Get list of all data attribute names (includes 'events' + column attributes)"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return ['events'] + self._data_column_names
    
    # ----------------------------------------------------------------------------

    def get_trial(self, trial_num: int) -> Dict[str, Any]:
        """
        Get all data for a specific trial.
        Args:
            trial_num: Trial number to retrieve 
        Returns:
            Dictionary with all column values for that trial
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Find trial index
        if hasattr(self, 'trial_num'):
            idx = np.where(self.trial_num == trial_num)[0]
            if len(idx) == 0:
                raise ValueError(f"Trial {trial_num} not found")
            idx = idx[0]
        else:
            # Assume 1-indexed trial numbers
            idx = trial_num - 1
            if idx < 0 or idx >= len(self.events):
                raise ValueError(f"Trial {trial_num} out of range")
        
        # Collect all data for this trial
        trial_data = {}
        for col in self._data_column_names:
            trial_data[col] = getattr(self, col)[idx]
        
        # Add events if available
        if not self.events.empty and hasattr(self, 'trial_num'):
            trial_events = self.events[self.events['trial_num'] == trial_num]
            trial_data['events'] = trial_events
        
        return trial_data
    
    # ----------------------------------------------------------------------------

    def __repr__(self):
        if not self._loaded:
            return f"TrialsData(file={self.filepath.name}, not loaded)"
        
        n_trials = self._get_num_trials()
        n_data_attrs = len(self._data_column_names)
        
        return f"TrialsData(file={self.filepath.name}, trials={n_trials}, data_attrs={n_data_attrs})"
    
    # ----------------------------------------------------------------------------

    def __len__(self):
        """Returns number of trials"""
        return self._get_num_trials() if self._loaded else 0

#=================================================================================================