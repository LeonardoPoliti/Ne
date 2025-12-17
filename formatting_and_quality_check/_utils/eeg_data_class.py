#=================================================================================================  
#                              EEG Data Class - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Load and parse g.tec Unicorn Recorder CSV data files 
#
#=================================================================================================

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
try:
    from _utils.base_data_class import BaseData # base class
except ImportError:
    class BaseData:
        """Minimal base class fallback"""
        def __init__(self, filepath: str):
            self.filepath = Path(filepath)
            self._loaded = False


#===================== EEG DATA CLASS ========================

class EEGData(BaseData):
    """
    Load and parse EEG CSV data files.
    
    Extracts:
    - time: shared timestamps array (n_samples,)
    - valid: boolean array indicating sample validity (n_samples,)
    - channels: DataFrame with EEG channels  
    - accelerometer: DataFrame with acc_x, acc_y, acc_z
    - gyroscope: DataFrame with gyr_x, gyr_y, gyr_z 
    - events: DataFrame with time, trigger, trial_num, state_num (sparse, has own time)
    - Metadata: sampling_rate, n_channels, channel_names

    Useful Methods:
    - get_trial_times(trial_num): Get start and end times for a specific trial
    - get_trial_data(trial_num, data_type): Extract data for a specific trial
    - get_data_in_time_window(start_time, end_time, data_type): Extract data in time window
    - get_data_attributes(): Get list of data attribute names
    - get_session_duration(): Get total duration of the EEG recording session
    - to_numpy(data_type): Convert data to numpy array (without time)
    - __repr__(): String representation of the EEGData object
    - __len__(): Number of samples in the EEG data
    
    """
    
    def __init__(self, filepath: str, verbose: bool = True):
        """
        Args:
            filepath: Path to EEG CSV file
            verbose: Whether to print loading information
        """
        super().__init__(filepath)
        
        # Internal variables
        self._verbose = verbose
        self._time_offset = 0  # Offset applied to align times
        self._raw_df = None    # Original dataframe
        
        # Timestamps and validity (shared by all continuous data)
        self.time = None          # numpy array: timestamps in ms (n_samples,)
        self.valid = None         # numpy bool array: sample validity (n_samples,)
        
        # Data 
        self.channels = None      # DataFrame: EEG channels only
        self.accelerometer = None # DataFrame: acc_x, acc_y, acc_z
        self.gyroscope = None     # DataFrame: gyr_x, gyr_y, gyr_z
        
        # Sparse events
        self.events = None        # DataFrame: time, trigger, trial_num, state_num
        
        # Metadata
        self.sampling_rate = None       # Sampling rate in Hz 
        self.n_channels = None          # Number of EEG channels
        self.channel_names = None       # List of EEG channel names
    
    #-------------------------------------------------------------------------------------

    def load(self, align_to_task_start: bool = False):
        """
        Load and parse EEG CSV file.
        
        Args:
            align_to_task_start: If True, subtract the time of the first trial's 
                                 first state from all timestamps (time=0 at task start)
        """
        if self._verbose:
            print(f"Loading EEG data from {self.filepath.name}...")
        
        self._parse_csv_file()
        
        if self._verbose:
            self._print_load_summary()
        
        self._loaded = True
        
        if align_to_task_start:
            self.decode_trial_events()
            self.align_times_to_task_start()
        
        return self
    
    #-------------------------------------------------------------------------------------

    def _print_load_summary(self):
        """Print loading summary"""
        print(f"  ✓ Sampling rate: {self.sampling_rate:.1f} Hz")
        print(f"  ✓ EEG channels: {self.n_channels} ({', '.join(self.channel_names)})")
        print(f"  ✓ Samples: {len(self.time):,} ({100*self.valid.mean():.1f}% valid)")
        print(f"  ✓ Duration: {self.get_session_duration():.1f} s")
        n_triggers = len(self.events) if self.events is not None else 0
        print(f"  ✓ Trigger events: {n_triggers}")
    
    #-------------------------------------------------------------------------------------

    def _parse_csv_file(self):
        """Parse the EEG CSV file"""
        # Load CSV
        self._raw_df = pd.read_csv(self.filepath)
        
        # Clean column names (remove leading spaces)
        self._raw_df.columns = self._raw_df.columns.str.strip()
        
        # Identify column types
        eeg_cols = [col for col in self._raw_df.columns if col.startswith('EEG')]
        acc_cols = [col for col in self._raw_df.columns if col.startswith('ACC')]
        gyr_cols = [col for col in self._raw_df.columns if col.startswith('GYR')]
        
        # Store metadata
        self.n_channels = len(eeg_cols)
        self.channel_names = eeg_cols
        
        # Create time array from cumulative DT (delta time in ms)
        if 'DT' in self._raw_df.columns:
            self.time = self._raw_df['DT'].cumsum().values
        else:
            # Fallback: use sample index
            self.time = np.arange(len(self._raw_df), dtype=float)
        
        # Parse VALID column (boolean array)
        if 'VALID' in self._raw_df.columns:
            self.valid = self._raw_df['VALID'].values.astype(bool)
        else:
            # All samples valid if no VALID column
            self.valid = np.ones(len(self._raw_df), dtype=bool)
        
        # Compute sampling rate from samples over total time
        total_duration_ms = self.time[-1] - self.time[0]
        if total_duration_ms > 0:
            self.sampling_rate = 1000.0 * (len(self.time) - 1) / total_duration_ms
        else:
            self.sampling_rate = None
                
        # Build EEG DataFrame 
        eeg_data = {}
        for col in eeg_cols:
            new_name = col.lower().replace(' ', '_')
            eeg_data[new_name] = self._raw_df[col].values
        self.channels = pd.DataFrame(eeg_data)
        
        # Build Accelerometer DataFrame
        if acc_cols:
            acc_data = {}
            for col in acc_cols:
                new_name = col.lower().replace(' ', '_')
                acc_data[new_name] = self._raw_df[col].values
            self.accelerometer = pd.DataFrame(acc_data)
        else:
            self.accelerometer = pd.DataFrame()
        
        # Build Gyroscope DataFrame 
        if gyr_cols:
            gyr_data = {}
            for col in gyr_cols:
                new_name = col.lower().replace(' ', '_')
                gyr_data[new_name] = self._raw_df[col].values
            self.gyroscope = pd.DataFrame(gyr_data)
        else:
            self.gyroscope = pd.DataFrame()
        
        # Build Events DataFrame from TRIG column 
        if 'TRIG' in self._raw_df.columns:
            # Only store rows with non-zero triggers
            trigger_mask = self._raw_df['TRIG'] != 0
            self.events = pd.DataFrame({
                'time': self.time[trigger_mask],
                'trigger': self._raw_df.loc[trigger_mask, 'TRIG'].values
            }).reset_index(drop=True)
        else:
            self.events = pd.DataFrame(columns=['time', 'trigger'])
    
    #-------------------------------------------------------------------------------------
    #                                        METHODS
    #-------------------------------------------------------------------------------------

    def decode_trial_events(self) -> pd.DataFrame:
        """
        Decode trigger values into trial number and state number.
        
        Trigger format: 1000 * trial_num + state_num
        Decoding: trial_num = trigger // 1000, state_num = trigger % 1000
        
        Adds 'trial_num' and 'state_num' columns to self.events DataFrame.
        
        Returns:
            self.events DataFrame with added columns
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if self.events is None or self.events.empty:
            self.events['trial_num'] = pd.Series(dtype=float)
            self.events['state_num'] = pd.Series(dtype=float)
            return self.events
        
        # Decode all triggers
        triggers = self.events['trigger'].astype(int)
        self.events['trial_num'] = triggers // 1000
        self.events['state_num'] = triggers % 1000
        
        if self._verbose:
            n_trials = self.events['trial_num'].nunique()
            n_states = self.events['state_num'].nunique()
            print(f"  ✓ Decoded {len(self.events)} trigger events: {n_trials} trials, {n_states} unique states")
        
        return self.events
    
    #-------------------------------------------------------------------------------------

    def get_task_start_time(self) -> Optional[float]:
        """
        Get the timestamp of the first trial's first state.
        
        Returns:
            Timestamp in milliseconds, or None if no trial events found
        """
        if 'trial_num' not in self.events.columns:
            self.decode_trial_events()
        
        if self.events.empty:
            return None
        
        min_trial = self.events['trial_num'].min()
        first_trial_events = self.events[self.events['trial_num'] == min_trial]
        min_state = first_trial_events['state_num'].min()
        first_event = first_trial_events[first_trial_events['state_num'] == min_state]
        
        return float(first_event['time'].iloc[0])
    
    #-------------------------------------------------------------------------------------

    def align_times_to_task_start(self):
        """
        Subtract the time of the first trial's first state from all timestamps.
        This makes time=0 correspond to the start of the task.
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if 'trial_num' not in self.events.columns: 
            self.decode_trial_events()
        
        task_start = self.get_task_start_time()
        
        if task_start is None:
            raise ValueError("Could not determine task start time. No trial events found.")
        
        if self._verbose:
            print(f"  ✓ Aligning times: subtracting {task_start:.0f} ms (task start)")
        
        self._time_offset = task_start
        
        # Align time array
        self.time = self.time - task_start
        
        # Align events time
        if not self.events.empty:
            self.events['time'] = self.events['time'] - task_start
        
        return self
    
    #-------------------------------------------------------------------------------------

    def get_trial_times(self, trial_num: int) -> Optional[Tuple[float, float]]:
        """
        Get the start and end times for a specific trial.
        
        Args:
            trial_num: Trial number to get times for
            
        Returns:
            Tuple of (start_time, end_time) in milliseconds, or None if trial not found
        """
        if 'trial_num' not in self.events.columns:
            self.decode_trial_events()
        
        trial_events = self.events[self.events['trial_num'] == trial_num]
        
        if trial_events.empty:
            return None
        
        return (float(trial_events['time'].min()), float(trial_events['time'].max()))
    
    #-------------------------------------------------------------------------------------

    def get_trial_data(self, trial_num: int, data_type: str = 'eeg') -> Optional[pd.DataFrame]:
        """
        Extract data for a specific trial.
        
        Args:
            trial_num: Trial number
            data_type: 'eeg', 'accelerometer', 'gyroscope', or 'all'
            
        Returns:
            DataFrame with time + data for the specified trial
        """
        times = self.get_trial_times(trial_num)
        if times is None:
            return None
        
        start_time, end_time = times
        return self.get_data_in_time_window(start_time, end_time, data_type)
    
    #-------------------------------------------------------------------------------------

    def get_data_in_time_window(self, start_time: float, end_time: float, 
                                 data_type: str = 'eeg') -> pd.DataFrame:
        """
        Extract data within a specific time window.
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            data_type: 'eeg', 'accelerometer', 'gyroscope', or 'all'
            
        Returns:
            DataFrame with time + requested data
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        mask = (self.time >= start_time) & (self.time <= end_time)
        
        if data_type == 'eeg':
            df = self.channels.loc[mask].copy()
        elif data_type == 'accelerometer':
            df = self.accelerometer.loc[mask].copy()
        elif data_type == 'gyroscope':
            df = self.gyroscope.loc[mask].copy()
        elif data_type == 'all':
            df = pd.concat([self.channels, self.accelerometer, self.gyroscope], axis=1).loc[mask].copy()
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'eeg', 'accelerometer', 'gyroscope', or 'all'")
        
        # Insert time as first column
        df.insert(0, 'time', self.time[mask])
        return df.reset_index(drop=True)
    
    
    #-------------------------------------------------------------------------------------

    def get_data_attributes(self) -> list:
        """Get list of data attribute names"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return ['time', 'valid', 'eeg', 'accelerometer', 'gyroscope', 'events']
    
    #-------------------------------------------------------------------------------------

    def get_session_duration(self) -> Optional[float]:
        """Calculate total duration of recording in seconds"""
        if self.time is None or len(self.time) == 0:
            return None
        return (self.time[-1] - self.time[0]) / 1000.0
    
    #-------------------------------------------------------------------------------------

    def to_numpy(self, data_type: str = 'eeg') -> np.ndarray:
        """
        Convert data to numpy array (without time).
        
        Args:
            data_type: 'eeg', 'accelerometer', 'gyroscope', or 'all'
            
        Returns:
            numpy array of shape (n_samples, n_channels)
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if data_type == 'eeg':
            return self.channels.values
        elif data_type == 'accelerometer':
            return self.accelerometer.values
        elif data_type == 'gyroscope':
            return self.gyroscope.values
        elif data_type == 'all':
            return pd.concat([self.channels, self.accelerometer, self.gyroscope], axis=1).values
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    #-------------------------------------------------------------------------------------


    def __repr__(self):
        if not self._loaded:
            return f"EEGData(file={self.filepath.name}, not loaded)"
        
        duration = self.get_session_duration()
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        
        return (f"EEGData(file={self.filepath.name}, "
                f"channels={self.n_channels}, "
                f"duration={duration_str}, "
                f"samples={len(self.time):,}, "
                f"rate={self.sampling_rate:.1f}Hz)")
    
    #-------------------------------------------------------------------------------------
    
    def __len__(self):
        """Return number of samples"""
        return len(self.time) if self.time is not None else 0

#=================================================================================================