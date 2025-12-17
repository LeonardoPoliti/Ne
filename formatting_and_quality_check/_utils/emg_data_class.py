#=================================================================================================  
#                              EMG Data Class - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Load and parse EMG analog data (Digitimer D440-4) from Vicon C3D files
#
#=================================================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import ezc3d

try:
    from _utils.base_data_class import BaseData
except ImportError:
    class BaseData:
        """Minimal base class fallback"""
        def __init__(self, filepath: str):
            self.filepath = Path(filepath)
            self._loaded = False


#===================== EMG DATA CLASS ========================

class EMGData(BaseData):
    """
    Load and parse EMG analog data from Vicon C3D files.
    
    Extracts:
    - time: timestamps array in milliseconds (n_samples,)
    - channels: DataFrame with EMG channels 
    - events: DataFrame with time, trial_num, event_type ('start' or 'end')
    - Metadata: sampling_rate, n_channels, channel_names

    Useful Methods:
    - get_trial_times(trial_num): Get start and end times for a specific trial
    - get_trial_data(trial_num): Extract EMG data for a specific trial
    - get_data_in_time_window(start_time, end_time): Extract EMG data in time window
    - get_data_attributes(): Get list of data attribute names
    - get_session_duration(): Get total duration of the EMG recording session
    - to_numpy(): Convert EMG data to numpy array (without time)
    - __repr__(): String representation of the EMGData object
    - __len__(): Number of samples in the EMG data
    
    """
    
    def __init__(self, filepath: str, verbose: bool = True):
        """
        Args:
            filepath: Path to C3D file
            verbose: Whether to print loading information
        """
        super().__init__(filepath)
        
        # Internal variables
        self._verbose = verbose
        self._time_offset = 0              # Offset applied to align times to task start
        self._c3d = None                   # Raw C3D object
        self._trigger_channel_name = None  # Name of trigger channel
        self._trigger_threshold = None     # Threshold for edge detection
        
        # Timestamps 
        self.time = None          # numpy array: timestamps in ms (n_samples,)
        
        # Data 
        self.channels = None           # DataFrame: EMG channels only
        
        # Sparse events
        self.events = None        # DataFrame: time, trial_num, event_type
        
        # Metadata
        self.sampling_rate = None       # Sampling rate in Hz
        self.n_channels = None          # Number of EMG channels
        self.channel_names = None       # List of EMG channel names
        self.n_trials = None            # Number of trials detected
    
    #-------------------------------------------------------------------------------------

    def load(self, 
             emg_channels: Optional[List[str]] = None,
             trigger_channel: str = 'marker.trial_on',
             trigger_threshold: Optional[float] = None,
             align_to_task_start: bool = False):
        """
        Load and parse EMG data from C3D file.
        
        Args:
            emg_channels: List of EMG channel names to load. If None, auto-detect
                          (loads all analog channels except the trigger channel)
            trigger_channel: Name of the analog channel containing the square wave
                             trigger signal (default: 'marker.trial_on')
            trigger_threshold: Threshold for detecting rising/falling edges.
                               If None, uses midpoint between min and max.
            align_to_task_start: If True, subtract the time of the first trial start
                                 from all timestamps (time=0 at task start)
        """
        if self._verbose:
            print(f"Loading EMG data from {self.filepath.name}...")
        
        self._trigger_channel_name = trigger_channel
        self._trigger_threshold = trigger_threshold
        
        self._parse_c3d_file(emg_channels, trigger_channel)
        
        if self._verbose:
            self._print_load_summary()
        
        self._loaded = True
        
        if align_to_task_start:
            self.align_times_to_task_start()
        
        return self
    
    #-------------------------------------------------------------------------------------

    def _print_load_summary(self):
        """Print loading summary"""
        print(f"  ✓ Sampling rate: {self.sampling_rate:.1f} Hz")
        print(f"  ✓ EMG channels: {self.n_channels} ({', '.join(self.channel_names)})")
        print(f"  ✓ Samples: {len(self.time):,}")
        print(f"  ✓ Duration: {self.get_session_duration():.1f} s")
        print(f"  ✓ Trials detected: {self.n_trials}")
    
    #-------------------------------------------------------------------------------------

    def _parse_c3d_file(self, emg_channels: Optional[List[str]], trigger_channel: str):
        """Parse the C3D file for analog (EMG) data"""
        # Load C3D file
        self._c3d = ezc3d.c3d(str(self.filepath))
        
        # Get analog metadata
        analog_labels = self._c3d['parameters']['ANALOG']['LABELS']['value']
        self.sampling_rate = float(self._c3d['header']['analogs']['frame_rate'])
        
        # Get analog data: shape (1, n_channels, n_samples)
        analog_data = self._c3d['data']['analogs'][0]  # Shape: (n_channels, n_samples)
        n_samples = analog_data.shape[1]
        
        # Create time array in milliseconds
        self.time = np.arange(n_samples) * (1000.0 / self.sampling_rate)
        
        # Determine EMG channels (exclude trigger channel)
        if emg_channels is None:
            # Auto-detect: all channels except trigger
            emg_channels = [ch for ch in analog_labels if ch != trigger_channel]
        
        # Validate channels exist
        for ch in emg_channels:
            if ch not in analog_labels:
                raise ValueError(f"EMG channel '{ch}' not found in C3D file. "
                                 f"Available channels: {analog_labels}")
        
        if trigger_channel not in analog_labels:
            raise ValueError(f"Trigger channel '{trigger_channel}' not found in C3D file. "
                             f"Available channels: {analog_labels}")
        
        # Store metadata
        self.n_channels = len(emg_channels)
        self.channel_names = emg_channels
        
        # Build EMG DataFrame
        emg_data = {}
        for ch in emg_channels:
            ch_idx = analog_labels.index(ch)
            # Clean channel name for column
            clean_name = ch.lower().replace(' ', '_').replace('.', '_')
            emg_data[clean_name] = analog_data[ch_idx]
        self.channels = pd.DataFrame(emg_data)
        
        # Parse trigger channel for trial events
        trigger_idx = analog_labels.index(trigger_channel)
        trigger_signal = analog_data[trigger_idx]
        self._parse_trigger_events(trigger_signal)
    
    #-------------------------------------------------------------------------------------

    def _parse_trigger_events(self, trigger_signal: np.ndarray):
        """
        Parse square wave trigger signal to detect trial start/end events.
        
        Rising edges = trial starts
        Falling edges = trial ends
        """
        # Determine threshold
        if self._trigger_threshold is None:
            self._trigger_threshold = (trigger_signal.max() + trigger_signal.min()) / 2
        
        # Binarize signal
        binary_trigger = (trigger_signal > self._trigger_threshold).astype(int)
        
        # Detect edges
        diff_signal = np.diff(binary_trigger)
        rising_edges = np.where(diff_signal == 1)[0] + 1   # +1 to get actual transition sample
        falling_edges = np.where(diff_signal == -1)[0] + 1
        
        # Build events DataFrame
        events_list = []
        trial_num = 1
        
        # Match rising and falling edges to form trials
        for start_idx in rising_edges:
            start_time = self.time[start_idx]
            
            # Find corresponding falling edge (first one after this rising edge)
            end_candidates = falling_edges[falling_edges > start_idx]
            
            events_list.append({
                'time': start_time,
                'trial_num': trial_num,
                'event_type': 'start'
            })
            
            if len(end_candidates) > 0:
                end_idx = end_candidates[0]
                end_time = self.time[end_idx]
                events_list.append({
                    'time': end_time,
                    'trial_num': trial_num,
                    'event_type': 'end'
                })
            
            trial_num += 1
        
        self.events = pd.DataFrame(events_list)
        self.n_trials = len(rising_edges)
    
    #-------------------------------------------------------------------------------------
    #                                        METHODS
    #-------------------------------------------------------------------------------------

    def get_task_start_time(self) -> Optional[float]:
        """
        Get the timestamp of the first trial start.
        
        Returns:
            Timestamp in milliseconds, or None if no trial events found
        """
        if self.events is None or self.events.empty:
            return None
        
        start_events = self.events[self.events['event_type'] == 'start']
        if start_events.empty:
            return None
        
        return float(start_events['time'].iloc[0])
    
    #-------------------------------------------------------------------------------------

    def align_times_to_task_start(self):
        """
        Subtract the time of the first trial start from all timestamps.
        This makes time=0 correspond to the start of the first trial.
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
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
            trial_num: Trial number to get times for (1-indexed)
            
        Returns:
            Tuple of (start_time, end_time) in milliseconds, or None if trial not found
        """
        if self.events is None or self.events.empty:
            return None
        
        trial_events = self.events[self.events['trial_num'] == trial_num]
        
        if trial_events.empty:
            return None
        
        start_event = trial_events[trial_events['event_type'] == 'start']
        end_event = trial_events[trial_events['event_type'] == 'end']
        
        if start_event.empty:
            return None
        
        start_time = float(start_event['time'].iloc[0])
        
        if end_event.empty:
            # No end found, use end of recording
            end_time = float(self.time[-1])
        else:
            end_time = float(end_event['time'].iloc[0])
        
        return (start_time, end_time)
    
    #-------------------------------------------------------------------------------------

    def get_trial_data(self, trial_num: int) -> Optional[pd.DataFrame]:
        """
        Extract EMG data for a specific trial.
        
        Args:
            trial_num: Trial number (1-indexed)
            
        Returns:
            DataFrame with time + EMG data for the specified trial
        """
        times = self.get_trial_times(trial_num)
        if times is None:
            return None
        
        start_time, end_time = times
        return self.get_data_in_time_window(start_time, end_time)
    
    #-------------------------------------------------------------------------------------

    def get_data_in_time_window(self, start_time: float, end_time: float) -> pd.DataFrame:
        """
        Extract EMG data within a specific time window .
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            DataFrame with time + EMG data
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        mask = (self.time >= start_time) & (self.time <= end_time)
        
        df = self.channels.loc[mask].copy()
        
        # Insert time as first column
        df.insert(0, 'time', self.time[mask])
        return df.reset_index(drop=True)
    
    #-------------------------------------------------------------------------------------

    def get_data_attributes(self) -> list:
        """Get list of data attribute names"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return ['time', 'emg', 'events']
    
    #-------------------------------------------------------------------------------------

    def get_session_duration(self) -> Optional[float]:
        """Calculate total duration of recording in seconds"""
        if self.time is None or len(self.time) == 0:
            return None
        return (self.time[-1] - self.time[0]) / 1000.0
    
    #-------------------------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """
        Convert EMG data to numpy array (without time).
        
        Returns:
            numpy array of shape (n_samples, n_channels)
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        return self.channels.values
    
    #-------------------------------------------------------------------------------------

    def get_trigger_info(self) -> Dict:
        """
        Get information about the trigger channel configuration.
        
        Returns:
            Dictionary with trigger channel name and threshold
        """
        return {
            'channel_name': self._trigger_channel_name,
            'threshold': self._trigger_threshold
        }

    #-------------------------------------------------------------------------------------

    def __repr__(self):
        if not self._loaded:
            return f"EMGData(file={self.filepath.name}, not loaded)"
        
        duration = self.get_session_duration()
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        
        return (f"EMGData(file={self.filepath.name}, "
                f"channels={self.n_channels}, "
                f"duration={duration_str}, "
                f"samples={len(self.time):,}, "
                f"rate={self.sampling_rate:.1f}Hz, "
                f"trials={self.n_trials})")
    
    #-------------------------------------------------------------------------------------
    
    def __len__(self):
        """Return number of samples"""
        return len(self.time) if self.time is not None else 0


#=================================================================================================