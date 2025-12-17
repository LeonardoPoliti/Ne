#=================================================================================================  
#                          Kinematics Data Class - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Load and parse Vicon marker data from C3D files
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


#===================== KINEMATICS DATA CLASS ========================

class KinematicsData(BaseData):
    """
    Load and parse Vicon marker data from C3D files.
    
    Extracts:
    - time: timestamps array in milliseconds (n_frames,)
    - markers: DataFrame with marker positions (x, y, z per marker)
    - residuals: DataFrame with marker residuals/confidence
    - events: DataFrame with time, trial_num, event_type ('start' or 'end')
    - Metadata: sampling_rate, n_markers, marker_names

    Useful Methods:
    - get_marker(marker_name): Get single marker data as DataFrame
    - get_trial_times(trial_num): Get start and end times for a specific trial
    - get_trial_data(trial_num, markers): Extract marker data for a specific trial
    - get_data_in_time_window(start_time, end_time, markers): Extract marker data in time window
    - get_task_start_time(): Get timestamp of first trial start
    - align_times_to_task_start(): Align time=0 to first trial start
    - compute_velocity(marker_name): Compute velocity for a marker
    - compute_acceleration(marker_name): Compute acceleration for a marker
    - get_data_attributes(): Get list of data attribute names
    - get_session_duration(): Get total duration of the recording session
    - to_numpy(markers): Convert marker data to numpy array
    - __repr__(): String representation of the KinematicsData object
    - __len__(): Number of frames in the marker data
    
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
        self._time_offset = 0       # Offset applied to align times to task start
        self._c3d = None            # Raw C3D object
        self._trigger_channel_name = None
        self._trigger_threshold = None
        self._analog_rate = None    # Analog sampling rate (for trigger)
        
        # Timestamps 
        self.time = None            # numpy array: timestamps in ms (n_frames,)
        
        # Data 
        self.markers = None         # DataFrame: x, y, z columns per marker
        self.residuals = None       # DataFrame: residual/confidence per marker
        
        # Sparse events (from trigger channel)
        self.events = None          # DataFrame: time, trial_num, event_type
        
        # Metadata
        self.sampling_rate = None   # Sampling rate in Hz
        self.n_markers = None       # Number of markers
        self.marker_names = None    # List of marker names
        self.n_trials = None        # Number of trials detected
    
    #-------------------------------------------------------------------------------------

    def load(self, 
             markers: Optional[List[str]] = None,
             trigger_channel: str = 'marker.trial_on',
             trigger_threshold: Optional[float] = None,
             align_to_task_start: bool = False):
        """
        Load and parse marker data from C3D file.
        
        Args:
            markers: List of marker names to load. If None, loads all markers.
            trigger_channel: Name of the analog channel containing the square wave
                             trigger signal (default: 'marker.trial_on')
            trigger_threshold: Threshold for detecting rising/falling edges.
                               If None, uses midpoint between min and max.
            align_to_task_start: If True, subtract the time of the first trial start
                                 from all timestamps (time=0 at task start)
        """
        if self._verbose:
            print(f"Loading kinematics data from {self.filepath.name}...")
        
        self._trigger_channel_name = trigger_channel
        self._trigger_threshold = trigger_threshold
        
        self._parse_c3d_file(markers, trigger_channel)
        
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
        print(f"  ✓ Markers: {self.n_markers} ({', '.join(self.marker_names[:5])}{'...' if self.n_markers > 5 else ''})")
        print(f"  ✓ Frames: {len(self.time):,}")
        print(f"  ✓ Duration: {self.get_session_duration():.1f} s")
        print(f"  ✓ Trials detected: {self.n_trials}")
    
    #-------------------------------------------------------------------------------------

    def _parse_c3d_file(self, selected_markers: Optional[List[str]], trigger_channel: str):
        """Parse the C3D file for marker (point) data"""
        # Load C3D file
        self._c3d = ezc3d.c3d(str(self.filepath))
        
        # Get marker metadata
        all_marker_names = self._c3d['parameters']['POINT']['LABELS']['value']
        self.sampling_rate = float(self._c3d['header']['points']['frame_rate'])
        
        # Get point data: shape (3, n_markers, n_frames) where 3 = [x, y, z]
        points_data = self._c3d['data']['points']
        n_frames = points_data.shape[2]
        meta_residuals = self._c3d['data']['meta_points']['residuals'][0]
        
        # Create time array in milliseconds
        self.time = np.arange(n_frames) * (1000.0 / self.sampling_rate)
        
        # Determine which markers to load
        if selected_markers is None:
            selected_markers = all_marker_names
        else:
            # Validate markers exist
            for m in selected_markers:
                if m not in all_marker_names:
                    raise ValueError(f"Marker '{m}' not found in C3D file. "
                                     f"Available markers: {all_marker_names}")
        
        # Store metadata
        self.n_markers = len(selected_markers)
        self.marker_names = selected_markers
        
        # Build markers DataFrame
        markers_data = {}
        residuals_data = {}
        
        for marker_name in selected_markers:
            marker_idx = all_marker_names.index(marker_name)
            
            # Clean marker name for column names
            clean_name = marker_name.lower().replace(' ', '_')
            
            # Extract x, y, z coordinates
            x = points_data[0, marker_idx, :]
            y = points_data[1, marker_idx, :]
            z = points_data[2, marker_idx, :]
            residual = meta_residuals[marker_idx, :]
            
            # Store in dicts
            markers_data[f'{clean_name}_x'] = x
            markers_data[f'{clean_name}_y'] = y
            markers_data[f'{clean_name}_z'] = z
            residuals_data[clean_name] = residual
        
        self.markers = pd.DataFrame(markers_data)
        self.residuals = pd.DataFrame(residuals_data)
        
        # Parse trigger channel for trial events (square wave)
        self._parse_trigger_events(trigger_channel)
    
    #-------------------------------------------------------------------------------------

    def _parse_trigger_events(self, trigger_channel: str):
        """
        Parse square wave trigger signal from analog data to detect trial events.
        - High during trials
        - Low inter-trials.
        
        The trigger is in analog data which may have higher sampling rate than markers.
        Events are converted to marker time base.
        """
        # Get analog data
        analog_labels = self._c3d['parameters']['ANALOG']['LABELS']['value']
        
        if trigger_channel not in analog_labels:
            if self._verbose:
                print(f"  WARNING: trigger channel '{trigger_channel}' not found. No events parsed.")
                print(f"    Available analog channels: {analog_labels}")
            self.events = pd.DataFrame(columns=['time', 'trial_num', 'event_type'])
            self.n_trials = 0
            return
        
        self._analog_rate = float(self._c3d['header']['analogs']['frame_rate'])
        analog_data = self._c3d['data']['analogs'][0]  # Shape: (n_channels, n_samples)
        
        trigger_idx = analog_labels.index(trigger_channel)
        trigger_signal = analog_data[trigger_idx]
        
        # Determine threshold
        if self._trigger_threshold is None:
            self._trigger_threshold = (trigger_signal.max() + trigger_signal.min()) / 2
        
        # Binarize signal
        binary_trigger = (trigger_signal > self._trigger_threshold).astype(int)
        
        # Detect edges
        diff_signal = np.diff(binary_trigger)
        rising_edges = np.where(diff_signal == 1)[0] + 1
        falling_edges = np.where(diff_signal == -1)[0] + 1
        
        # Convert analog sample indices to milliseconds
        analog_times = np.arange(len(trigger_signal)) * (1000.0 / self._analog_rate)
        
        # Build events DataFrame
        events_list = []
        trial_num = 1
        
        for start_idx in rising_edges:
            start_time = analog_times[start_idx]
            
            events_list.append({
                'time': start_time,
                'trial_num': trial_num,
                'event_type': 'start'
            })
            
            # Find corresponding falling edge
            end_candidates = falling_edges[falling_edges > start_idx]
            if len(end_candidates) > 0:
                end_idx = end_candidates[0]
                end_time = analog_times[end_idx]
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

    def get_marker(self, marker_name: str) -> pd.DataFrame:
        """
        Get data for a single marker.
        
        Args:
            marker_name: Name of the marker
            
        Returns:
            DataFrame with columns: time, x, y, z
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        clean_name = marker_name.lower().replace(' ', '_')
        
        cols = [f'{clean_name}_x', f'{clean_name}_y', f'{clean_name}_z']
        
        for col in cols:
            if col not in self.markers.columns:
                raise ValueError(f"Marker '{marker_name}' not found. "
                                 f"Available markers: {self.marker_names}")
        
        df = pd.DataFrame({
            'time': self.time,
            'x': self.markers[f'{clean_name}_x'].values,
            'y': self.markers[f'{clean_name}_y'].values,
            'z': self.markers[f'{clean_name}_z'].values
        })
        
        return df
    
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
            end_time = float(self.time[-1])
        else:
            end_time = float(end_event['time'].iloc[0])
        
        return (start_time, end_time)
    
    #-------------------------------------------------------------------------------------

    def get_trial_data(self, trial_num: int, 
                       markers: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Extract marker data for a specific trial.
        
        Args:
            trial_num: Trial number (1-indexed)
            markers: List of marker names to include. If None, includes all markers.
            
        Returns:
            DataFrame with time + marker data for the specified trial
        """
        times = self.get_trial_times(trial_num)
        if times is None:
            return None
        
        start_time, end_time = times
        return self.get_data_in_time_window(start_time, end_time, markers)
    
    #-------------------------------------------------------------------------------------

    def get_data_in_time_window(self, start_time: float, end_time: float,
                                 markers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract marker data within a specific time window
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            markers: List of marker names to include. If None, includes all markers.
            
        Returns:
            DataFrame with time + marker data
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        mask = (self.time >= start_time) & (self.time <= end_time)
        
        if markers is None:
            df = self.markers.loc[mask].copy()
        else:
            # Select only requested markers
            cols = []
            for m in markers:
                clean_name = m.lower().replace(' ', '_')
                cols.extend([f'{clean_name}_x', f'{clean_name}_y', f'{clean_name}_z'])
            
            # Validate columns exist
            for col in cols:
                if col not in self.markers.columns:
                    raise ValueError(f"Column '{col}' not found in markers data.")
            
            df = self.markers.loc[mask, cols].copy()
        
        # Insert time as first column
        df.insert(0, 'time', self.time[mask])
        return df.reset_index(drop=True)
    
    #-------------------------------------------------------------------------------------

    def get_data_attributes(self) -> list:
        """Get list of data attribute names"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return ['time', 'markers', 'residuals', 'events']
    
    #-------------------------------------------------------------------------------------

    def get_session_duration(self) -> Optional[float]:
        """Calculate total duration of recording in seconds"""
        if self.time is None or len(self.time) == 0:
            return None
        return (self.time[-1] - self.time[0]) / 1000.0
    
    #-------------------------------------------------------------------------------------

    def to_numpy(self, markers: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert marker data to numpy array (without time).
        
        Args:
            markers: List of marker names. If None, includes all markers.
            
        Returns:
            numpy array of shape (n_frames, n_markers * 3)
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if markers is None:
            return self.markers.values
        
        # Select specific markers
        cols = []
        for m in markers:
            clean_name = m.lower().replace(' ', '_')
            cols.extend([f'{clean_name}_x', f'{clean_name}_y', f'{clean_name}_z'])
        
        return self.markers[cols].values
    
    #-------------------------------------------------------------------------------------

    def compute_velocity(self, marker_name: str) -> pd.DataFrame:
        """
        Compute velocity (first derivative) for a marker.
        
        Args:
            marker_name: Name of the marker
            
        Returns:
            DataFrame with columns: time, vx, vy, vz, speed
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        marker_df = self.get_marker(marker_name)
        
        # Compute dt in seconds
        dt = np.diff(marker_df['time'].values) / 1000.0
        
        # Compute velocity components
        vx = np.diff(marker_df['x'].values) / dt
        vy = np.diff(marker_df['y'].values) / dt
        vz = np.diff(marker_df['z'].values) / dt
        
        # Compute speed (magnitude)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Use midpoint times
        mid_times = (marker_df['time'].values[:-1] + marker_df['time'].values[1:]) / 2
        
        return pd.DataFrame({
            'time': mid_times,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'speed': speed
        })
    
    #-------------------------------------------------------------------------------------

    def compute_acceleration(self, marker_name: str) -> pd.DataFrame:
        """
        Compute acceleration (second derivative) for a marker.
        
        Args:
            marker_name: Name of the marker
            
        Returns:
            DataFrame with columns: time, ax, ay, az, magnitude
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        vel_df = self.compute_velocity(marker_name)
        
        # Compute dt in seconds
        dt = np.diff(vel_df['time'].values) / 1000.0
        
        # Compute acceleration components
        ax = np.diff(vel_df['vx'].values) / dt
        ay = np.diff(vel_df['vy'].values) / dt
        az = np.diff(vel_df['vz'].values) / dt
        
        # Compute magnitude
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Use midpoint times
        mid_times = (vel_df['time'].values[:-1] + vel_df['time'].values[1:]) / 2
        
        return pd.DataFrame({
            'time': mid_times,
            'ax': ax,
            'ay': ay,
            'az': az,
            'magnitude': mag
        })

    #-------------------------------------------------------------------------------------

    def __repr__(self):
        if not self._loaded:
            return f"KinematicsData(file={self.filepath.name}, not loaded)"
        
        duration = self.get_session_duration()
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        
        return (f"KinematicsData(file={self.filepath.name}, "
                f"markers={self.n_markers}, "
                f"duration={duration_str}, "
                f"frames={len(self.time):,}, "
                f"rate={self.sampling_rate:.1f}Hz, "
                f"trials={self.n_trials})")
    
    #-------------------------------------------------------------------------------------
    
    def __len__(self):
        """Return number of frames"""
        return len(self.time) if self.time is not None else 0

#=================================================================================================