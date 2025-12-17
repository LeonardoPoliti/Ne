#=================================================================================================  
#                              Eyes Data Classes - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Load and parse EyeLink ASC eye-tracking data files
# 
#=================================================================================================

import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, Tuple
from pathlib import Path
try:
    from _utils.base_data_class import BaseData # base class
except ImportError:
    class BaseData:
        """Minimal base class fallback"""
        def __init__(self, filepath: str):
            self.filepath = Path(filepath)
            self._loaded = False


#===================== EYES DATA CLASS ========================

class EyesData(BaseData):
    """
    Load and parse EyeLink ASC data files.
    
    Extracts:
    - Gaze data (continuous eye position: x, y - left and/or right)
    - Pupil data (continuous pupil diameter/area - left and/or right)
    - Input channel data (digital input signals)
    - Fixations (x, y, duration, avg pupil)
    - Saccades (start/end positions, amplitude, peak velocity)
    - Blinks (start, end, duration)
    - Event markers (events)
    - Metadata (sampling rate, eye tracked, pupil mode, tracking mode, filter level)

    Useful Methods:
    - get_task_start_time(): Get timestamp of first trial's first state
    - get_trial_times(trial_num): Get start and end times for a specific trial
    - get_trial_data(trial_num, data_type): Extract data for a specific trial
    - get_gaze_in_time_window(start_time, end_time): Extract gaze data in time window
    - get_pupil_in_time_window(start_time, end_time): Extract pupil data in time window
    - get_eyes_events_in_time_window(start_time, end_time, event_types): Extract events in time window
    - compute_velocity(eye, optional: pixels_per_degree): Compute gaze velocity
    - compute_acceleration(eye, optional: pixels_per_degree): Compute gaze acceleration
    - get_data_attributes(): Get list of data attribute names
    - get_session_duration(): Get total duration of the eye-tracking session
    - to_numpy(data_type): Convert gaze and/or pupil data to numpy array
    - __repr__(): String representation of the EyesData object

    """
    
    def __init__(self, filepath: str, verbose: bool = True):
        """
        Args:
            filepath: Path to EyeLink ASC file
            verbose: Whether to print loading information
        """
        super().__init__(filepath)
        
        # Internal variables
        self._verbose = verbose
        self._is_binocular = False
        self._monocular_eye = 'left'
        self._time_offset = 0     # Offset applied to align times to task start
        self._header_sampling_rate = None  # Sampling rate from header (if available)   
        
        # Timestamps 
        self.time = None          # numpy array: timestamps in ms (n_samples,)
        
        # Continuous data 
        self.gaze = None          # DataFrame: x, y (left and/or right)
        self.pupil = None         # DataFrame: pupil diameter (left and/or right)
        self.input_data = None    # DataFrame: input (digital input channel)
        
        # Sparse eye events 
        self.fixations = None     # DataFrame: eye, start, end, duration, x, y, avg_pupil
        self.saccades = None      # DataFrame: eye, start, end, duration, positions, amplitude, velocity
        self.blinks = None        # DataFrame: eye, start, end, duration
        
        # Sparse trial events
        self.events = None        # DataFrame: time, message, trial_num, state_num 
        
        # Metadata
        self.sampling_rate = None       # Sampling rate in Hz
        self.eye_tracked = None         # 'L', 'R', or 'LR'/'BINOCULAR'
        self.pupil_mode = None          # 'DIAMETER' or 'AREA'
        self.tracking_mode = None       # 'CR' (corneal reflection) or 'P' (pupil only)
        self.filter_level = None        # Filter level (0, 1, or 2)
    
    #-------------------------------------------------------------------------------------

    def load(self, align_to_task_start: bool = False):
        """
        Load and parse EyeLink ASC file.
        
        Args:
            align_to_task_start: If True, subtract the time of the first trial's 
                                 first state from all timestamps (time=0 at task start)
        """
        if self._verbose:
            print(f"Loading eye tracking data from {self.filepath.name}...")
        
        self._parse_asc_file()
        
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
        sr_str = f"{self.sampling_rate:.2f} Hz" if self.sampling_rate is not None else self._header_sampling_rate
        print(f"  ✓ Sampling rate: {sr_str}")
        mode = "binocular" if self._is_binocular else f"monocular ({self._monocular_eye})"
        print(f"  ✓ Recording mode: {mode}")
        print(f"  ✓ Pupil mode: {self.pupil_mode or 'unknown'}")
        print(f"  ✓ Tracking mode: {self.tracking_mode or 'unknown'}")
        print(f"  ✓ Filter level: {self.filter_level}")
        print(f"  ✓ Samples: {len(self.time):,}")
        print(f"  ✓ Duration: {self.get_session_duration():.1f} s")
        print(f"  ✓ Fixations: {len(self.fixations):,}")
        print(f"  ✓ Saccades: {len(self.saccades):,}")
        print(f"  ✓ Blinks: {len(self.blinks):,}")
        print(f"  ✓ Trial events: {len(self.events):,}")
    
    #-------------------------------------------------------------------------------------

    def _parse_asc_file(self):
        """Main parser for ASC file"""
        # Initialize storage lists
        time_data = []
        gaze_data = []
        pupil_data = []
        input_data = []
        fixations_data = []
        saccades_data = []
        blinks_data = []
        events_data = []
        
        # Pre-compile regex for message parsing
        msg_pattern = re.compile(r'MSG\s+(\d+)\s+(.*)')
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                # Skip header lines
                if line.startswith('**'):
                    continue
                
                # Parse configuration lines
                if line.startswith(('START', 'PRESCALER', 'VPRESCALER', 'PUPIL', 
                                    'EVENTS', 'SAMPLES', 'END')):
                    self._parse_config(line)
                    continue
                
                # First character determines line type
                first_char = line[0]
                
                # Parse gaze sample data (starts with digit)
                if first_char.isdigit():
                    result = self._parse_gaze_sample(line)
                    if result:
                        timestamp, gaze_sample, pupil_sample, input_sample = result
                        time_data.append(timestamp)
                        gaze_data.append(gaze_sample)
                        pupil_data.append(pupil_sample)
                        input_data.append(input_sample)
                    continue
                
                # Parse events based on prefix
                if first_char == 'E':
                    if line.startswith('EFIX'):
                        fix_data = self._parse_efix(line)
                        if fix_data:
                            fixations_data.append(fix_data)
                    elif line.startswith('ESACC'):
                        sacc_data = self._parse_esacc(line)
                        if sacc_data:
                            saccades_data.append(sacc_data)
                    elif line.startswith('EBLINK'):
                        blink_data = self._parse_eblink(line)
                        if blink_data:
                            blinks_data.append(blink_data)
                
                # Parse trial events
                elif first_char == 'M' and line.startswith('MSG'):
                    msg_data = self._parse_message(line, msg_pattern)
                    if msg_data:
                        events_data.append(msg_data)
        
        # Convert to arrays and DataFrames
        self._build_dataframes(time_data, gaze_data, pupil_data, input_data, 
                               fixations_data, saccades_data, blinks_data, 
                               events_data)
        
    #-------------------------------------------------------------------------------------
    
    def _parse_config(self, line: str):
        """Parse configuration lines"""
        parts = line.split()
        
        if line.startswith('SAMPLES') and 'RATE' in line:
            try:
                rate_idx = parts.index('RATE') + 1
                self._header_sampling_rate = float(parts[rate_idx])
            except (ValueError, IndexError):
                pass
            
            if 'TRACKING' in line:
                try:
                    track_idx = parts.index('TRACKING') + 1
                    self.tracking_mode = parts[track_idx]
                except (ValueError, IndexError):
                    pass
            
            if 'FILTER' in line:
                try:
                    filter_idx = parts.index('FILTER') + 1
                    self.filter_level = int(parts[filter_idx])
                except (ValueError, IndexError):
                    pass
        
        elif line.startswith('START'):
            if len(parts) >= 3:
                # Check if both LEFT and RIGHT appear in the line (binocular)
                # or 'LR' or 'BINOCULAR'
                line_upper = line.upper()
                self._is_binocular = (('LEFT' in line_upper and 'RIGHT' in line_upper) or 
                                    'BINOCULAR' in line_upper or
                                    'LR' in parts)
                
                # Set eye_tracked based on what's found
                if self._is_binocular:
                    self.eye_tracked = 'LR'
                else:
                    eye_info = parts[2].upper()
                    self.eye_tracked = 'L' if 'LEFT' in eye_info or eye_info == 'L' else 'R'
                    self._monocular_eye = 'left' if self.eye_tracked == 'L' else 'right'
        
        elif line.startswith('PUPIL'):
            self.pupil_mode = parts[1] if len(parts) > 1 else None
    
    #-------------------------------------------------------------------------------------

    def _parse_gaze_sample(self, line: str) -> Optional[Tuple]:
        """
        Parse gaze sample line - handles missing data (.) during blinks
        
        Returns: (timestamp, gaze_tuple, pupil_tuple, input_value) or None
        """
        try:
            parts = line.split()
            timestamp = int(parts[0])
            
            if self._is_binocular:
                if len(parts) < 8:
                    return None
                
                x_left = np.nan if parts[1] == '.' else float(parts[1])
                y_left = np.nan if parts[2] == '.' else float(parts[2])
                pupil_left = np.nan if parts[3] == '.' else float(parts[3])
                x_right = np.nan if parts[4] == '.' else float(parts[4])
                y_right = np.nan if parts[5] == '.' else float(parts[5])
                pupil_right = np.nan if parts[6] == '.' else float(parts[6])
                input_val = float(parts[7]) if len(parts) > 7 and parts[7] != '...' else np.nan
                
                gaze_tuple = (x_left, y_left, x_right, y_right)
                pupil_tuple = (pupil_left, pupil_right)
                
            else:
                if len(parts) < 4:
                    return None
                
                x = np.nan if parts[1] == '.' else float(parts[1])
                y = np.nan if parts[2] == '.' else float(parts[2])
                pupil = np.nan if parts[3] == '.' else float(parts[3])
                
                input_val = np.nan
                if len(parts) > 4 and parts[4] != '...':
                    try:
                        input_val = float(parts[4])
                    except ValueError:
                        pass
                
                gaze_tuple = (x, y)
                pupil_tuple = (pupil,)
            
            return timestamp, gaze_tuple, pupil_tuple, input_val
                
        except (ValueError, IndexError):
            return None
    
    #-------------------------------------------------------------------------------------

    def _parse_efix(self, line: str) -> Optional[Dict]:
        """Parse fixation end: EFIX L 4260563 4260929 367 637.5 488.9 9515"""
        try:
            parts = line.split()
            return {
                'eye': parts[1],
                'start': int(parts[2]),
                'end': int(parts[3]),
                'duration': int(parts[4]),
                'x': float(parts[5]),
                'y': float(parts[6]),
                'avg_pupil': float(parts[7]) if len(parts) > 7 else np.nan
            }
        except (ValueError, IndexError):
            return None
    
    #-------------------------------------------------------------------------------------

    def _parse_esacc(self, line: str) -> Optional[Dict]:
        """Parse saccade end: ESACC L 4260930 4260950 21 638.8 487.0 603.5 484.4 1.51 134"""
        try:
            parts = line.split()
            return {
                'eye': parts[1],
                'start': int(parts[2]),
                'end': int(parts[3]),
                'duration': int(parts[4]),
                'start_x': float(parts[5]),
                'start_y': float(parts[6]),
                'end_x': float(parts[7]),
                'end_y': float(parts[8]),
                'amplitude': float(parts[9]),
                'peak_velocity': float(parts[10]) if len(parts) > 10 else np.nan
            }
        except (ValueError, IndexError):
            return None
    
    #-------------------------------------------------------------------------------------

    def _parse_eblink(self, line: str) -> Optional[Dict]:
        """Parse blink end: EBLINK L 4261507 4261580 74"""
        try:
            parts = line.split()
            return {
                'eye': parts[1],
                'start': int(parts[2]),
                'end': int(parts[3]),
                'duration': int(parts[4])
            }
        except (ValueError, IndexError):
            return None
    
    #-------------------------------------------------------------------------------------

    def _parse_message(self, line: str, pattern: re.Pattern) -> Optional[Dict]:
        """Parse message line: MSG 4261126 1000"""
        try:
            match = pattern.match(line)
            if match:
                return {
                    'time': int(match.group(1)),
                    'message': match.group(2).strip()
                }
        except (ValueError, AttributeError):
            pass
        return None
    
    #-------------------------------------------------------------------------------------

    def _build_dataframes(self, time_data, gaze_data, pupil_data, input_data,
                          fixations_data, saccades_data, blinks_data, 
                          events_data):
        """Convert parsed data to arrays and DataFrames"""
        
        # Time array
        self.time = np.array(time_data, dtype=float) if time_data else np.array([], dtype=float)

        # Sampling rate 
        if len(self.time) > 1:
            duration_sec = (self.time[-1] - self.time[0]) / 1000.0  # Convert ms to seconds
            self.sampling_rate = (len(self.time) - 1) / duration_sec if duration_sec > 0 else None
        else:
            self.sampling_rate = None
        
        # Gaze DataFrame 
        if self._is_binocular:
            cols = ['x_l', 'y_l', 'x_r', 'y_r']
        else:
            suffix = self._monocular_eye[0]
            cols = [f'x_{suffix}', f'y_{suffix}']
        
        if gaze_data:
            self.gaze = pd.DataFrame(gaze_data, columns=cols)
        else:
            self.gaze = pd.DataFrame(columns=cols)
        
        # Pupil DataFrame 
        if self._is_binocular:
            pupil_cols = ['pupil_l', 'pupil_r']
        else:
            suffix = self._monocular_eye[0]
            pupil_cols = [f'pupil_{suffix}']
        
        if pupil_data:
            self.pupil = pd.DataFrame(pupil_data, columns=pupil_cols)
        else:
            self.pupil = pd.DataFrame(columns=pupil_cols)
        
        # Input DataFrame 
        if input_data:
            self.input_data = pd.DataFrame({'input': input_data})
        else:
            self.input_data = pd.DataFrame(columns=['input'])
        
        # Sparse eye events
        self.fixations = pd.DataFrame(fixations_data) if fixations_data else \
            pd.DataFrame(columns=['eye', 'start', 'end', 'duration', 'x', 'y', 'avg_pupil'])
        self.saccades = pd.DataFrame(saccades_data) if saccades_data else \
            pd.DataFrame(columns=['eye', 'start', 'end', 'duration', 'start_x', 'start_y', 'end_x', 'end_y', 'amplitude', 'peak_velocity'])
        self.blinks = pd.DataFrame(blinks_data) if blinks_data else \
            pd.DataFrame(columns=['eye', 'start', 'end', 'duration'])
        
        # Trial events - filter config messages
        if events_data:
            self.events = pd.DataFrame(events_data)
            config_patterns = 'RECCFG|ELCLCFG|GAZE_COORDS|THRESHOLDS|!MODE|ELCL_|CAMERA_LENS|FILE:|SUBJECT_ID'
            self.events = self.events[~self.events['message'].str.contains(config_patterns, na=False, regex=True)]
            self.events = self.events.reset_index(drop=True)
        else:
            self.events = pd.DataFrame(columns=['time', 'message'])

    #-------------------------------------------------------------------------------------
    #                                     METHODS
    #-------------------------------------------------------------------------------------

    def decode_trial_events(self) -> pd.DataFrame:
        """
        Decode trial event messages into trial number and state number.
        
        Event message format: 1000 * trial_num + state_num
        Decoding: trial_num = message // 1000, state_num = message % 1000
        
        Returns:
            self.events DataFrame with added trial_num and state_num columns
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if self.events is None or self.events.empty:
            self.events['trial_num'] = pd.Series(dtype=float)
            self.events['state_num'] = pd.Series(dtype=float)
            return self.events
        
        # Initialize columns with NaN
        self.events['trial_num'] = np.nan
        self.events['state_num'] = np.nan
        
        # Find numeric messages (trial events)
        numeric_mask = self.events['message'].str.match(r'^\d+$', na=False)
        
        if numeric_mask.any():
            codes = self.events.loc[numeric_mask, 'message'].astype(int)
            self.events.loc[numeric_mask, 'trial_num'] = codes // 1000
            self.events.loc[numeric_mask, 'state_num'] = codes % 1000
        
        if self._verbose:
            n_decoded = numeric_mask.sum()
            n_trials = self.events['trial_num'].nunique()
            n_states = self.events['state_num'].nunique()
            print(f"  ✓ Decoded {n_decoded} trial events: {n_trials} trials, {n_states} unique states")
        
        return self.events
    
    #-------------------------------------------------------------------------------------

    def get_task_start_time(self) -> Optional[int]:
        """
        Get the timestamp of the first trial's first state.
        
        Returns:
            Timestamp in milliseconds, or None if no trial events found
        """
        if 'trial_num' not in self.events.columns:
            self.decode_trial_events()
        
        trial_events = self.events[self.events['trial_num'].notna()]
        
        if trial_events.empty:
            return None
        
        min_trial = trial_events['trial_num'].min()
        first_trial_events = trial_events[trial_events['trial_num'] == min_trial]
        min_state = first_trial_events['state_num'].min()
        first_event = first_trial_events[first_trial_events['state_num'] == min_state]
        
        return int(first_event['time'].iloc[0])
    
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
            print(f"  ✓ Aligning times: subtracting {task_start} ms (task start)")
        
        self._time_offset = task_start
        
        # Align shared time array
        self.time = self.time - task_start
        
        # Align sparse eye events (start and end)
        if not self.fixations.empty:
            self.fixations['start'] = self.fixations['start'] - task_start
            self.fixations['end'] = self.fixations['end'] - task_start
        
        if not self.saccades.empty:
            self.saccades['start'] = self.saccades['start'] - task_start
            self.saccades['end'] = self.saccades['end'] - task_start
        
        if not self.blinks.empty:
            self.blinks['start'] = self.blinks['start'] - task_start
            self.blinks['end'] = self.blinks['end'] - task_start
        
        # Align trial events
        if not self.events.empty:
            self.events['time'] = self.events['time'] - task_start
        
        return self
    
    #-------------------------------------------------------------------------------------

    def get_trial_times(self, trial_num: int) -> Optional[Tuple[int, int]]:
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
        
        return (int(trial_events['time'].min()), int(trial_events['time'].max()))
    
    #-------------------------------------------------------------------------------------

    def get_trial_data(self, trial_num: int, data_type: str = 'gaze') -> Optional[pd.DataFrame]:
        """
        Extract data for a specific trial.
        
        Args:
            trial_num: Trial number
            data_type: 'gaze', 'pupil', 'fixations', 'saccades', 'blinks', or 'all'
            
        Returns:
            DataFrame with data for the specified trial
        """
        times = self.get_trial_times(trial_num)
        if times is None:
            return None
        
        start_time, end_time = times
        
        if data_type in ['fixations', 'saccades', 'blinks']:
            return self.get_eyes_events_in_time_window(start_time, end_time, data_type)
        else:
            return self.get_data_in_time_window(start_time, end_time, data_type)
    
    #-------------------------------------------------------------------------------------

    def get_data_in_time_window(self, start_time: int, end_time: int, 
                                 data_type: str = 'gaze') -> pd.DataFrame:
        """
        Extract continuous data within a specific time window.
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            data_type: 'gaze', 'pupil', 'input', or 'all'
            
        Returns:
            DataFrame with time + requested data
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        mask = (self.time >= start_time) & (self.time <= end_time)
        
        if data_type == 'gaze':
            df = self.gaze.loc[mask].copy()
        elif data_type == 'pupil':
            df = self.pupil.loc[mask].copy()
        elif data_type == 'input':
            df = self.input_data.loc[mask].copy()
        elif data_type == 'all':
            df = pd.concat([self.gaze, self.pupil, self.input_data], axis=1).loc[mask].copy()
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'gaze', 'pupil', 'input', or 'all'")
        
        # Insert time as first column
        df.insert(0, 'time', self.time[mask])
        return df.reset_index(drop=True)
    
    #-------------------------------------------------------------------------------------

    def get_eyes_events_in_time_window(self, start_time: int, end_time: int, 
                                       event_type: str = 'all'):
        """
        Extract eye events (fixations, saccades, or blinks) within a time window.
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            event_type: 'fixations', 'saccades', 'blinks', or 'all'
            
        Returns:
            DataFrame or dict of DataFrames
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if event_type == 'fixations':
            mask = (self.fixations['start'] >= start_time) & (self.fixations['end'] <= end_time)
            return self.fixations[mask].copy()
        elif event_type == 'saccades':
            mask = (self.saccades['start'] >= start_time) & (self.saccades['end'] <= end_time)
            return self.saccades[mask].copy()
        elif event_type == 'blinks':
            mask = (self.blinks['start'] >= start_time) & (self.blinks['end'] <= end_time)
            return self.blinks[mask].copy()
        elif event_type == 'all':
            return {
                'fixations': self.get_eyes_events_in_time_window(start_time, end_time, 'fixations'),
                'saccades': self.get_eyes_events_in_time_window(start_time, end_time, 'saccades'),
                'blinks': self.get_eyes_events_in_time_window(start_time, end_time, 'blinks')
            }
        else:
            raise ValueError(f"Unknown event_type: {event_type}")
        
    #-------------------------------------------------------------------------------------

    def compute_velocity(self, eye: str = 'auto', pixels_per_degree: Optional[float] = None) -> pd.DataFrame:
        """
        Compute gaze velocity (first derivative).
        
        Args:
            eye: Which eye to use - 'left', 'right', 'average', or 'auto' (auto selects 
                based on recording mode: monocular eye or average for binocular)
            pixels_per_degree: Optional. If provided, velocity is returned in degrees/second.
                            Otherwise, returns pixels/second.
            
        Returns:
            DataFrame with columns: time, vx, vy, speed
            Units: degrees/sec if pixels_per_degree provided, else pixels/sec
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Determine which eye(s) to use
        if eye == 'auto':
            eye = 'average' if self._is_binocular else self._monocular_eye
        
        # Get gaze coordinates
        if eye == 'average' and self._is_binocular:
            x = (self.gaze['x_l'].values + self.gaze['x_r'].values) / 2
            y = (self.gaze['y_l'].values + self.gaze['y_r'].values) / 2
        elif eye == 'left':
            x = self.gaze['x_l'].values
            y = self.gaze['y_l'].values
        elif eye == 'right':
            x = self.gaze['x_r'].values
            y = self.gaze['y_r'].values
        else:
            raise ValueError(f"Unknown eye: {eye}. Use 'left', 'right', 'average', or 'auto'")
        
        # Compute dt in seconds
        dt = np.diff(self.time) / 1000.0
        
        # Handle zero dt (duplicate timestamps)
        dt[dt == 0] = np.nan
        
        # Compute velocity components (pixels/sec)
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Convert to degrees/sec if calibration provided
        if pixels_per_degree is not None:
            vx = vx / pixels_per_degree
            vy = vy / pixels_per_degree
        
        # Compute speed (magnitude)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Use midpoint times
        mid_times = (self.time[:-1] + self.time[1:]) / 2
        
        return pd.DataFrame({
            'time': mid_times,
            'vx': vx,
            'vy': vy,
            'speed': speed
        })

    #-------------------------------------------------------------------------------------

    def compute_acceleration(self, eye: str = 'auto', pixels_per_degree: Optional[float] = None) -> pd.DataFrame:
        """
        Compute gaze acceleration (second derivative).
        
        Args:
            eye: Which eye to use - 'left', 'right', 'average', or 'auto'
            pixels_per_degree: Optional. If provided,,<|fim_middle|>1, 100=70
                            Otherwise, returns pixels/sec².
            
        Returns:
            DataFrame with columns: time, ax, ay, magnitude
            Units: degrees/sec² if pixels_per_degree provided, else pixels/sec²
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Get velocity first
        vel_df = self.compute_velocity(eye=eye, pixels_per_degree=pixels_per_degree)
        
        # Compute dt in seconds
        dt = np.diff(vel_df['time'].values) / 1000.0
        
        # Handle zero dt
        dt[dt == 0] = np.nan
        
        # Compute acceleration components
        ax = np.diff(vel_df['vx'].values) / dt
        ay = np.diff(vel_df['vy'].values) / dt
        
        # Compute magnitude
        magnitude = np.sqrt(ax**2 + ay**2)
        
        # Use midpoint times
        mid_times = (vel_df['time'].values[:-1] + vel_df['time'].values[1:]) / 2
        
        return pd.DataFrame({
            'time': mid_times,
            'ax': ax,
            'ay': ay,
            'magnitude': magnitude
        })

    #-------------------------------------------------------------------------------------


    def get_data_attributes(self) -> list:
        """Get list of data attribute names"""
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        return ['time', 'gaze', 'pupil', 'input_data', 'fixations', 'saccades', 'blinks', 'events']
    
    #-------------------------------------------------------------------------------------

    def get_session_duration(self) -> Optional[float]:
        """Calculate total duration of recording in seconds"""
        if self.time is None or len(self.time) == 0:
            return None
        return (self.time[-1] - self.time[0]) / 1000.0
    
    #-------------------------------------------------------------------------------------

    def to_numpy(self, data_type: str = 'gaze') -> np.ndarray:
        """
        Convert data to numpy array (without time).
        
        Args:
            data_type: 'gaze', 'pupil', or 'all' (concatenated array)
            
        Returns:
            numpy array
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")
        
        if data_type == 'gaze':
            return self.gaze.values
        elif data_type == 'pupil':
            return self.pupil.values
        elif data_type == 'all':
            return pd.concat([self.gaze, self.pupil], axis=1).values
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    #-------------------------------------------------------------------------------------

    def __repr__(self):
        if not self._loaded:
            return f"EyesData(file={self.filepath.name}, not loaded)"
        
        duration = self.get_session_duration()
        duration_str = f"{duration:.1f}s" if duration else "N/A"
        mode = "binocular" if self._is_binocular else "monocular"
        
        return (f"EyesData(file={self.filepath.name}, mode={mode}, "
                f"duration={duration_str}, samples={len(self.time):,}, "
                f"fixations={len(self.fixations)}, saccades={len(self.saccades)}, "
                f"blinks={len(self.blinks)})")
    
    #-------------------------------------------------------------------------------------
    
    def __len__(self):
        """Return number of samples"""
        return len(self.time) if self.time is not None else 0

#=================================================================================================
