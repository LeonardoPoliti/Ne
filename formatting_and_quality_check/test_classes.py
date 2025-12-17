# !!! TEST SCRIPT FOR ALL DATA CLASSES - by Claude Opus 4.5 !!! 

from _utils.task_data_class import TrialsData, Metadata
from _utils.eyes_data_class import EyesData
from _utils.eeg_data_class import EEGData
from _utils.emg_data_class import EMGData
from _utils.kinematics_data_class import KinematicsData
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - update these to your data location

TRIALS_PATH = r"C:\Users\leonardo_politi\Downloads\data\hand_foot\frli_1112\task\frli_01_2025-12-11_12-01\data.csv"
METADATA_PATH = r"C:\Users\leonardo_politi\Downloads\data\hand_foot\frli_1112\task\frli_01_2025-12-11_12-01\metadata.json"
EYES_PATH = r"C:\Users\leonardo_politi\Downloads\data\hand_foot\frli_1112\eyelink\fl011211.asc"
EEG_PATH = r"C:\Users\leonardo_politi\Downloads\data\hand_foot\frli_1112\unicorn\frli_handfoot_01_11_12_2025_11_59_281.csv"
C3D_PATH = r"C:\Users\leonardo_politi\Downloads\data\hand_foot\frli_1112\vicon\frli_111225_01.c3d"

# Analysis parameters
TRIAL_TO_ANALYZE = 5


# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_data(load_eeg: bool = True, load_vicon: bool = True):
    """Load all data sources"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load trial data
    trials = TrialsData(TRIALS_PATH)
    trials.load()
    
    # Load metadata
    metadata = Metadata(METADATA_PATH)
    metadata.load()
    
    # Load eye tracking data with time alignment to task start
    eyes = EyesData(EYES_PATH)
    eyes.load(align_to_task_start=True)
    
    # Load EEG data with time alignment to task start
    eeg = None
    if load_eeg:
        eeg = EEGData(EEG_PATH)
        eeg.load(align_to_task_start=True)
    
    # Load EMG and Kinematics from C3D
    emg = None
    kin = None
    if load_vicon:
        emg = EMGData(C3D_PATH)
        emg.load(align_to_task_start=True)
        
        kin = KinematicsData(C3D_PATH)
        kin.load(align_to_task_start=True)
    
    return trials, metadata, eyes, eeg, emg, kin


# ============================================================================
# DATA SUMMARIES
# ============================================================================

def print_trials_summary(trials: TrialsData):
    """Print summary of trial data"""
    print("\n" + "=" * 70)
    print("TRIALS DATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {trials.filepath.name}")
    print(f"  Number of trials: {len(trials)}")
    print(f"  Data attributes: {trials.get_data_attributes()}")
    
    print(f"\n  Events DataFrame:")
    print(f"    Shape: {trials.events.shape}")
    print(f"    Columns: {trials.events.columns.tolist()}")
    print(f"    First 3 rows:")
    print(trials.events.head(3).to_string(index=False))


def print_metadata_summary(metadata: Metadata):
    """Print summary of metadata"""
    print("\n" + "=" * 70)
    print("METADATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {metadata.filepath.name}")
    print(f"  Sections: {metadata.get_data_attributes()}")
    
    for section in metadata.get_data_attributes():
        data = getattr(metadata, section)
        print(f"\n  [{section}]")
        if isinstance(data, dict):
            for key, value in data.items():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                print(f"    {key}: {value_str}")
        else:
            print(f"    {data}")


def print_eyes_summary(eyes: EyesData):
    """Print summary of eye tracking data"""
    print("\n" + "=" * 70)
    print("EYE TRACKING DATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {eyes.filepath.name}")
    print(f"  Recording mode: {'binocular' if eyes._is_binocular else f'monocular ({eyes._monocular_eye})'}")
    print(f"  Sampling rate: {eyes.sampling_rate} Hz")
    print(f"  Duration: {eyes.get_session_duration():.1f} s")
    
    print(f"\n  Continuous Data:")
    print(f"    Time array: {len(eyes.time):,} samples")
    print(f"    Gaze: {eyes.gaze.shape}")
    print(f"    Pupil: {eyes.pupil.shape}")
    
    print(f"\n  Eye Events:")
    print(f"    Fixations: {len(eyes.fixations):,}")
    print(f"    Saccades: {len(eyes.saccades):,}")
    print(f"    Blinks: {len(eyes.blinks):,}")
    
    print(f"\n  Time Range:")
    print(f"    Start: {eyes.time[0]:,.0f} ms")
    print(f"    End: {eyes.time[-1]:,.0f} ms")


def print_eeg_summary(eeg: EEGData):
    """Print summary of EEG data"""
    print("\n" + "=" * 70)
    print("EEG DATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {eeg.filepath.name}")
    print(f"  Sampling rate: {eeg.sampling_rate:.1f} Hz")
    print(f"  Duration: {eeg.get_session_duration():.1f} s")
    print(f"  Channels: {eeg.n_channels} ({', '.join(eeg.channel_names)})")
    
    print(f"\n  Continuous Data:")
    print(f"    Time array: {len(eeg.time):,} samples")
    print(f"    Valid samples: {eeg.valid.sum():,} ({100*eeg.valid.mean():.1f}%)")
    print(f"    EEG: {eeg.channels.shape}")
    print(f"    Accelerometer: {eeg.accelerometer.shape}")
    print(f"    Gyroscope: {eeg.gyroscope.shape}")
    
    print(f"\n  Trigger Events: {len(eeg.events):,}")
    if 'trial_num' in eeg.events.columns:
        n_trials = int(eeg.events['trial_num'].max())
        print(f"    Trials: {n_trials}")
    
    print(f"\n  Time Range:")
    print(f"    Start: {eeg.time[0]:,.1f} ms")
    print(f"    End: {eeg.time[-1]:,.1f} ms")


def print_emg_summary(emg: EMGData):
    """Print summary of EMG data"""
    print("\n" + "=" * 70)
    print("EMG DATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {emg.filepath.name}")
    print(f"  Sampling rate: {emg.sampling_rate:.1f} Hz")
    print(f"  Duration: {emg.get_session_duration():.1f} s")
    print(f"  Channels: {emg.n_channels} ({', '.join(emg.channel_names)})")
    
    print(f"\n  Continuous Data:")
    print(f"    Time array: {len(emg.time):,} samples")
    print(f"    EMG: {emg.channels.shape}")
    
    print(f"\n  Trial Events: {emg.n_trials} trials")
    
    print(f"\n  Time Range:")
    print(f"    Start: {emg.time[0]:,.1f} ms")
    print(f"    End: {emg.time[-1]:,.1f} ms")
    
    # Trigger info
    trigger_info = emg.get_trigger_info()
    print(f"\n  Trigger Configuration:")
    print(f"    Channel: {trigger_info['channel_name']}")
    print(f"    Threshold: {trigger_info['threshold']:.4f}")


def print_kinematics_summary(kin: KinematicsData):
    """Print summary of Kinematics data"""
    print("\n" + "=" * 70)
    print("KINEMATICS DATA SUMMARY")
    print("=" * 70)
    
    print(f"  File: {kin.filepath.name}")
    print(f"  Sampling rate: {kin.sampling_rate:.1f} Hz")
    print(f"  Duration: {kin.get_session_duration():.1f} s")
    print(f"  Markers: {kin.n_markers}")
    for m in kin.marker_names:
        print(f"    - {m}")
    
    print(f"\n  Continuous Data:")
    print(f"    Time array: {len(kin.time):,} frames")
    print(f"    Markers: {kin.markers.shape}")
    print(f"    Residuals: {kin.residuals.shape}")
    
    print(f"\n  Trial Events: {kin.n_trials} trials")
    
    print(f"\n  Time Range:")
    print(f"    Start: {kin.time[0]:,.1f} ms")
    print(f"    End: {kin.time[-1]:,.1f} ms")
    
    # Residual stats
    print(f"\n  Residuals (3D reconstruction error in mm):")
    for col in kin.residuals.columns[:5]:
        r = kin.residuals[col]
        print(f"    {col}: mean={r.mean():.1f}, std={r.std():.2f}")
    if kin.n_markers > 5:
        print(f"    ... and {kin.n_markers - 5} more markers")


# ============================================================================
# TEST INDIVIDUAL METHODS
# ============================================================================

def test_eyes_methods(eyes: EyesData):
    """Test various EyesData methods"""
    print("\n" + "=" * 70)
    print("TESTING EYESDATA METHODS")
    print("=" * 70)
    
    # Test get_trial_times
    print("\n  [get_trial_times]")
    for trial in [1, 5, 10]:
        times = eyes.get_trial_times(trial)
        if times:
            print(f"    Trial {trial}: {times[0]} to {times[1]} ms (duration: {times[1]-times[0]} ms)")
    
    # Test get_data_in_time_window
    print("\n  [get_data_in_time_window]")
    gaze_window = eyes.get_data_in_time_window(0, 1000, 'gaze')
    print(f"    Window 0-1000ms gaze: {len(gaze_window)} samples")
    
    # Test get_trial_data
    print("\n  [get_trial_data]")
    for data_type in ['gaze', 'pupil', 'fixations', 'saccades', 'blinks']:
        data = eyes.get_trial_data(1, data_type)
        if data is not None:
            print(f"    Trial 1 {data_type}: {len(data)} rows")
    
    # Test to_numpy
    print("\n  [to_numpy]")
    gaze_np = eyes.to_numpy('gaze')
    print(f"    Gaze as numpy: shape={gaze_np.shape}")
    
    # Test compute_velocity
    print("\n  [compute_velocity]")
    # Test with default (auto) eye selection
    vel_df = eyes.compute_velocity(eye='auto')
    print(f"    Velocity (auto eye): {vel_df.shape}, columns: {list(vel_df.columns)}")
    print(f"    Speed stats (pixels/s): mean={vel_df['speed'].mean():.1f}, max={vel_df['speed'].max():.1f}, std={vel_df['speed'].std():.1f}")
    
    # Test with pixels_per_degree conversion (typical value ~30-40 for standard setup)
    # Adjust this value based on your actual screen/distance configuration
    PIXELS_PER_DEGREE = 35.0  # Example value - update for your setup
    vel_deg = eyes.compute_velocity(eye='auto', pixels_per_degree=PIXELS_PER_DEGREE)
    print(f"    Speed stats (deg/s): mean={vel_deg['speed'].mean():.1f}, max={vel_deg['speed'].max():.1f}, std={vel_deg['speed'].std():.1f}")
    
    # Check for physiologically impossible velocities (>1000 deg/s)
    impossible_vel = (vel_deg['speed'] > 1000).sum()
    print(f"    Samples with speed > 1000 deg/s: {impossible_vel} ({100*impossible_vel/len(vel_deg):.2f}%)")
    
    # Test compute_acceleration
    print("\n  [compute_acceleration]")
    acc_df = eyes.compute_acceleration(eye='auto')
    print(f"    Acceleration (auto eye): {acc_df.shape}, columns: {list(acc_df.columns)}")
    print(f"    Magnitude stats (pixels/s²): mean={acc_df['magnitude'].mean():.1f}, max={acc_df['magnitude'].max():.1f}")
    
    acc_deg = eyes.compute_acceleration(eye='auto', pixels_per_degree=PIXELS_PER_DEGREE)
    print(f"    Magnitude stats (deg/s²): mean={acc_deg['magnitude'].mean():.1f}, max={acc_deg['magnitude'].max():.1f}")
    
    # Test velocity in specific time window (e.g., during a trial)
    print("\n  [velocity during trial 1]")
    trial_times = eyes.get_trial_times(1)
    if trial_times:
        vel_trial = eyes.compute_velocity(eye='auto', pixels_per_degree=PIXELS_PER_DEGREE)
        trial_vel = vel_trial[(vel_trial['time'] >= trial_times[0]) & (vel_trial['time'] <= trial_times[1])]
        print(f"    Trial 1 velocity samples: {len(trial_vel)}")
        print(f"    Trial 1 speed: mean={trial_vel['speed'].mean():.1f}, max={trial_vel['speed'].max():.1f} deg/s")
    
    print("\n  [__len__ and __repr__]")
    print(f"    len(eyes) = {len(eyes)}")
    print(f"    repr: {repr(eyes)}")

def test_eeg_methods(eeg: EEGData):
    """Test various EEGData methods"""
    print("\n" + "=" * 70)
    print("TESTING EEGDATA METHODS")
    print("=" * 70)
    
    # Test get_trial_times
    print("\n  [get_trial_times]")
    for trial in [1, 5, 10]:
        times = eeg.get_trial_times(trial)
        if times:
            print(f"    Trial {trial}: {times[0]:.1f} to {times[1]:.1f} ms (duration: {times[1]-times[0]:.1f} ms)")
    
    # Test get_data_in_time_window
    print("\n  [get_data_in_time_window]")
    eeg_window = eeg.get_data_in_time_window(0, 1000, 'eeg')
    print(f"    Window 0-1000ms EEG: {len(eeg_window)} samples")
    
    all_window = eeg.get_data_in_time_window(0, 1000, 'all')
    print(f"    Window 0-1000ms all: {len(all_window)} samples, columns: {list(all_window.columns)}")
    
    # Test get_trial_data
    print("\n  [get_trial_data]")
    for data_type in ['eeg', 'accelerometer', 'gyroscope', 'all']:
        data = eeg.get_trial_data(1, data_type)
        if data is not None:
            print(f"    Trial 1 {data_type}: {data.shape}")
    
    # Test to_numpy
    print("\n  [to_numpy]")
    eeg_np = eeg.to_numpy('eeg')
    print(f"    EEG as numpy: shape={eeg_np.shape}")
    
    print("\n  [__len__ and __repr__]")
    print(f"    len(eeg) = {len(eeg)}")
    print(f"    repr: {repr(eeg)}")


def test_emg_methods(emg: EMGData):
    """Test various EMGData methods"""
    print("\n" + "=" * 70)
    print("TESTING EMGDATA METHODS")
    print("=" * 70)
    
    # Test get_trial_times
    print("\n  [get_trial_times]")
    for trial in [1, 5, 10]:
        times = emg.get_trial_times(trial)
        if times:
            print(f"    Trial {trial}: {times[0]:.1f} to {times[1]:.1f} ms (duration: {times[1]-times[0]:.1f} ms)")
    
    # Test get_data_in_time_window
    print("\n  [get_data_in_time_window]")
    emg_window = emg.get_data_in_time_window(0, 1000)
    print(f"    Window 0-1000ms EMG: {len(emg_window)} samples, columns: {list(emg_window.columns)}")
    
    # Test get_trial_data
    print("\n  [get_trial_data]")
    trial_data = emg.get_trial_data(1)
    if trial_data is not None:
        print(f"    Trial 1: {trial_data.shape}")
        print(f"    Columns: {list(trial_data.columns)}")
    
    # Test to_numpy
    print("\n  [to_numpy]")
    emg_np = emg.to_numpy()
    print(f"    EMG as numpy: shape={emg_np.shape}")
    
    # Test get_task_start_time
    print("\n  [get_task_start_time]")
    start_time = emg.get_task_start_time()
    print(f"    Task start (after alignment): {start_time:.1f} ms")
    
    # Test get_trigger_info
    print("\n  [get_trigger_info]")
    trigger_info = emg.get_trigger_info()
    print(f"    Channel: {trigger_info['channel_name']}")
    print(f"    Threshold: {trigger_info['threshold']:.4f}")
    
    print("\n  [__len__ and __repr__]")
    print(f"    len(emg) = {len(emg)}")
    print(f"    repr: {repr(emg)}")


def test_kinematics_methods(kin: KinematicsData):
    """Test various KinematicsData methods"""
    print("\n" + "=" * 70)
    print("TESTING KINEMATICSDATA METHODS")
    print("=" * 70)
    
    # Test get_trial_times
    print("\n  [get_trial_times]")
    for trial in [1, 5, 10]:
        times = kin.get_trial_times(trial)
        if times:
            print(f"    Trial {trial}: {times[0]:.1f} to {times[1]:.1f} ms (duration: {times[1]-times[0]:.1f} ms)")
    
    # Test get_marker
    print("\n  [get_marker]")
    marker_name = kin.marker_names[0]
    marker_df = kin.get_marker(marker_name)
    print(f"    {marker_name}: {marker_df.shape}, columns: {list(marker_df.columns)}")
    print(f"    First row: x={marker_df['x'].iloc[0]:.2f}, y={marker_df['y'].iloc[0]:.2f}, z={marker_df['z'].iloc[0]:.2f}")
    
    # Test get_data_in_time_window
    print("\n  [get_data_in_time_window]")
    markers_window = kin.get_data_in_time_window(0, 1000)
    print(f"    Window 0-1000ms all markers: {markers_window.shape}")
    
    # Test with specific markers
    markers_window_subset = kin.get_data_in_time_window(0, 1000, markers=['elbow', 'wrist1'])
    print(f"    Window 0-1000ms [elbow, wrist1]: {markers_window_subset.shape}")
    
    # Test get_trial_data
    print("\n  [get_trial_data]")
    trial_data = kin.get_trial_data(1)
    if trial_data is not None:
        print(f"    Trial 1 all markers: {trial_data.shape}")
    
    trial_data_subset = kin.get_trial_data(1, markers=['index', 'hand1'])
    if trial_data_subset is not None:
        print(f"    Trial 1 [index, hand1]: {trial_data_subset.shape}")
    
    # Test compute_velocity
    print("\n  [compute_velocity]")
    vel_df = kin.compute_velocity('index')
    print(f"    Index velocity: {vel_df.shape}, columns: {list(vel_df.columns)}")
    print(f"    Speed stats: mean={vel_df['speed'].mean():.1f}, max={vel_df['speed'].max():.1f} mm/s")
    
    # Test compute_acceleration
    print("\n  [compute_acceleration]")
    acc_df = kin.compute_acceleration('index')
    print(f"    Index acceleration: {acc_df.shape}, columns: {list(acc_df.columns)}")
    print(f"    Magnitude stats: mean={acc_df['magnitude'].mean():.1f}, max={acc_df['magnitude'].max():.1f} mm/s²")
    
    # Test to_numpy
    print("\n  [to_numpy]")
    markers_np = kin.to_numpy()
    print(f"    All markers as numpy: shape={markers_np.shape}")
    
    markers_np_subset = kin.to_numpy(markers=['elbow', 'wrist1'])
    print(f"    [elbow, wrist1] as numpy: shape={markers_np_subset.shape}")
    
    print("\n  [__len__ and __repr__]")
    print(f"    len(kin) = {len(kin)}")
    print(f"    repr: {repr(kin)}")


# ============================================================================
# SYNCHRONIZATION CHECK
# ============================================================================

def check_synchronization(emg: EMGData, kin: KinematicsData):
    """Check that EMG and Kinematics are synchronized (same trial events)"""
    print("\n" + "=" * 70)
    print("SYNCHRONIZATION CHECK (EMG vs KINEMATICS)")
    print("=" * 70)
    
    print(f"\n  EMG trials: {emg.n_trials}")
    print(f"  Kinematics trials: {kin.n_trials}")
    
    # Compare trial times
    print(f"\n  Trial time comparison:")
    print(f"  {'Trial':>6} {'EMG Start':>12} {'Kin Start':>12} {'Diff':>8} {'EMG Dur':>10} {'Kin Dur':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    
    for trial in range(1, min(6, emg.n_trials + 1)):
        emg_times = emg.get_trial_times(trial)
        kin_times = kin.get_trial_times(trial)
        
        if emg_times and kin_times:
            emg_start, emg_end = emg_times
            kin_start, kin_end = kin_times
            emg_dur = emg_end - emg_start
            kin_dur = kin_end - kin_start
            diff = emg_start - kin_start
            
            print(f"  {trial:>6} {emg_start:>12.1f} {kin_start:>12.1f} {diff:>8.1f} {emg_dur:>10.1f} {kin_dur:>10.1f}")
    
    # Check if they match
    emg_start_1 = emg.get_trial_times(1)[0] if emg.get_trial_times(1) else None
    kin_start_1 = kin.get_trial_times(1)[0] if kin.get_trial_times(1) else None
    
    if emg_start_1 is not None and kin_start_1 is not None:
        if abs(emg_start_1 - kin_start_1) < 1:  # within 1ms
            print(f"\n  ✅ EMG and Kinematics are synchronized (same trigger source)")
        else:
            print(f"\n  ⚠ Warning: EMG and Kinematics have different start times")


# ============================================================================
# TRIAL ANALYSIS
# ============================================================================

def analyze_trial(trial_num: int, trials: TrialsData, eyes: EyesData, 
                  eeg: EEGData = None, emg: EMGData = None, kin: KinematicsData = None):
    """Analyze a specific trial with all synchronized data"""
    print("\n" + "=" * 70)
    print(f"TRIAL {trial_num} ANALYSIS")
    print("=" * 70)
    
    # Configuration
    PIXELS_PER_DEGREE = 35.0  # Adjust for your setup
    
    # --- Task Data ---
    print("\n  [Task Data]")
    trial_info = trials.get_trial(trial_num)
    for key, value in trial_info.items():
        if key != 'events' and not isinstance(value, (pd.DataFrame, np.ndarray)):
            print(f"    {key}: {value}")
    
    # --- Eye Data ---
    print("\n  [Eye Data]")
    eye_times = eyes.get_trial_times(trial_num)
    if eye_times:
        print(f"    Window: {eye_times[0]} to {eye_times[1]} ms")
        trial_gaze = eyes.get_trial_data(trial_num, 'gaze')
        trial_fix = eyes.get_trial_data(trial_num, 'fixations')
        trial_sacc = eyes.get_trial_data(trial_num, 'saccades')
        trial_blinks = eyes.get_trial_data(trial_num, 'blinks')
        print(f"    Gaze samples: {len(trial_gaze) if trial_gaze is not None else 0}")
        print(f"    Fixations: {len(trial_fix) if trial_fix is not None else 0}")
        print(f"    Saccades: {len(trial_sacc) if trial_sacc is not None else 0}")
        print(f"    Blinks: {len(trial_blinks) if trial_blinks is not None else 0}")
        
        # Velocity analysis for trial
        vel_df = eyes.compute_velocity(eye='auto', pixels_per_degree=PIXELS_PER_DEGREE)
        trial_vel = vel_df[(vel_df['time'] >= eye_times[0]) & (vel_df['time'] <= eye_times[1])]
        if len(trial_vel) > 0:
            print(f"    Gaze velocity: mean={trial_vel['speed'].mean():.1f}, max={trial_vel['speed'].max():.1f} deg/s")
            
            # Check for impossible velocities in this trial
            impossible = (trial_vel['speed'] > 1000).sum()
            if impossible > 0:
                print(f"    ⚠ Impossible velocities (>1000 deg/s): {impossible} samples")
        
    # --- EEG Data ---
    if eeg is not None:
        print("\n  [EEG Data]")
        eeg_times = eeg.get_trial_times(trial_num)
        if eeg_times:
            print(f"    Window: {eeg_times[0]:.1f} to {eeg_times[1]:.1f} ms")
            trial_eeg = eeg.get_trial_data(trial_num, 'eeg')
            print(f"    EEG samples: {len(trial_eeg) if trial_eeg is not None else 0}")
    
    # --- EMG Data ---
    if emg is not None:
        print("\n  [EMG Data]")
        emg_times = emg.get_trial_times(trial_num)
        if emg_times:
            print(f"    Window: {emg_times[0]:.1f} to {emg_times[1]:.1f} ms")
            trial_emg = emg.get_trial_data(trial_num)
            print(f"    EMG samples: {len(trial_emg) if trial_emg is not None else 0}")
            
            if trial_emg is not None and len(trial_emg) > 0:
                print(f"    EMG Statistics:")
                for col in trial_emg.columns:
                    if col != 'time':
                        data = trial_emg[col]
                        print(f"      {col}: mean={data.mean():.6f}, std={data.std():.6f}")
    
    # --- Kinematics Data ---
    if kin is not None:
        print("\n  [Kinematics Data]")
        kin_times = kin.get_trial_times(trial_num)
        if kin_times:
            print(f"    Window: {kin_times[0]:.1f} to {kin_times[1]:.1f} ms")
            trial_markers = kin.get_trial_data(trial_num)
            print(f"    Marker frames: {len(trial_markers) if trial_markers is not None else 0}")
            
            # Compute velocity for index finger during trial
            if trial_markers is not None:
                vel_df = kin.compute_velocity('index')
                # Filter to trial window
                trial_vel = vel_df[(vel_df['time'] >= kin_times[0]) & (vel_df['time'] <= kin_times[1])]
                print(f"    Index finger speed: mean={trial_vel['speed'].mean():.1f}, max={trial_vel['speed'].max():.1f} mm/s")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run all tests"""
    
    # Load all data
    trials, metadata, eyes, eeg, emg, kin = load_all_data(load_eeg=True, load_vicon=True)
    
    # Print summaries
    print_trials_summary(trials)
    print_metadata_summary(metadata)
    print_eyes_summary(eyes)
    if eeg is not None:
        print_eeg_summary(eeg)
    if emg is not None:
        print_emg_summary(emg)
    if kin is not None:
        print_kinematics_summary(kin)
    
    # Check synchronization between EMG and Kinematics
    if emg is not None and kin is not None:
        check_synchronization(emg, kin)
    
    # Test methods for each class
    test_eyes_methods(eyes)
    if eeg is not None:
        test_eeg_methods(eeg)
    if emg is not None:
        test_emg_methods(emg)
    if kin is not None:
        test_kinematics_methods(kin)
    
    # Analyze specific trial
    analyze_trial(TRIAL_TO_ANALYZE, trials, eyes, eeg, emg, kin)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return trials, metadata, eyes, eeg, emg, kin


if __name__ == "__main__":
    trials, metadata, eyes, eeg, emg, kin = main()