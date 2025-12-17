#=================================================================================================
#                              EEG Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy import signal

#--------------------------------------------------------------------------------------

def check_eeg_quality(eeg_data, expected_fs: float = 250.0) -> Dict[str, Any]:
    """
    Run quality checks on EEG data.
    
    Args:
        eeg_data: EEGData object
        expected_fs: Expected sampling rate in Hz (default 250 for g.tec Unicorn)
    
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # metadata
    results['metadata'] = {
        'sampling_rate': eeg_data.sampling_rate,
        'n_channels': eeg_data.n_channels,
        'channel_names': eeg_data.channel_names,
        'n_samples': len(eeg_data.time),
        'duration_s': eeg_data.get_session_duration()
    }
    
    # trial count from events - use max trial number (not nunique)
    # First ensure trial_num column exists by decoding if needed
    if eeg_data.events is not None and not eeg_data.events.empty:
        if 'trial_num' not in eeg_data.events.columns:
            if hasattr(eeg_data, 'decode_trial_events'):
                eeg_data.decode_trial_events()
        
        if 'trial_num' in eeg_data.events.columns:
            trial_nums = eeg_data.events['trial_num'].dropna()
            if len(trial_nums) > 0:
                results['metadata']['n_trials'] = int(trial_nums.max())
            else:
                results['metadata']['n_trials'] = None
        else:
            results['metadata']['n_trials'] = None
    else:
        results['metadata']['n_trials'] = None
    
    # temporal integrity
    results['temporal'] = _check_temporal_integrity(eeg_data, expected_fs)
    
    # signal validity
    results['signal'] = _check_signal_validity(eeg_data)
    
    # accelerometer and gyroscope
    results['motion'] = _check_motion_sensors(eeg_data)
    
    # *trigger events
    results['triggers'] = _check_trigger_events(eeg_data)
    
    return results

#--------------------------------------------------------------------------------------

def _check_temporal_integrity(eeg_data, expected_fs: float) -> Dict[str, Any]:
    """Check temporal integrity of EEG data."""
    results = {}
    
    # Sampling rate check
    results['expected_fs'] = expected_fs
    results['actual_fs'] = eeg_data.sampling_rate
    results['fs_deviation_pct'] = abs(eeg_data.sampling_rate - expected_fs) / expected_fs * 100
    
    # Delta time analysis
    dt = np.diff(eeg_data.time)  # Time is in ms
    expected_dt = 1000.0 / expected_fs  # Expected dt in ms
    
    results['dt'] = {
        'expected_ms': expected_dt,
        'median': float(np.median(dt)),
        'mean': float(np.mean(dt)),
        'std': float(np.std(dt)),
        'min': float(np.min(dt)),
        'max': float(np.max(dt)),
        'values': dt  # For histogram
    }
    
    # Duplicate timestamps (DT = 0)
    n_duplicates = np.sum(dt == 0)
    results['duplicate_timestamps'] = int(n_duplicates)
    results['duplicate_pct'] = float(n_duplicates / len(dt) * 100)
    
    return results

#--------------------------------------------------------------------------------------

def _check_signal_validity(eeg_data) -> Dict[str, Any]:
    """Check signal validity metrics."""
    results = {}
    
    # Overall invalid samples
    n_invalid = np.sum(~eeg_data.valid)
    results['invalid_samples_total'] = int(n_invalid)
    results['invalid_pct_total'] = float(n_invalid / len(eeg_data.valid) * 100)
    
    # Per-channel analysis (using VALID flag - same for all channels in Unicorn)
    # Note: Unicorn has a single VALID flag for all channels
    results['invalid_per_channel'] = {
        'all_channels': float(n_invalid / len(eeg_data.valid) * 100)
    }
    
    # Invalid segment durations
    results['invalid_segments'] = _analyze_invalid_segments(eeg_data)
    
    # Channel correlation matrix
    results['correlation_matrix'] = _compute_channel_correlation(eeg_data)
    
    # DC offset per channel
    results['dc_offset'] = _compute_dc_offset(eeg_data)
    
    # Power spectral density
    results['psd'] = _compute_psd(eeg_data)
    
    # Clipping/saturation detection
    results['clipping'] = _detect_clipping(eeg_data)
    
    return results

#--------------------------------------------------------------------------------------

def _analyze_invalid_segments(eeg_data) -> Dict[str, Any]:
    """Analyze invalid segment durations."""
    valid = eeg_data.valid
    time = eeg_data.time
    
    # Find invalid segment boundaries
    invalid = ~valid
    diff = np.diff(invalid.astype(int))
    
    # Segment starts (0->1 transition)
    starts = np.where(diff == 1)[0] + 1
    # Segment ends (1->0 transition)
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if invalid[0]:
        starts = np.insert(starts, 0, 0)
    if invalid[-1]:
        ends = np.append(ends, len(invalid))
    
    # Calculate durations
    if len(starts) == 0 or len(ends) == 0:
        return {
            'n_segments': 0,
            'durations_ms': [],
            'stats': None
        }
    
    # Match starts and ends
    n_segments = min(len(starts), len(ends))
    durations = []
    segment_info = []
    
    for i in range(n_segments):
        if starts[i] < ends[i]:
            dur = time[ends[i]-1] - time[starts[i]]
            durations.append(dur)
            segment_info.append({
                'start_idx': int(starts[i]),
                'end_idx': int(ends[i]),
                'start_time': float(time[starts[i]]),
                'end_time': float(time[ends[i]-1]),
                'duration_ms': float(dur)
            })
    
    durations = np.array(durations)
    
    stats = None
    if len(durations) > 0:
        stats = {
            'median': float(np.median(durations)),
            'mean': float(np.mean(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations))
        }
    
    return {
        'n_segments': n_segments,
        'durations_ms': durations.tolist(),
        'segments': segment_info,
        'stats': stats
    }

#--------------------------------------------------------------------------------------

def _compute_channel_correlation(eeg_data) -> Dict[str, Any]:
    """Compute correlation matrix between EEG channels."""
    data = eeg_data.channels.values
    
    # Handle NaN values
    valid_mask = eeg_data.valid
    data_valid = data#[valid_mask]
    
    if len(data_valid) < 2:
        return {'matrix': None, 'channel_names': eeg_data.channel_names}
    
    corr_matrix = np.corrcoef(data_valid.T)
    
    return {
        'matrix': corr_matrix.tolist(),
        'channel_names': list(eeg_data.channels.columns)
    }

#--------------------------------------------------------------------------------------

def _compute_dc_offset(eeg_data) -> Dict[str, Any]:
    """Compute DC offset (mean voltage) per channel."""
    results = {}
    
    for col in eeg_data.channels.columns:
        data = eeg_data.channels[col].values
        valid_data = data[eeg_data.valid]
        
        results[col] = {
            'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else None,
            'min': float(np.min(valid_data)) if len(valid_data) > 0 else None,
            'max': float(np.max(valid_data)) if len(valid_data) > 0 else None
        }
    
    return results

#--------------------------------------------------------------------------------------

def _compute_psd(eeg_data, nperseg: int = 1024) -> Dict[str, Any]:
    """Compute power spectral density for each channel."""
    results = {}
    
    fs = eeg_data.sampling_rate
    if fs is None or fs <= 0:
        return {'error': 'Invalid sampling rate'}
    
    for col in eeg_data.channels.columns:
        data = eeg_data.channels[col].values
        valid_data = data[eeg_data.valid]
        
        if len(valid_data) < nperseg:
            nperseg_use = len(valid_data) // 2
            if nperseg_use < 4:
                results[col] = {'error': 'Insufficient data'}
                continue
        else:
            nperseg_use = nperseg
        
        try:
            freqs, psd = signal.welch(valid_data, fs=fs, nperseg=nperseg_use)
            results[col] = {
                'frequencies': freqs.tolist(),
                'power': psd.tolist()
            }
        except Exception as e:
            results[col] = {'error': str(e)}
    
    return results

#--------------------------------------------------------------------------------------

def _detect_clipping(eeg_data, n_consecutive: int = 3) -> Dict[str, Any]:
    """Detect clipping/saturation (consecutive samples at min/max)."""
    results = {}
    
    for col in eeg_data.channels.columns:
        data = eeg_data.channels[col].values
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Find consecutive min values
        at_min = (data == min_val).astype(int)
        min_clips = _count_consecutive_runs(at_min, n_consecutive)
        
        # Find consecutive max values
        at_max = (data == max_val).astype(int)
        max_clips = _count_consecutive_runs(at_max, n_consecutive)
        
        results[col] = {
            'min_value': float(min_val),
            'max_value': float(max_val),
            'clipping_at_min': min_clips,
            'clipping_at_max': max_clips,
            'total_clipping_events': min_clips + max_clips
        }
    
    return results

#--------------------------------------------------------------------------------------

def _count_consecutive_runs(binary_array: np.ndarray, min_length: int) -> int:
    """Count runs of consecutive 1s with at least min_length."""
    if len(binary_array) == 0:
        return 0
    
    # Find run lengths
    diff = np.diff(np.concatenate([[0], binary_array, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    run_lengths = ends - starts
    return int(np.sum(run_lengths >= min_length))

#--------------------------------------------------------------------------------------

def _check_motion_sensors(eeg_data) -> Dict[str, Any]:
    """Check accelerometer and gyroscope data."""
    results = {}
    
    # Accelerometer
    if eeg_data.accelerometer is not None and not eeg_data.accelerometer.empty:
        acc_stats = {}
        for col in eeg_data.accelerometer.columns:
            data = eeg_data.accelerometer[col].values
            valid_data = data[eeg_data.valid]
            
            acc_stats[col] = {
                'median': float(np.median(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'values': valid_data  # For histogram
            }
        results['accelerometer'] = acc_stats
    else:
        results['accelerometer'] = None
    
    # Gyroscope
    if eeg_data.gyroscope is not None and not eeg_data.gyroscope.empty:
        gyr_stats = {}
        for col in eeg_data.gyroscope.columns:
            data = eeg_data.gyroscope[col].values
            valid_data = data[eeg_data.valid]
            
            gyr_stats[col] = {
                'median': float(np.median(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'values': valid_data  # For histogram
            }
        results['gyroscope'] = gyr_stats
    else:
        results['gyroscope'] = None
    
    return results

#--------------------------------------------------------------------------------------

def _check_trigger_events(eeg_data) -> Dict[str, Any]:
    """Check trigger event counts."""
    events = eeg_data.events
    
    if events is None or events.empty:
        return {'n_events': 0, 'trigger_counts': {}}
    
    # Count events
    n_events = len(events)
    
    # Count by trigger value
    trigger_counts = events['trigger'].value_counts().to_dict()
    
    # Count by trial/state if decoded
    trial_state_counts = {}
    if 'trial_num' in events.columns and 'state_num' in events.columns:
        n_trials = events['trial_num'].nunique()
        n_states_per_trial = events.groupby('trial_num')['state_num'].nunique().to_dict()
        trial_state_counts = {
            'n_trials': n_trials,
            'states_per_trial': n_states_per_trial
        }
    
    return {
        'n_events': n_events,
        'trigger_counts': {int(k): int(v) for k, v in trigger_counts.items()},
        'trial_state_info': trial_state_counts
    }

#=================================================================================================