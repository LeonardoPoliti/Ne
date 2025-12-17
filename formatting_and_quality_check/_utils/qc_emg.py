#=================================================================================================
#                              EMG Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import signal

#--------------------------------------------------------------------------------------

def check_emg_quality(emg_data) -> Dict[str, Any]:
    """
    Run quality checks on EMG data.
    
    Args:
        emg_data: EMGData object 
    
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # Metadata
    results['metadata'] = {
        'sampling_rate': emg_data.sampling_rate,
        'n_channels': emg_data.n_channels,
        'channel_names': emg_data.channel_names,
        'n_samples': len(emg_data.time),
        'duration_s': emg_data.get_session_duration(),
        'n_trials': emg_data.n_trials
    }
    
    # Signal Validity
    results['signal'] = _check_signal_validity(emg_data)
    
    # Trigger Events
    results['triggers'] = _check_trigger_events(emg_data)
    
    return results

#--------------------------------------------------------------------------------------

def _check_signal_validity(emg_data) -> Dict[str, Any]:
    """Check signal validity metrics for EMG."""
    results = {}
    
    # DC offset per channel
    results['dc_offset'] = _compute_dc_offset(emg_data)
    
    # Power spectral density
    results['psd'] = _compute_psd(emg_data)
    
    # Clipping/saturation detection
    results['clipping'] = _detect_clipping(emg_data)
    
    return results

#--------------------------------------------------------------------------------------

def _compute_dc_offset(emg_data) -> Dict[str, Any]:
    """Compute DC offset (mean voltage) per channel."""
    results = {}
    
    for col in emg_data.channels.columns:
        data = emg_data.channels[col].values
        # Remove NaN
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            results[col] = {
                'mean': float(np.mean(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data))
            }
        else:
            results[col] = {'error': 'No valid data'}
    
    return results

#--------------------------------------------------------------------------------------

def _compute_psd(emg_data, nperseg: int = 1024) -> Dict[str, Any]:
    """Compute power spectral density for each channel."""
    results = {}
    
    fs = emg_data.sampling_rate
    if fs is None or fs <= 0:
        return {'error': 'Invalid sampling rate'}
    
    for col in emg_data.channels.columns:
        data = emg_data.channels[col].values
        valid_data = data[~np.isnan(data)]
        
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

def _detect_clipping(emg_data, n_consecutive: int = 3) -> Dict[str, Any]:
    """Detect clipping/saturation (consecutive samples at min/max)."""
    results = {}
    
    for col in emg_data.channels.columns:
        data = emg_data.channels[col].values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            results[col] = {'error': 'No valid data'}
            continue
        
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
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

def _check_trigger_events(emg_data) -> Dict[str, Any]:
    """Check trigger event counts."""
    events = emg_data.events
    
    if events is None or events.empty:
        return {'n_events': 0, 'n_trials': 0}
    
    n_events = len(events)
    n_trials = emg_data.n_trials
    
    # Trials info
    trial_info = {}
    if 'trial_num' in events.columns:
        for trial_num in events['trial_num'].unique():
            trial_events = events[events['trial_num'] == trial_num]
            start_event = trial_events[trial_events['event_type'] == 'start']
            end_event = trial_events[trial_events['event_type'] == 'end']
            
            trial_info[int(trial_num)] = {
                'has_start': not start_event.empty,
                'has_end': not end_event.empty,
                'start_time': float(start_event['time'].iloc[0]) if not start_event.empty else None,
                'end_time': float(end_event['time'].iloc[0]) if not end_event.empty else None
            }
    
    return {
        'n_events': n_events,
        'n_trials': n_trials,
        'trial_info': trial_info
    }

#=================================================================================================