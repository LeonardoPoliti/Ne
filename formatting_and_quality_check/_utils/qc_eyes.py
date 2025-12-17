#=================================================================================================
#                              Eyes Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

#--------------------------------------------------------------------------------------

def check_eyes_quality(eyes_data, expected_fs: float = 1000.0, 
                       screen_bounds: tuple = None) -> Dict[str, Any]:
    """
    Run quality checks on eye-tracking data.
    
    Args:
        eyes_data: EyesData object 
        expected_fs: Expected sampling rate in Hz (default 1000 for EyeLink)
        screen_bounds: Optional tuple (x_min, x_max, y_min, y_max) for screen coordinates
    
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # Metadata
    results['metadata'] = {
        'sampling_rate': eyes_data.sampling_rate,
        'eye_tracked': eyes_data.eye_tracked,
        'pupil_mode': eyes_data.pupil_mode,
        'tracking_mode': eyes_data.tracking_mode,
        'filter_level': eyes_data.filter_level,
        'is_binocular': eyes_data._is_binocular,
        'n_samples': len(eyes_data.time),
        'duration_s': eyes_data.get_session_duration()
    }
    
    # eye event counts
    results['metadata']['n_fixations'] = len(eyes_data.fixations)
    results['metadata']['n_saccades'] = len(eyes_data.saccades)
    results['metadata']['n_blinks'] = len(eyes_data.blinks)
    
    # trial count from events - use max trial number (not nunique)
    # First ensure trial_num column exists by decoding if needed
    if eyes_data.events is not None and not eyes_data.events.empty:
        if 'trial_num' not in eyes_data.events.columns:
            if hasattr(eyes_data, 'decode_trial_events'):
                eyes_data.decode_trial_events()
        
        if 'trial_num' in eyes_data.events.columns:
            trial_nums = eyes_data.events['trial_num'].dropna()
            if len(trial_nums) > 0:
                results['metadata']['n_trials'] = int(trial_nums.max())
            else:
                results['metadata']['n_trials'] = None
        else:
            results['metadata']['n_trials'] = None
    else:
        results['metadata']['n_trials'] = None
    
    # Temporal integrity
    results['temporal'] = _check_temporal_integrity(eyes_data, expected_fs)
    
    # Signal validity
    results['signal'] = _check_signal_validity(eyes_data, screen_bounds)
    
    # Event consistency
    results['events'] = _check_eye_event_consistency(eyes_data)
    
    # Binocular checks
    if eyes_data._is_binocular:
        results['binocular'] = _check_binocular_consistency(eyes_data)
    else:
        results['binocular'] = None
    
    # Trigger events
    results['triggers'] = _check_trigger_events(eyes_data)
    
    return results

#--------------------------------------------------------------------------------------

def _check_temporal_integrity(eyes_data, expected_fs: float) -> Dict[str, Any]:
    results = {}
    
    # Sampling rate check
    results['expected_fs'] = expected_fs
    results['actual_fs'] = eyes_data.sampling_rate
    if eyes_data.sampling_rate:
        results['fs_deviation_pct'] = abs(eyes_data.sampling_rate - expected_fs) / expected_fs * 100
    else:
        results['fs_deviation_pct'] = None
    
    # Duplicate timestamps in time array
    time = eyes_data.time
    unique_times = np.unique(time)
    n_duplicates = len(time) - len(unique_times)
    results['duplicate_timestamps'] = int(n_duplicates)
    results['duplicate_pct'] = float(n_duplicates / len(time) * 100) if len(time) > 0 else 0
    
    # Event duration coverage - handle binocular data properly
    total_duration = eyes_data.get_session_duration() * 1000  # Convert to ms
    is_binocular = eyes_data._is_binocular
    
    # For binocular data, events are logged per eye, so we need to compute coverage per eye
    if is_binocular:
        fix_df = eyes_data.fixations
        sacc_df = eyes_data.saccades
        blink_df = eyes_data.blinks
        
        # Try to separate by eye if 'eye' column exists
        if not fix_df.empty and 'eye' in fix_df.columns:
            # Compute per-eye coverage
            coverage_per_eye = {}
            for eye in ['L', 'R']:
                fix_eye = fix_df[fix_df['eye'] == eye]['duration'].sum() if not fix_df.empty else 0
                sacc_eye = sacc_df[sacc_df['eye'] == eye]['duration'].sum() if not sacc_df.empty and 'eye' in sacc_df.columns else 0
                blink_eye = blink_df[blink_df['eye'] == eye]['duration'].sum() if not blink_df.empty and 'eye' in blink_df.columns else 0
                event_total_eye = fix_eye + sacc_eye + blink_eye
                coverage_per_eye[eye] = {
                    'fixation_ms': float(fix_eye),
                    'saccade_ms': float(sacc_eye),
                    'blink_ms': float(blink_eye),
                    'event_total_ms': float(event_total_eye),
                    'coverage_pct': float(event_total_eye / total_duration * 100) if total_duration > 0 else 0
                }
            
            # Average coverage across eyes
            avg_coverage = (coverage_per_eye.get('L', {}).get('coverage_pct', 0) + 
                          coverage_per_eye.get('R', {}).get('coverage_pct', 0)) / 2
            
            results['duration_coverage'] = {
                'total_recording_ms': float(total_duration),
                'is_binocular': True,
                'per_eye': coverage_per_eye,
                'coverage_pct': float(avg_coverage),
                'note': 'Coverage computed per eye and averaged for binocular data'
            }
        else:
            # No eye column - assume events are duplicated for both eyes
            fix_duration = fix_df['duration'].sum() if not fix_df.empty else 0
            sacc_duration = sacc_df['duration'].sum() if not sacc_df.empty else 0
            blink_duration = blink_df['duration'].sum() if not blink_df.empty else 0
            event_total = fix_duration + sacc_duration + blink_duration
            
            # For binocular without eye column, estimate by dividing by 2
            estimated_single_eye_total = event_total / 2
            
            results['duration_coverage'] = {
                'total_recording_ms': float(total_duration),
                'is_binocular': True,
                'fixation_ms': float(fix_duration / 2),
                'saccade_ms': float(sacc_duration / 2),
                'blink_ms': float(blink_duration / 2),
                'event_total_ms': float(estimated_single_eye_total),
                'coverage_pct': float(estimated_single_eye_total / total_duration * 100) if total_duration > 0 else 0,
                'note': 'Binocular data - coverage estimated by dividing total by 2'
            }
    else:
        # Monocular - standard calculation
        fix_duration = eyes_data.fixations['duration'].sum() if not eyes_data.fixations.empty else 0
        sacc_duration = eyes_data.saccades['duration'].sum() if not eyes_data.saccades.empty else 0
        blink_duration = eyes_data.blinks['duration'].sum() if not eyes_data.blinks.empty else 0
        event_total = fix_duration + sacc_duration + blink_duration
        
        results['duration_coverage'] = {
            'total_recording_ms': float(total_duration),
            'is_binocular': False,
            'fixation_ms': float(fix_duration),
            'saccade_ms': float(sacc_duration),
            'blink_ms': float(blink_duration),
            'event_total_ms': float(event_total),
            'coverage_pct': float(event_total / total_duration * 100) if total_duration > 0 else 0
        }
    
    # Check for overlapping events
    results['overlapping_events'] = _check_overlapping_events(eyes_data)
    
    return results

#--------------------------------------------------------------------------------------

def _check_overlapping_events(eyes_data) -> Dict[str, int]:
    """Check for overlapping events between fixations, saccades, blinks."""
    overlaps = {'fixation_saccade': 0, 'fixation_blink': 0, 'saccade_blink': 0}
    
    # Get event times
    def get_intervals(df):
        if df.empty:
            return []
        return list(zip(df['start'].values, df['end'].values))
    
    fix_intervals = get_intervals(eyes_data.fixations)
    sacc_intervals = get_intervals(eyes_data.saccades)
    blink_intervals = get_intervals(eyes_data.blinks)
    
    def count_overlaps(intervals1, intervals2):
        count = 0
        for s1, e1 in intervals1:
            for s2, e2 in intervals2:
                # Check if intervals overlap
                if s1 < e2 and s2 < e1:
                    count += 1
        return count
    
    overlaps['fixation_saccade'] = count_overlaps(fix_intervals, sacc_intervals)
    overlaps['fixation_blink'] = count_overlaps(fix_intervals, blink_intervals)
    overlaps['saccade_blink'] = count_overlaps(sacc_intervals, blink_intervals)
    
    return overlaps

#--------------------------------------------------------------------------------------

def _check_signal_validity(eyes_data, screen_bounds: tuple) -> Dict[str, Any]:
    results = {}
    
    gaze = eyes_data.gaze
    pupil = eyes_data.pupil
    
    # NaN samples in gaze (from blinks/tracking loss)
    gaze_nan = gaze.isna().sum()
    total_samples = len(gaze)
    
    results['nan_samples_gaze'] = {col: int(gaze_nan[col]) for col in gaze.columns}
    results['nan_pct_gaze'] = {col: float(gaze_nan[col] / total_samples * 100) for col in gaze.columns}
    
    # Gap duration analysis (consecutive NaNs)
    results['gap_durations'] = _analyze_gaps(eyes_data)
    
    # Zero or negative pupil values
    results['invalid_pupil'] = _check_invalid_pupil(eyes_data)
    
    # Saccade vs fixation count check
    n_saccades = len(eyes_data.saccades)
    n_fixations = len(eyes_data.fixations)
    results['saccade_fixation_ratio'] = {
        'n_saccades': n_saccades,
        'n_fixations': n_fixations,
        'ratio': float(n_saccades / n_fixations) if n_fixations > 0 else None,
        'expected_ratio': 1.0  # Each saccade should be followed by a fixation
    }
    
    # Gaze velocity during fixations check
    results['fixation_velocity'] = _check_fixation_velocity(eyes_data)
    
    # Gaze velocity and acceleration stats
    results['velocity_stats'] = _compute_velocity_stats(eyes_data)
    
    # Gaze distribution for heatmap
    results['gaze_distribution'] = _compute_gaze_distribution(eyes_data, screen_bounds)
    
    return results

#--------------------------------------------------------------------------------------

def _analyze_gaps(eyes_data) -> Dict[str, Any]:
    """Analyze gap durations (consecutive NaN periods)."""
    gaze = eyes_data.gaze
    time = eyes_data.time
    
    # Use first gaze column for gap analysis
    gaze_col = gaze.columns[0]
    is_nan = gaze[gaze_col].isna().values
    
    # Find gap boundaries
    diff = np.diff(np.concatenate([[False], is_nan, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return {
            'n_gaps': 0,
            'durations_ms': [],
            'stats': None
        }
    
    # Calculate gap durations
    durations = []
    for i in range(min(len(starts), len(ends))):
        if ends[i] > starts[i] and ends[i] <= len(time):
            dur = time[min(ends[i]-1, len(time)-1)] - time[starts[i]]
            durations.append(dur)
    
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
        'n_gaps': len(durations),
        'durations_ms': durations.tolist(),
        'stats': stats
    }

#--------------------------------------------------------------------------------------

def _check_invalid_pupil(eyes_data) -> Dict[str, Any]:
    """Check for zero, negative, or NaN pupil values."""
    pupil = eyes_data.pupil
    results = {}
    
    for col in pupil.columns:
        data = pupil[col].dropna().values
        n_zero = int(np.sum(data == 0))
        n_negative = int(np.sum(data < 0))
        n_nan = int(np.sum(np.isnan(pupil[col].values)))
        
        results[col] = {
            'n_zero': n_zero,
            'n_negative': n_negative,
            'n_nan': n_nan,
            'total_invalid': n_zero + n_negative + n_nan
        }
    
    return results

#--------------------------------------------------------------------------------------

def _check_fixation_velocity(eyes_data) -> Dict[str, Any]:
    """Check gaze velocity during fixations (should be ~0)."""
    fixations = eyes_data.fixations
    
    if fixations.empty:
        return {'error': 'No fixations found'}
    
    # Compute velocity
    try:
        vel_df = eyes_data.compute_velocity()
    except:
        return {'error': 'Could not compute velocity'}
    
    # Check velocity during fixations
    fix_velocities = []
    
    for _, fix in fixations.iterrows():
        mask = (vel_df['time'] >= fix['start']) & (vel_df['time'] <= fix['end'])
        if mask.any():
            speeds = vel_df.loc[mask, 'speed'].values
            fix_velocities.extend(speeds[~np.isnan(speeds)])
    
    if len(fix_velocities) == 0:
        return {'error': 'No velocity data during fixations'}
    
    fix_velocities = np.array(fix_velocities)
    
    return {
        'median': float(np.median(fix_velocities)),
        'mean': float(np.mean(fix_velocities)),
        'std': float(np.std(fix_velocities)),
        'max': float(np.max(fix_velocities)),
        'expected': 'Should be near 0'
    }

#--------------------------------------------------------------------------------------

def _compute_velocity_stats(eyes_data) -> Dict[str, Any]:
    try:
        vel_df = eyes_data.compute_velocity()  # Returns pixels/second by default
        acc_df = eyes_data.compute_acceleration()  # Returns pixels/second^2 by default
    except:
        return {'error': 'Could not compute velocity/acceleration'}
    
    # Velocity stats - cap to plausible range (0-3000 px/s for typical displays)
    speed = vel_df['speed'].dropna().values
    speed_capped = speed[speed <= 3000]  # Remove implausibly high values
    
    vel_stats = {
        'median': float(np.median(speed_capped)) if len(speed_capped) > 0 else 0,
        'mean': float(np.mean(speed_capped)) if len(speed_capped) > 0 else 0,
        'std': float(np.std(speed_capped)) if len(speed_capped) > 0 else 0,
        'min': float(np.min(speed_capped)) if len(speed_capped) > 0 else 0,
        'max': float(np.max(speed_capped)) if len(speed_capped) > 0 else 0,
        'values': speed_capped,  # For histogram
        'n_capped': int(len(speed) - len(speed_capped))  # Track how many were removed
    }
    
    # Acceleration stats - cap to plausible range (0-200000 px/s^2 for typical displays)
    # Use percentile-based capping to be more robust
    acc_mag = acc_df['magnitude'].dropna().values
    if len(acc_mag) > 0:
        # Cap at 99.5th percentile or 200000, whichever is lower
        cap_value = min(np.percentile(acc_mag, 99.5), 200000)
        acc_capped = acc_mag[acc_mag <= cap_value]
    else:
        acc_capped = acc_mag
    
    acc_stats = {
        'median': float(np.median(acc_capped)) if len(acc_capped) > 0 else 0,
        'mean': float(np.mean(acc_capped)) if len(acc_capped) > 0 else 0,
        'std': float(np.std(acc_capped)) if len(acc_capped) > 0 else 0,
        'min': float(np.min(acc_capped)) if len(acc_capped) > 0 else 0,
        'max': float(np.max(acc_capped)) if len(acc_capped) > 0 else 0,
        'values': acc_capped,  # For histogram
        'n_capped': int(len(acc_mag) - len(acc_capped))  # Track how many were removed
    }
    
    return {'velocity': vel_stats, 'acceleration': acc_stats}

#--------------------------------------------------------------------------------------

def _compute_gaze_distribution(eyes_data, screen_bounds: tuple) -> Dict[str, Any]:
    """ Compute gaze distribution for heatmap."""
    gaze = eyes_data.gaze
    
    # Get x,y coordinates (average if binocular)
    if eyes_data._is_binocular:
        x = ((gaze['x_l'] + gaze['x_r']) / 2).dropna().values
        y = ((gaze['y_l'] + gaze['y_r']) / 2).dropna().values
    else:
        x_col = [c for c in gaze.columns if 'x' in c.lower()][0]
        y_col = [c for c in gaze.columns if 'y' in c.lower()][0]
        x = gaze[x_col].dropna().values
        y = gaze[y_col].dropna().values
    
    # Use minimum length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    return {
        'x': x,
        'y': y,
        'screen_bounds': screen_bounds
    }

#--------------------------------------------------------------------------------------

def _check_eye_event_consistency(eyes_data) -> Dict[str, Any]:

    results = {}
    
    # Fixation duration stats
    if not eyes_data.fixations.empty:
        fix_dur = eyes_data.fixations['duration'].values
        results['fixation_duration'] = {
            'median': float(np.median(fix_dur)),
            'mean': float(np.mean(fix_dur)),
            'std': float(np.std(fix_dur)),
            'min': float(np.min(fix_dur)),
            'max': float(np.max(fix_dur))
        }
    
    # Saccade stats
    if not eyes_data.saccades.empty:
        sacc_dur = eyes_data.saccades['duration'].values
        results['saccade_duration'] = {
            'median': float(np.median(sacc_dur)),
            'mean': float(np.mean(sacc_dur)),
            'std': float(np.std(sacc_dur)),
            'min': float(np.min(sacc_dur)),
            'max': float(np.max(sacc_dur))
        }
        
        if 'amplitude' in eyes_data.saccades.columns:
            sacc_amp = eyes_data.saccades['amplitude'].dropna().values
            results['saccade_amplitude'] = {
                'median': float(np.median(sacc_amp)),
                'mean': float(np.mean(sacc_amp)),
                'std': float(np.std(sacc_amp)),
                'min': float(np.min(sacc_amp)),
                'max': float(np.max(sacc_amp))
            }
    
    # Blink stats
    if not eyes_data.blinks.empty:
        blink_dur = eyes_data.blinks['duration'].values
        results['blink_duration'] = {
            'median': float(np.median(blink_dur)),
            'mean': float(np.mean(blink_dur)),
            'std': float(np.std(blink_dur)),
            'min': float(np.min(blink_dur)),
            'max': float(np.max(blink_dur))
        }
    
    return results

#--------------------------------------------------------------------------------------

def _check_binocular_consistency(eyes_data) -> Dict[str, Any]:

    gaze = eyes_data.gaze
    pupil = eyes_data.pupil
    
    results = {}
    
    # Vergence check: left/right gaze position difference
    if 'x_l' in gaze.columns and 'x_r' in gaze.columns:
        x_diff = (gaze['x_l'] - gaze['x_r']).dropna().values
        y_diff = (gaze['y_l'] - gaze['y_r']).dropna().values
        
        results['vergence'] = {
            'x_diff_mean': float(np.mean(x_diff)),
            'x_diff_std': float(np.std(x_diff)),
            'y_diff_mean': float(np.mean(y_diff)),
            'y_diff_std': float(np.std(y_diff))
        }
    
    # Left/right gaze correlation
    if 'x_l' in gaze.columns and 'x_r' in gaze.columns:
        # Remove NaN for correlation
        mask = ~(gaze['x_l'].isna() | gaze['x_r'].isna())
        if mask.sum() > 2:
            x_corr = np.corrcoef(gaze.loc[mask, 'x_l'], gaze.loc[mask, 'x_r'])[0, 1]
            y_corr = np.corrcoef(gaze.loc[mask, 'y_l'], gaze.loc[mask, 'y_r'])[0, 1]
            results['gaze_correlation'] = {
                'x': float(x_corr),
                'y': float(y_corr)
            }
    
    # Left/right pupil correlation
    if 'pupil_l' in pupil.columns and 'pupil_r' in pupil.columns:
        mask = ~(pupil['pupil_l'].isna() | pupil['pupil_r'].isna())
        if mask.sum() > 2:
            pupil_corr = np.corrcoef(pupil.loc[mask, 'pupil_l'], pupil.loc[mask, 'pupil_r'])[0, 1]
            results['pupil_correlation'] = float(pupil_corr)
    
    return results

#--------------------------------------------------------------------------------------

def _check_trigger_events(eyes_data) -> Dict[str, Any]:
    """Check trigger event counts from messages."""
    events = eyes_data.events
    
    if events is None or events.empty:
        return {'n_events': 0, 'n_unique_triggers': 0, 'trial_state_info': {}}
    
    n_events = len(events)
    
    # Count unique trigger messages (numeric messages)
    n_unique_triggers = 0
    if 'message' in events.columns:
        # Get numeric messages
        numeric_mask = events['message'].str.match(r'^\d+$', na=False)
        if numeric_mask.any():
            numeric_events = events.loc[numeric_mask, 'message'].astype(int)
            n_unique_triggers = numeric_events.nunique()
    
    # Count by trial/state if decoded
    trial_state_info = {}
    if 'trial_num' in events.columns and 'state_num' in events.columns:
        n_trials = int(events['trial_num'].dropna().max()) if len(events['trial_num'].dropna()) > 0 else 0
        states_per_trial = events.groupby('trial_num')['state_num'].nunique().to_dict()
        trial_state_info = {
            'n_trials': n_trials,
            'states_per_trial': states_per_trial
        }
    
    return {
        'n_events': n_events,
        'n_unique_triggers': n_unique_triggers,
        'trial_state_info': trial_state_info
    }

#=================================================================================================