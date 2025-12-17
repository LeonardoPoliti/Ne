#=================================================================================================
#                              Kinematics Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

#--------------------------------------------------------------------------------------

def check_kinematics_quality(kinematics_data, 
                              rigid_body_markers: List[List[str]] = None) -> Dict[str, Any]:
    """
    Run quality checks on kinematics (Vicon markers) data.
    
    Args:
        kinematics_data: KinematicsData object 
        rigid_body_markers: Optional list of marker groups that should have constant 
                           inter-marker distances (e.g., reference markers)
    
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # Metadata
    results['metadata'] = {
        'sampling_rate': kinematics_data.sampling_rate,
        'n_markers': kinematics_data.n_markers,
        'marker_names': kinematics_data.marker_names,
        'n_frames': len(kinematics_data.time),
        'duration_s': kinematics_data.get_session_duration(),
        'n_trials': kinematics_data.n_trials
    }
    
    # Temporal and Spatial Integrity
    results['temporal'] = _check_velocity_acceleration(kinematics_data)
    
    # Inter-marker distances for rigid bodies
    if rigid_body_markers:
        results['rigid_body'] = _check_rigid_body_distances(kinematics_data, rigid_body_markers)
    else:
        results['rigid_body'] = None

    # Signal Validity
    results['signal'] = _check_signal_validity(kinematics_data)
    
    # Trigger Events
    results['triggers'] = _check_trigger_events(kinematics_data)
    
    return results

#--------------------------------------------------------------------------------------

def _check_velocity_acceleration(kinematics_data) -> Dict[str, Any]:
    """Compute velocity and acceleration stats per marker."""
    results = {}
    
    for marker_name in kinematics_data.marker_names:
        try:
            vel_df = kinematics_data.compute_velocity(marker_name)
            acc_df = kinematics_data.compute_acceleration(marker_name)
            
            speed = vel_df['speed'].dropna().values
            acc_mag = acc_df['magnitude'].dropna().values
            
            results[marker_name] = {
                'velocity': {
                    'median': float(np.median(speed)) if len(speed) > 0 else None,
                    'mean': float(np.mean(speed)) if len(speed) > 0 else None,
                    'std': float(np.std(speed)) if len(speed) > 0 else None,
                    'min': float(np.min(speed)) if len(speed) > 0 else None,
                    'max': float(np.max(speed)) if len(speed) > 0 else None,
                    'values': speed  # For histogram
                },
                'acceleration': {
                    'median': float(np.median(acc_mag)) if len(acc_mag) > 0 else None,
                    'mean': float(np.mean(acc_mag)) if len(acc_mag) > 0 else None,
                    'std': float(np.std(acc_mag)) if len(acc_mag) > 0 else None,
                    'min': float(np.min(acc_mag)) if len(acc_mag) > 0 else None,
                    'max': float(np.max(acc_mag)) if len(acc_mag) > 0 else None,
                    'values': acc_mag  # For histogram
                }
            }
        except Exception as e:
            results[marker_name] = {'error': str(e)}
    
    return results

#--------------------------------------------------------------------------------------

def _check_rigid_body_distances(kinematics_data, 
                                 rigid_body_markers: List[List[str]]) -> Dict[str, Any]:
    """Check inter-marker distances for rigid body consistency."""
    results = {}
    
    markers_df = kinematics_data.markers
    
    for group_idx, marker_group in enumerate(rigid_body_markers):
        group_name = f"group_{group_idx}"
        group_results = {}
        
        # Compute all pairwise distances
        for i, m1 in enumerate(marker_group):
            for m2 in marker_group[i+1:]:
                # Get marker coordinates
                m1_clean = m1.lower().replace(' ', '_')
                m2_clean = m2.lower().replace(' ', '_')
                
                try:
                    x1 = markers_df[f'{m1_clean}_x'].values
                    y1 = markers_df[f'{m1_clean}_y'].values
                    z1 = markers_df[f'{m1_clean}_z'].values
                    
                    x2 = markers_df[f'{m2_clean}_x'].values
                    y2 = markers_df[f'{m2_clean}_y'].values
                    z2 = markers_df[f'{m2_clean}_z'].values
                    
                    # Compute distances
                    distances = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    
                    # Remove NaN/invalid
                    distances = distances[~np.isnan(distances)]
                    distances = distances[distances > 0]
                    
                    if len(distances) > 0:
                        pair_name = f"{m1}-{m2}"
                        group_results[pair_name] = {
                            'median': float(np.median(distances)),
                            'mean': float(np.mean(distances)),
                            'std': float(np.std(distances)),
                            'min': float(np.min(distances)),
                            'max': float(np.max(distances)),
                            'cv': float(np.std(distances) / np.mean(distances) * 100),  # Coefficient of variation
                            'expected': 'Should be constant (low std/cv)'
                        }
                except KeyError as e:
                    group_results[f"{m1}-{m2}"] = {'error': f'Marker not found: {e}'}
        
        results[group_name] = {
            'markers': marker_group,
            'distances': group_results
        }
    
    return results

#--------------------------------------------------------------------------------------

def _check_signal_validity(kinematics_data) -> Dict[str, Any]:
    """Check signal validity - missing frames, residuals."""
    results = {}
    
    markers_df = kinematics_data.markers
    residuals_df = kinematics_data.residuals
    time = kinematics_data.time
    n_frames = len(time)
    
    # Missing frames per marker
    missing_per_marker = {}
    gap_analysis = {}
    
    for marker_name in kinematics_data.marker_names:
        clean_name = marker_name.lower().replace(' ', '_')
        
        # Check for missing data (NaN or zero coordinates)
        x_col = f'{clean_name}_x'
        if x_col in markers_df.columns:
            x_data = markers_df[x_col].values
            
            # Missing = NaN or exactly 0 
            is_missing = np.isnan(x_data) | (x_data == 0)
            n_missing = np.sum(is_missing)
            
            missing_per_marker[marker_name] = {
                'n_missing': int(n_missing),
                'pct_missing': float(n_missing / n_frames * 100)
            }
            
            # Gap analysis
            gap_analysis[marker_name] = _analyze_gaps(is_missing, time)
    
    results['missing_frames'] = missing_per_marker
    results['gap_analysis'] = gap_analysis
    
    # Residual analysis per marker
    residual_stats = {}
    for marker_name in kinematics_data.marker_names:
        clean_name = marker_name.lower().replace(' ', '_')
        
        if clean_name in residuals_df.columns:
            res = residuals_df[clean_name].values
            
            # Filter out invalid residuals (negative or NaN)
            valid_res = res[(res >= 0) & ~np.isnan(res)]
            
            if len(valid_res) > 0:
                residual_stats[marker_name] = {
                    'median': float(np.median(valid_res)),
                    'mean': float(np.mean(valid_res)),
                    'std': float(np.std(valid_res)),
                    'min': float(np.min(valid_res)),
                    'max': float(np.max(valid_res)),
                    'values': valid_res  # For histogram
                }
            else:
                residual_stats[marker_name] = {'error': 'No valid residuals'}
    
    results['residuals'] = residual_stats
    
    return results

#--------------------------------------------------------------------------------------

def _analyze_gaps(is_missing: np.ndarray, time: np.ndarray) -> Dict[str, Any]:
    """Analyze gap durations for a marker."""
    # Find gap boundaries
    diff = np.diff(np.concatenate([[False], is_missing, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return {
            'n_gaps': 0,
            'durations_ms': [],
            'stats': None,
            'gap_info': []
        }
    
    # Calculate gap durations
    durations = []
    gap_info = []
    
    for i in range(min(len(starts), len(ends))):
        if ends[i] > starts[i]:
            start_time = time[starts[i]]
            end_time = time[min(ends[i]-1, len(time)-1)]
            dur = end_time - start_time
            durations.append(dur)
            gap_info.append({
                'start_idx': int(starts[i]),
                'end_idx': int(ends[i]),
                'start_time': float(start_time),
                'end_time': float(end_time),
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
        'n_gaps': len(durations),
        'durations_ms': durations.tolist(),
        'stats': stats,
        'gap_info': gap_info
    }

#--------------------------------------------------------------------------------------

def _check_trigger_events(kinematics_data) -> Dict[str, Any]:
    """Check trigger event counts."""
    events = kinematics_data.events
    
    if events is None or events.empty:
        return {'n_events': 0, 'n_trials': 0}
    
    n_events = len(events)
    n_trials = kinematics_data.n_trials
    
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