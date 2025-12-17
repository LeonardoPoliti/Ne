#=================================================================================================
#                         Cross-Modal Alignment Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
from typing import Dict, Any
from scipy import stats

#--------------------------------------------------------------------------------------

def check_crossmodal_alignment(task_data=None, eeg_data=None, eyes_data=None, 
                                kinematics_data=None, emg_data=None) -> Dict[str, Any]:
    """
    Check cross-modal alignment and consistency across data modalities.
    
    Args:
        task_data: TrialsData object (loaded) - optional
        eeg_data: EEGData object (loaded) - optional
        eyes_data: EyesData object (loaded) - optional
        kinematics_data: KinematicsData object (loaded) - optional
        emg_data: EMGData object (loaded) - optional
    
    Returns:
        Dictionary with alignment check results
    """
    results = {}
    
    # Collect available modalities
    modalities = {}
    if task_data is not None:
        modalities['task'] = task_data
    if eeg_data is not None:
        modalities['eeg'] = eeg_data
    if eyes_data is not None:
        modalities['eyes'] = eyes_data
    if kinematics_data is not None:
        modalities['kinematics'] = kinematics_data
    if emg_data is not None:
        modalities['emg'] = emg_data
    
    if len(modalities) < 2:
        return {'error': 'Need at least 2 modalities for cross-modal alignment check'}
    
    # Consisency 
    results['consistency'] = _check_consistency(modalities)
    
    # Event alignment 
    results['event_alignment'] = _check_event_alignment(modalities)
    
    return results

#--------------------------------------------------------------------------------------

def _check_consistency(modalities: Dict) -> Dict[str, Any]:
    """Check consistency of duration, trials, and states across modalities."""
    results = {}
    
    # Recording durations
    durations = {}
    for name, data in modalities.items():
        if name == 'task':
            # Task doesn't have continuous time
            continue
        try:
            dur = data.get_session_duration()
            if dur is not None:
                durations[name] = dur
        except:
            pass
    
    results['durations'] = {
        'values_s': durations,
        'min': min(durations.values()) if durations else None,
        'max': max(durations.values()) if durations else None,
        'range': max(durations.values()) - min(durations.values()) if len(durations) > 1 else 0
    }

    # Trial counts
    trial_counts = {}
    
    # From task - use len()
    if 'task' in modalities:
        trial_counts['task'] = len(modalities['task'])
    
    # From kinematics - use n_trials attribute
    if 'kinematics' in modalities:
        kin = modalities['kinematics']
        if hasattr(kin, 'n_trials') and kin.n_trials is not None:
            trial_counts['kinematics'] = kin.n_trials
    
    # From EMG - use n_trials attribute
    if 'emg' in modalities:
        emg = modalities['emg']
        if hasattr(emg, 'n_trials') and emg.n_trials is not None:
            trial_counts['emg'] = emg.n_trials
    
    # From EEG - check events DataFrame for trial_num (call decode_trial_events if needed)
    if 'eeg' in modalities:
        eeg = modalities['eeg']
        try:
            if hasattr(eeg, 'events') and eeg.events is not None:
                events_df = eeg.events
                if hasattr(events_df, 'empty') and not events_df.empty:
                    # Check if trial_num column exists, if not try to decode
                    if 'trial_num' not in events_df.columns:
                        if hasattr(eeg, 'decode_trial_events'):
                            eeg.decode_trial_events()
                            events_df = eeg.events  # Refresh reference
                    
                    # Now try to get trial count
                    if 'trial_num' in events_df.columns:
                        trial_nums = events_df['trial_num'].dropna()
                        if len(trial_nums) > 0:
                            trial_counts['eeg'] = int(trial_nums.max())
        except Exception as e:
            pass  # Silently fail - EEG trial count is optional
    
    # From eyes - check events DataFrame for trial_num (call decode_trial_events if needed)
    if 'eyes' in modalities:
        eyes = modalities['eyes']
        try:
            if hasattr(eyes, 'events') and eyes.events is not None:
                events_df = eyes.events
                if hasattr(events_df, 'empty') and not events_df.empty:
                    # Check if trial_num column exists, if not try to decode
                    if 'trial_num' not in events_df.columns:
                        if hasattr(eyes, 'decode_trial_events'):
                            eyes.decode_trial_events()
                            events_df = eyes.events  # Refresh reference
                    
                    # Now try to get trial count
                    if 'trial_num' in events_df.columns:
                        trial_nums = events_df['trial_num'].dropna()
                        if len(trial_nums) > 0:
                            trial_counts['eyes'] = int(trial_nums.max())
        except Exception as e:
            pass  # Silently fail - eyes trial count is optional
    
    results['trial_counts'] = {
        'values': trial_counts,
        'consistent': len(set(trial_counts.values())) == 1 if len(trial_counts) > 1 else None
    }
    
    return results

#--------------------------------------------------------------------------------------

def _check_event_alignment(modalities: Dict) -> Dict[str, Any]:
    """Check event timing alignment across modalities."""
    results = {}
    
    # Get trial start/end times from modalities that have them
    # Format: {modality_name: {trial_num: {'start': time, 'end': time}}}
    trial_times = {}
    
    # Kinematics trial times (most reliable - from trigger channel)
    if 'kinematics' in modalities:
        kin = modalities['kinematics']
        if hasattr(kin, 'n_trials') and kin.n_trials and kin.n_trials > 0:
            kin_trials = {}
            for trial_num in range(1, kin.n_trials + 1):
                try:
                    times = kin.get_trial_times(trial_num)
                    if times:
                        kin_trials[trial_num] = {'start': times[0], 'end': times[1]}
                except:
                    pass
            if kin_trials:
                trial_times['kinematics'] = kin_trials
    
    # EMG trial times
    if 'emg' in modalities:
        emg = modalities['emg']
        if hasattr(emg, 'n_trials') and emg.n_trials and emg.n_trials > 0:
            emg_trials = {}
            for trial_num in range(1, emg.n_trials + 1):
                try:
                    times = emg.get_trial_times(trial_num)
                    if times:
                        emg_trials[trial_num] = {'start': times[0], 'end': times[1]}
                except:
                    pass
            if emg_trials:
                trial_times['emg'] = emg_trials
    
    # EEG trial times (if available) - call decode_trial_events if needed
    if 'eeg' in modalities:
        eeg = modalities['eeg']
        try:
            if hasattr(eeg, 'get_trial_times') and hasattr(eeg, 'events'):
                if eeg.events is not None and hasattr(eeg.events, 'empty') and not eeg.events.empty:
                    # Check if trial_num exists, if not try to decode
                    if 'trial_num' not in eeg.events.columns:
                        if hasattr(eeg, 'decode_trial_events'):
                            eeg.decode_trial_events()
                    
                    # Now get trial times
                    if 'trial_num' in eeg.events.columns:
                        eeg_trials = {}
                        unique_trials = eeg.events['trial_num'].dropna().unique()
                        for trial_num in unique_trials:
                            try:
                                times = eeg.get_trial_times(int(trial_num))
                                if times and len(times) >= 2:
                                    eeg_trials[int(trial_num)] = {'start': times[0], 'end': times[1]}
                            except:
                                pass
                        if eeg_trials:
                            trial_times['eeg'] = eeg_trials
        except Exception as e:
            pass  # Silently fail
    
    # Eyes trial times (if available) - call decode_trial_events if needed
    if 'eyes' in modalities:
        eyes = modalities['eyes']
        try:
            if hasattr(eyes, 'get_trial_times') and hasattr(eyes, 'events'):
                if eyes.events is not None and hasattr(eyes.events, 'empty') and not eyes.events.empty:
                    # Check if trial_num exists, if not try to decode
                    if 'trial_num' not in eyes.events.columns:
                        if hasattr(eyes, 'decode_trial_events'):
                            eyes.decode_trial_events()
                    
                    # Now get trial times
                    if 'trial_num' in eyes.events.columns:
                        eyes_trials = {}
                        unique_trials = eyes.events['trial_num'].dropna().unique()
                        for trial_num in unique_trials:
                            try:
                                times = eyes.get_trial_times(int(trial_num))
                                if times and len(times) >= 2:
                                    eyes_trials[int(trial_num)] = {'start': times[0], 'end': times[1]}
                            except:
                                pass
                        if eyes_trials:
                            trial_times['eyes'] = eyes_trials
        except Exception as e:
            pass  # Silently fail
    
    results['modalities_with_trial_times'] = list(trial_times.keys())
    results['trials_per_modality'] = {k: len(v) for k, v in trial_times.items()}
    
    # ==================== OFFSET ANALYSIS ====================
    if len(trial_times) >= 2:
        try:
            results['offset_analysis'] = _compute_offset_analysis(trial_times)
        except Exception as e:
            results['offset_analysis'] = {'error': f'Offset analysis failed: {str(e)}'}
    else:
        results['offset_analysis'] = {'note': 'Need at least 2 modalities with trial times'}
    
    return results

#--------------------------------------------------------------------------------------

def _compute_offset_analysis(trial_times: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[str, Any]:
    """
    Compute systematic offset between modalities using regression.
    
    Args:
        trial_times: Dictionary format:
            {
                'modality1': {trial_num: {'start': time, 'end': time}},
                'modality2': {trial_num: {'start': time, 'end': time}},
                ...
            }
    
    Returns:
        Dictionary with offset analysis for each pair of modalities
    """
    results = {}
    
    modality_names = list(trial_times.keys())
    
    for i, mod1 in enumerate(modality_names):
        for mod2 in modality_names[i+1:]:
            pair_name = f"{mod1}_vs_{mod2}"
            
            # Get trial dictionaries
            trials_mod1 = trial_times[mod1]  # {trial_num: {'start': ..., 'end': ...}}
            trials_mod2 = trial_times[mod2]
            
            # Find common trials
            common_trials = set(trials_mod1.keys()) & set(trials_mod2.keys())
            
            if len(common_trials) < 3:
                results[pair_name] = {'error': f'Insufficient matched trials ({len(common_trials)})'}
                continue
            
            # Extract start times for matching
            matched_times = []
            for trial_num in sorted(common_trials):
                start1 = trials_mod1[trial_num]['start']
                start2 = trials_mod2[trial_num]['start']
                matched_times.append((start1, start2))
            
            times1 = np.array([t[0] for t in matched_times])
            times2 = np.array([t[1] for t in matched_times])
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(times1, times2)
            
            # Compute differences
            diffs = times2 - times1
            
            results[pair_name] = {
                'n_matched_events': len(matched_times),
                'regression': {
                    'slope': float(slope),
                    'intercept_ms': float(intercept),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value)
                },
                'offset_stats': {
                    'median_ms': float(np.median(diffs)),
                    'mean_ms': float(np.mean(diffs)),
                    'std_ms': float(np.std(diffs)),
                    'min_ms': float(np.min(diffs)),
                    'max_ms': float(np.max(diffs))
                },
                'drift_estimate': {
                    'slope_deviation': float(abs(slope - 1.0)),
                    'has_drift': abs(slope - 1.0) > 0.001
                }
            }
    
    return results

#=================================================================================================