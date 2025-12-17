#=================================================================================================
#                              Task Data Quality Check - NeuroRobCoRe
#=================================================================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

#--------------------------------------------------------------------------------------

def check_task_quality(metadata, trials_data) -> Dict[str, Any]:
    """
    Run quality checks on task metadata and trials data.
    
    Args:
        metadata: Metadata object 
        trials_data: TrialsData object 
    
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # metadata
    results['metadata'] = {}
    results['metadata']['sections'] = metadata.get_data_attributes()
    results['metadata']['content'] = metadata.get_raw_data()
    
    # Trials data
    results['trials'] = {}
    
    # Column names
    results['trials']['columns'] = trials_data.get_data_attributes()
    
    # NaN count per column
    nan_counts = {}
    for col in trials_data._data_column_names:
        arr = getattr(trials_data, col)
        if arr is not None:
            nan_counts[col] = int(np.sum(pd.isna(arr)))
    results['trials']['nan_counts'] = nan_counts
    results['trials']['total_nan'] = sum(nan_counts.values())
    
    # Trial numbering check
    if hasattr(trials_data, 'trial_num'):
        trial_nums = trials_data.trial_num
        results['trials']['n_trials'] = len(trial_nums)
        
        # Check sequential
        sorted_trials = np.sort(trial_nums)
        expected = np.arange(sorted_trials[0], sorted_trials[0] + len(sorted_trials))
        is_sequential = np.array_equal(sorted_trials, expected)
        results['trials']['is_sequential'] = bool(is_sequential)
        
        # Check for gaps
        diffs = np.diff(sorted_trials)
        gaps = np.where(diffs > 1)[0]
        results['trials']['gaps'] = [int(sorted_trials[g]) for g in gaps] if len(gaps) > 0 else []
        
        # Check for duplicates
        unique, counts = np.unique(trial_nums, return_counts=True)
        duplicates = unique[counts > 1]
        results['trials']['duplicates'] = duplicates.tolist()
    else:
        results['trials']['n_trials'] = len(trials_data)
        results['trials']['is_sequential'] = None
        results['trials']['gaps'] = None
        results['trials']['duplicates'] = None
    
    # Trial duration statistics
    results['trials']['duration'] = _compute_trial_durations(trials_data)
    
    return results

#--------------------------------------------------------------------------------------

def _compute_trial_durations(trials_data) -> Optional[Dict[str, Any]]:
    """Compute trial duration statistics from events DataFrame."""
    events = trials_data.events
    
    if events is None or events.empty:
        return None
    
    # Try to find trial start/end times
    # Look for timing columns 
    time_cols = [c for c in events.columns if 'time' in c.lower()]
    
    if len(time_cols) < 2:
        return None
    
    start_col = None
    end_col = None
    
    for col in events.columns:
        col_lower = col.lower()
        if 'start' in col_lower or col_lower == time_cols[0]:
            start_col = col
        if 'end' in col_lower or col_lower == time_cols[-1]:
            end_col = col
    
    if start_col is None or end_col is None:
        # Fallback: use first and last time columns
        if len(time_cols) >= 2:
            start_col = time_cols[0]
            end_col = time_cols[-1]
        else:
            return None
    
    try:
        durations = events[end_col].values - events[start_col].values
        durations = durations[~np.isnan(durations)]
        
        if len(durations) == 0:
            return None
        
        return {
            'median': float(np.median(durations)),
            'mean': float(np.mean(durations)),
            'std': float(np.std(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations)),
            'values': durations.tolist()
        }
    except:
        return None
    
#=============================================================================================