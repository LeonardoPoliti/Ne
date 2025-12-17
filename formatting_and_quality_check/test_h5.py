#=================================================================================================
#                              HDF5 Test Script - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Verify HDF5 file structure and data integrity
#
#=================================================================================================

import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def load_dataframe(group: h5py.Group) -> pd.DataFrame:
    """Reconstruct DataFrame from HDF5 group"""
    cols = list(group.attrs['columns'])
    data = {}
    for col in cols:
        values = group[col][:]
        # Decode byte strings back to regular strings
        if values.dtype.kind == 'S':
            values = np.array([v.decode() if isinstance(v, bytes) else v for v in values])
        data[col] = values
    return pd.DataFrame(data)


def test_h5(h5_path: str, verbose: bool = True):
    """
    Test HDF5 file structure and data integrity.
    
    Args:
        h5_path: Path to HDF5 file
        verbose: Print detailed info
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        print(f"ERROR: File not found: {h5_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Testing: {h5_path.name}")
    print(f"Size: {h5_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")
    
    errors = []
    warnings = []
    
    with h5py.File(h5_path, 'r') as f:
        
        # Find blocks
        blocks = [k for k in f.keys() if k.startswith('block_')]
        print(f"Blocks found: {len(blocks)}")
        
        for block_name in sorted(blocks):
            print(f"\n  {block_name}/")
            block = f[block_name]
            
            # Check metadata
            if 'metadata' in block:
                sections = list(block['metadata'].keys())
                print(f"    ✓ metadata ({len(sections)} sections: {', '.join(sections)})")
            else:
                warnings.append(f"{block_name}: No metadata")
                print(f"    ⚠ metadata missing")
            
            # Check trials_info
            if 'trials_info' in block:
                trials = block['trials_info']
                n_attrs = len([k for k in trials.keys() if k != 'events'])
                if 'events' in trials:
                    events_df = load_dataframe(trials['events'])
                    print(f"    ✓ trials_info ({n_attrs} attributes, events: {len(events_df)} rows)")
                    if verbose:
                        print(f"        events columns: {list(events_df.columns)}")
                else:
                    print(f"    ✓ trials_info ({n_attrs} attributes, no events)")
            else:
                warnings.append(f"{block_name}: No trials_info")
                print(f"    ⚠ trials_info missing")
            
            # Check EEG
            if 'eeg' in block:
                eeg = block['eeg']
                n_samples = len(eeg['timestamps'][:]) if 'timestamps' in eeg else 0
                sr = eeg.attrs.get('sampling_rate', 'N/A')
                
                channels_df = load_dataframe(eeg['channels']) if 'channels' in eeg else None
                n_ch = len(channels_df.columns) if channels_df is not None else 0
                
                print(f"    ✓ eeg ({n_ch} channels, {n_samples:,} samples, {sr} Hz)")
                if verbose and channels_df is not None:
                    print(f"        channels: {list(channels_df.columns)}")
            
            # Check eyes
            if 'eyes' in block:
                eyes = block['eyes']
                n_samples = len(eyes['timestamps'][:]) if 'timestamps' in eyes else 0
                sr = eyes.attrs.get('sampling_rate', 'N/A')
                
                parts = []
                for key in ['gaze', 'fixations', 'saccades', 'blinks']:
                    if key in eyes:
                        df = load_dataframe(eyes[key])
                        parts.append(f"{key}: {len(df)}")
                
                print(f"    ✓ eyes ({n_samples:,} samples, {sr} Hz)")
                if verbose and parts:
                    print(f"        {', '.join(parts)}")
                
                # Test string decoding
                if 'fixations' in eyes:
                    fix_df = load_dataframe(eyes['fixations'])
                    if 'eye' in fix_df.columns:
                        unique_eyes = fix_df['eye'].unique()
                        print(f"        eye values: {unique_eyes}")
            
            # Check EMG
            if 'emg' in block:
                emg = block['emg']
                n_samples = len(emg['timestamps'][:]) if 'timestamps' in emg else 0
                sr = emg.attrs.get('sampling_rate', 'N/A')
                
                channels_df = load_dataframe(emg['channels']) if 'channels' in emg else None
                n_ch = len(channels_df.columns) if channels_df is not None else 0
                
                print(f"    ✓ emg ({n_ch} channels, {n_samples:,} samples, {sr} Hz)")
                
                # Test events string decoding
                if 'events' in emg:
                    events_df = load_dataframe(emg['events'])
                    if 'event_type' in events_df.columns:
                        unique_types = events_df['event_type'].unique()
                        print(f"        event_type values: {unique_types}")
            
            # Check kinematics
            if 'kinematics' in block:
                kin = block['kinematics']
                n_samples = len(kin['timestamps'][:]) if 'timestamps' in kin else 0
                sr = kin.attrs.get('sampling_rate', 'N/A')
                
                markers_df = load_dataframe(kin['markers']) if 'markers' in kin else None
                n_markers = len(markers_df.columns) // 3 if markers_df is not None else 0
                
                print(f"    ✓ kinematics ({n_markers} markers, {n_samples:,} samples, {sr} Hz)")
    
    # Summary
    print(f"\n{'='*70}")
    if errors:
        print(f"ERRORS: {len(errors)}")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print("✓ All checks passed")
    print(f"{'='*70}\n")
    
    return len(errors) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HDF5 file integrity')
    parser.add_argument('h5_file', type=str, help='Path to HDF5 file')
    parser.add_argument('-q', '--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    test_h5(args.h5_file, verbose=not args.quiet)