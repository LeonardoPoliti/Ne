#=================================================================================================
#                              HDF5 Formatter - NeuroRobCoRe                                          
#-------------------------------------------------------------------------------------------------                
#
#  Saves session data to HDF5 format with hierarchical block structure
#  - Loads all blocks from a session folder
#  - Saves each block with metadata, trials, and neural/behavioral data
#  - Output: id_task_NN_date.h5
#
#=================================================================================================

import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

# data classes
try:
    from _utils.task_data_class import Metadata, TrialsData
    from _utils.eeg_data_class import EEGData
    from _utils.eyes_data_class import EyesData
    from _utils.emg_data_class import EMGData
    from _utils.kinematics_data_class import KinematicsData
except ImportError as e:
    print(f"ERROR: Could not import data classes: {e}")
    print("Make sure all data class files are in _utils directory")

# session parsing 
try:
    from _utils.session_parser import *
except ImportError as e:
    print(f"ERROR: Could not import session parsing utilities: {e}")
    print("Make sure session_parsing.py is in _utils directory")

# configs
VERBOSE = True  


#=================================================================================================
# UTILITY FUNCTIONS
#=================================================================================================

def generate_output_filename(session_folder: Path, task_name: str) -> str:
    """Generate output filename: id_task_date.h5"""
    participant_id, _, date_str = parse_session(session_folder.name)
    
    if participant_id and date_str:
        filename = f"{participant_id}_{task_name}_{date_str}.h5"
    else:
        filename = f"session_{task_name}_{datetime.now().strftime('%Y%m%d')}.h5"
    
    return str(session_folder / filename)

#-------------------------------------------------------------------------------------------------

def save_dict_as_attrs(group: h5py.Group, data: dict):
    """Save dictionary as HDF5 group attributes"""
    for key, value in data.items():
        if value is None:
            value = "None"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif isinstance(value, (np.ndarray, pd.Series)):
            value = value.tolist() if len(value) < 100 else json.dumps(value.tolist())
        
        try:
            group.attrs[key] = value
        except Exception as e:
            if VERBOSE:
                print(f" WARNING: Could not save attribute {key}: {e}")

#-------------------------------------------------------------------------------------------------

def save_dataframe(group: h5py.Group, name: str, df: pd.DataFrame):
    """Save pandas DataFrame to HDF5 group"""
    if df is None or df.empty:
        return
    
    df_group = group.create_group(name)
    df_group.attrs['columns'] = list(df.columns)
    
    for col in df.columns:
        try:
            data = df[col].values
            if data.dtype == object or data.dtype.kind == 'U':
                data = data.astype('S')  # Convert to byte strings for HDF5
            df_group.create_dataset(col, data=data)
        except Exception as e:
            print(f" ERROR: Could not save column {col}: {e}")


#=================================================================================================
# DATA SAVING FUNCTIONS
#=================================================================================================

def save_metadata(block_group: h5py.Group, task_block_folder: Path):
    """Save metadata from task block folder"""
    metadata_file = task_block_folder / "metadata.json"
    
    if not metadata_file.exists():
        if VERBOSE:
            print(f" WARNING: No metadata file found - skipping")
        return
    
    try:
        metadata = Metadata(str(metadata_file), verbose=False).load()
        metadata_group = block_group.create_group('metadata')
        
        for section_name in metadata.get_data_attributes():
            section_data = getattr(metadata, section_name)
            if isinstance(section_data, dict):
                section_group = metadata_group.create_group(section_name)
                save_dict_as_attrs(section_group, section_data)
        
        if VERBOSE:
            print(f"\n ✓ metadata ({len(metadata.get_data_attributes())} sections)")
    
    except Exception as e:
        print(f" ERROR loading metadata: {e}")

#-------------------------------------------------------------------------------------------------

def save_trials(block_group: h5py.Group, task_block_folder: Path):
    """Save trials data from task block folder"""
    trials_file = task_block_folder / "data.csv"
    
    if not trials_file.exists():
        if VERBOSE:
            print(f" WARNING: No trials file found - skipping")
        return
    
    try:
        trials = TrialsData(str(trials_file), verbose=False).load()
        trials_group = block_group.create_group('trials_info')
        
        if trials.events is not None and not trials.events.empty:
            save_dataframe(trials_group, 'events', trials.events)
        
        for attr_name in trials.get_data_attributes():
            if attr_name == 'events':
                continue
            data = getattr(trials, attr_name, None)
            if isinstance(data, np.ndarray) and len(data) > 0:
                trials_group.create_dataset(attr_name, data=data)
            elif VERBOSE:
                print(f" WARNING: Trials attribute - {attr_name} - is empty or not an array - skipping")
        
        if VERBOSE:
            n_attrs = len([a for a in trials.get_data_attributes()])
            print(f" ✓ trials_info ({len(trials)} trials, {n_attrs} attributes)")
    
    except Exception as e:
        print(f" ERROR loading trials info: {e}")

#-------------------------------------------------------------------------------------------------

def save_eeg(block_group: h5py.Group, eeg_file: Optional[Path]):
    """Save EEG data from Unicorn CSV file"""
    if eeg_file is None:
        return
    
    try:
        eeg = EEGData(str(eeg_file), verbose=False).load()
        eeg_group = block_group.create_group('eeg')
        
        if eeg.time is not None:
            eeg_group.create_dataset('timestamps', data=eeg.time)
        
        if eeg.channels is not None and not eeg.channels.empty:
            save_dataframe(eeg_group, 'channels', eeg.channels)
        
        if eeg.valid is not None:
            eeg_group.create_dataset('valid', data=eeg.valid)
        
        if eeg.accelerometer is not None and not eeg.accelerometer.empty:
            save_dataframe(eeg_group, 'accelerometer', eeg.accelerometer)
        
        if eeg.gyroscope is not None and not eeg.gyroscope.empty:
            save_dataframe(eeg_group, 'gyroscope', eeg.gyroscope)
        
        if eeg.events is not None and not eeg.events.empty:
            save_dataframe(eeg_group, 'events', eeg.events)
        
        if eeg.sampling_rate is not None:
            eeg_group.attrs['sampling_rate'] = eeg.sampling_rate
        
        if VERBOSE:
            channels = eeg.channels.shape[1] if eeg.channels is not None else 0
            n_samples = len(eeg.time) if eeg.time is not None else 0
            print(f" ✓ eeg ({channels:,} channels , {n_samples:,} samples)")
    except Exception as e:
        print(f" ERROR loading EEG: {e}")

#-------------------------------------------------------------------------------------------------

def save_eyes(block_group: h5py.Group, eyes_file: Optional[Path]):
    """Save eye tracking data from EyeLink ASC file"""
    if eyes_file is None:
        if VERBOSE:
            print(f" WARNING: No EyeLink file found - skipping")
        return
    
    try:
        eyes = EyesData(str(eyes_file), verbose=False).load()
        eyes_group = block_group.create_group('eyes')
        
        if eyes.time is not None:
            eyes_group.create_dataset('timestamps', data=eyes.time)
        
        if eyes.gaze is not None and not eyes.gaze.empty:
            save_dataframe(eyes_group, 'gaze', eyes.gaze)

        if eyes.pupil is not None and not eyes.pupil.empty:
            save_dataframe(eyes_group, 'pupil', eyes.pupil)
        
        if eyes.fixations is not None and not eyes.fixations.empty:
            save_dataframe(eyes_group, 'fixations', eyes.fixations)
        
        if eyes.saccades is not None and not eyes.saccades.empty:
            save_dataframe(eyes_group, 'saccades', eyes.saccades)
        
        if eyes.blinks is not None and not eyes.blinks.empty:
            save_dataframe(eyes_group, 'blinks', eyes.blinks)
        
        if eyes.events is not None and not eyes.events.empty:
            save_dataframe(eyes_group, 'events', eyes.events)
        
        if eyes.sampling_rate is not None:
            eyes_group.attrs['sampling_rate'] = eyes.sampling_rate
        
        config = {
            'eye_tracked': eyes.eye_tracked,
            'pupil_mode': eyes.pupil_mode,
            'tracking_mode': eyes.tracking_mode,
            'filter_level': eyes.filter_level
        }
        config_group = eyes_group.create_group('config')
        save_dict_as_attrs(config_group, config)
        
        if VERBOSE:
            n_samples = len(eyes.time) if eyes.time is not None else 0
            print(f" ✓ eyes ({eyes.eye_tracked}, {n_samples:,} samples)")
    
    except Exception as e:
        print(f" ERROR loading eyes: {e}")

#-------------------------------------------------------------------------------------------------

def save_emg(block_group: h5py.Group, vicon_file: Optional[Path]):
    """Save EMG data from Vicon C3D file"""
    if vicon_file is None:
        if VERBOSE:
            print(f" WARNING: No Vicon file found - skipping EMG")
        return
    
    try:
        emg = EMGData(str(vicon_file), verbose=False).load()
        emg_group = block_group.create_group('emg')
        
        if emg.time is not None:
            emg_group.create_dataset('timestamps', data=emg.time)
        
        if emg.channels is not None and not emg.channels.empty:
            save_dataframe(emg_group, 'channels', emg.channels)
        
        if emg.events is not None and not emg.events.empty:
            save_dataframe(emg_group, 'events', emg.events)
        
        if emg.sampling_rate is not None:
            emg_group.attrs['sampling_rate'] = emg.sampling_rate
        
        if VERBOSE:
            n_samples = len(emg.time) if emg.time is not None else 0
            n_channels = emg.channels.shape[1] if emg.channels is not None else 0
            print(f" ✓ emg ({n_channels:,} channels, {n_samples:,} samples)")
    
    except Exception as e:  
        if VERBOSE:
            print(f" ERROR loading EMG: {e}")
    
#-------------------------------------------------------------------------------------------------

def save_kinematics(block_group: h5py.Group, vicon_file: Optional[Path]):
    """Save kinematics data from Vicon C3D file"""
    if vicon_file is None:
        if VERBOSE:
            print(f" WARNING: No Vicon file found - skipping kinematics")
        return
    
    try:
        kin = KinematicsData(str(vicon_file), verbose=False).load()
        kin_group = block_group.create_group('kinematics')
        
        if kin.time is not None:
            kin_group.create_dataset('timestamps', data=kin.time)
        
        if kin.markers is not None and not kin.markers.empty:
            save_dataframe(kin_group, 'markers', kin.markers)
        
        if kin.residuals is not None and not kin.residuals.empty:
            save_dataframe(kin_group, 'residuals', kin.residuals)
        
        if kin.events is not None and not kin.events.empty:
            save_dataframe(kin_group, 'events', kin.events)
        
        if kin.sampling_rate is not None:
            kin_group.attrs['sampling_rate'] = kin.sampling_rate
        
        if VERBOSE:
            n_samples = len(kin.time) if kin.time is not None else 0
            n_markers = kin.markers.shape[1]//3 if kin.markers is not None else 0
            print(f" ✓ kinematics ({n_markers:,} markers, {n_samples:,} samples)")
    
    except Exception as e:
        if VERBOSE:
            print(f" ERROR loading kinematics: {e}")

#-------------------------------------------------------------------------------------------------

def block_to_h5(h5file: h5py.File, block_num: int, task_block_folder: Path,
                  session_folder: Path, participant_id: str):
    """Process a single block and save to HDF5"""
    block_name = f"block_{block_num:02d}"  
    block_group = h5file.create_group(block_name)
    
    # Find data files for this block
    eyelink_file = find_eyelink_file(session_folder, block_num, participant_id)
    unicorn_file = find_unicorn_file(session_folder, block_num)
    vicon_file = find_vicon_file(session_folder, block_num)
    
    # Save all data types
    save_metadata(block_group, task_block_folder)
    save_trials(block_group, task_block_folder)
    save_eeg(block_group, unicorn_file)
    save_eyes(block_group, eyelink_file)
    save_emg(block_group, vicon_file)
    save_kinematics(block_group, vicon_file)

    if VERBOSE:
        print(f"\nBlock {block_num}: {task_block_folder.name} saved")

#-------------------------------------------------------------------------------------------------

def inspect_h5(h5_path: str):
    """Print structure of HDF5 file"""
    print(f"\n{'='*50}")
    print(f"HDF5 Structure: {Path(h5_path).name}")
    print(f"{'='*50}\n")
    
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/")
            if obj.attrs:
                for key in list(obj.attrs.keys())[:3]:
                    print(f"{indent}  @{key}: {obj.attrs[key]}")
                if len(obj.attrs) > 3:
                    print(f"{indent}  ... ({len(obj.attrs)} attrs)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} [{obj.shape}, {obj.dtype}]")
    
    with h5py.File(h5_path, 'r') as f:
        f.visititems(print_structure)
    
    print(f"\n{'='*50}\n")


#=================================================================================================
# MAIN FUNCTION
#=================================================================================================

def format_session(session_folder: str, output_path: Optional[str] = None, verbose: bool = True) -> Path:
    """
    Format entire session into HDF5 file.
    
    Args:
        session_folder: Path to session folder (data/task_name/id_NN_date/)
        output_path: auto-generated if None
        verbose: Print progress information
    
    Returns: 
        Path to created HDF5 file
    """
    global VERBOSE
    VERBOSE = verbose
    
    session_folder = Path(session_folder)
    
    if not session_folder.exists():
        raise FileNotFoundError(f"Session folder not found: {session_folder}")
    if not session_folder.is_dir():
        raise ValueError(f"Path is not a directory: {session_folder}")
    
    # Parse session info
    participant_id, session_num, date_str = parse_session(session_folder.name)
    task_name = session_folder.parent.name  # Parent folder is task name
    
    if participant_id is None:
        participant_id = "unknown"
        if VERBOSE:
            print(f" WARNING: Could not parse session folder name: {session_folder.name}")
    
    # Find task blocks
    task_blocks = find_task_blocks(session_folder)
    n_blocks = len(task_blocks)
    
    if VERBOSE:
        print(f"\nSession: {participant_id}_{session_num}_{date_str} | Task: {task_name} | Blocks: {n_blocks}")
    
    # Generate output filename
    if output_path is None:
        output_path = generate_output_filename(session_folder, task_name) ## specify output folder 
    
    output_path = Path(output_path)
    
    if VERBOSE:
        print(f"Creating: {output_path.name}")
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5file:
        for block_num, task_block_folder in enumerate(task_blocks, start=1):
            block_to_h5(h5file, block_num, task_block_folder, session_folder, participant_id)
    
    if VERBOSE:
        print(f"\n✓ Created: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


#=================================================================================================
# COMMAND LINE INTERFACE
#=================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert session data to HDF5 format')
    parser.add_argument('session_folder', type=str, default=None, 
                        help='Path to session folder (data/task_name/id_NN_date/)')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Output HDF5 file path (auto-generated if not provided)')
    parser.add_argument('-q', '--quiet', action='store_true', 
                        help='Suppress progress output')
    parser.add_argument('-i', '--inspect', action='store_true',
                        help='Inspect HDF5 file structure after creation')
    
    args = parser.parse_args()

    if args.session_folder is None:
        session_folder = r"c:\Users\leonardo_politi\Downloads\data\reaching\frli_1112"
    else:
        session_folder = args.session_folder
    
    output_path = format_session(
        session_folder=session_folder,
        output_path=args.output,
        verbose=not args.quiet
    )
    
    if args.inspect:
        inspect_h5(str(output_path))
