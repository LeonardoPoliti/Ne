#=====================================================================================================
#                              Data Quality Check - NeuroRobCoRe
#=====================================================================================================
#
#  Run quality checks on a recording session (composed of multiple blocks) and generate a HTML report.
#  
#  Usage:
#      python run_quality_check.py <session_folder> [--output OUTPUT_FILE]
#
#=====================================================================================================

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import session parsing utilities 
from _utils.session_parser import (
    parse_session, 
    find_task_blocks, 
    find_eyelink_file, 
    find_unicorn_file, 
    find_vicon_file
)
from _utils.task_data_class import Metadata, TrialsData
from _utils.eeg_data_class import EEGData
from _utils.eyes_data_class import EyesData
from _utils.kinematics_data_class import KinematicsData
from _utils.emg_data_class import EMGData

# Import quality check functions
from _utils.qc_task import check_task_quality
from _utils.qc_eeg import check_eeg_quality
from _utils.qc_eyes import check_eyes_quality
from _utils.qc_kinematics import check_kinematics_quality
from _utils.qc_emg import check_emg_quality
from _utils.qc_crossmodal_align import check_crossmodal_alignment
from _utils.qc_report_generator import generate_session_report

#--------------------------------------------------------------------------------------

def run_quality_check(session_folder: str, 
                      output_path: Optional[str] = None,
                      verbose: bool = True) -> str:
    """
    Run quality check pipeline on all blocks in a session and generate a single HTML report.
    Args:
        session_folder: Path to session folder 
        output_path: Path for output HTML file. If None, saves to session_folder/qc_report.html
        verbose: Print progress information
    Returns:
        Path to generated HTML report
    """
    session_folder = Path(session_folder)
    
    # Parse session info
    participant_id, session_num, date_str = parse_session(session_folder.name)
    
    if participant_id is None:
        raise ValueError(f"Could not parse session folder name: {session_folder.name}")
    
    session_info = {
        'participant_id': participant_id,
        'session_num': session_num,
        'date': date_str,
        'folder': str(session_folder)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Quality Check Pipeline")
        print(f"{'='*60}")
        print(f"Participant: {participant_id}")
        print(f"Session: {session_num}")
        print(f"Date: {date_str}")
        print(f"{'='*60}\n")
    
    # Find all blocks
    block_folders = find_task_blocks(session_folder)
    
    if verbose:
        print(f"Found {len(block_folders)} blocks\n")
    
    # Process each block
    blocks_results = []
    
    for block_num, block_folder in enumerate(block_folders, 1):
        if verbose:
            print(f"{'─'*40}")
            print(f"Block {block_num}: {block_folder.name}")
            print(f"{'─'*40}")
        
        block_results = process_block(
            session_folder=session_folder,
            block_folder=block_folder,
            block_num=block_num,
            participant_id=participant_id,
            verbose=verbose
        )
        
        blocks_results.append(block_results)
    
    # Generate single report for entire session
    if output_path is None:
        output_path = session_folder / f"qc_report_{participant_id}_{session_num:02d}.html"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Generating report...")
    
    report_path = generate_session_report(
        output_path=str(output_path),
        session_info=session_info,
        blocks_results=blocks_results
    )
    
    if verbose:
        print(f"Report saved: {report_path}")
        print(f"{'='*60}\n")
    
    return report_path

#--------------------------------------------------------------------------------------

def process_block(session_folder: Path,
                  block_folder: Path,
                  block_num: int,
                  participant_id: str,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Process a single block and return quality check results.
    
    Returns:
        Dictionary with results for each data type
    """
    results = {
        'block_info': {
            'block_num': block_num,
            'block_name': block_folder.name
        },
        'task': None,
        'eeg': None,
        'eyes': None,
        'kinematics': None,
        'emg': None,
        'crossmodal': None
    }
    
    # Data objects for cross-modal
    task_data = None
    eeg_data = None
    eyes_data = None
    kinematics_data = None
    emg_data = None
    
    # Task Data
    try:
        metadata_file = block_folder / 'metadata.json'
        data_file = block_folder / 'data.csv'
        
        if metadata_file.exists() and data_file.exists():
            metadata = Metadata(str(metadata_file), verbose=False).load()
            trials = TrialsData(str(data_file), verbose=False).load()
            results['task'] = check_task_quality(metadata, trials)
            task_data = trials
            if verbose:
                print(f"  ✓ Task: {results['task']['trials']['n_trials']} trials")
    except Exception as e:
        if verbose:
            print(f"  ✗ Task: {e}")
    
    # EEG Data
    try:
        eeg_file = find_unicorn_file(session_folder, block_num)
        if eeg_file and eeg_file.exists():
            eeg = EEGData(str(eeg_file), verbose=False).load()
            results['eeg'] = check_eeg_quality(eeg, expected_fs=250.0)
            eeg_data = eeg
            if verbose:
                print(f"  ✓ EEG: {results['eeg']['metadata']['duration_s']:.1f}s, {results['eeg']['metadata']['n_channels']} ch")
    except Exception as e:
        if verbose:
            print(f"  ✗ EEG: {e}")
    
    # Eyes Data
    try:
        eyes_file = find_eyelink_file(session_folder, block_num, participant_id)
        if eyes_file and eyes_file.exists():
            eyes = EyesData(str(eyes_file), verbose=False).load()
            results['eyes'] = check_eyes_quality(eyes, expected_fs=1000.0)
            eyes_data = eyes
            if verbose:
                print(f"  ✓ Eyes: {results['eyes']['metadata']['duration_s']:.1f}s, {results['eyes']['metadata']['n_fixations']} fix")
    except Exception as e:
        if verbose:
            print(f"  ✗ Eyes: {e}")
    
    # Kinematics Data
    try:
        vicon_file = find_vicon_file(session_folder, block_num)
        if vicon_file and vicon_file.exists():
            kin = KinematicsData(str(vicon_file), verbose=False).load()
            results['kinematics'] = check_kinematics_quality(kin)
            kinematics_data = kin
            if verbose:
                print(f"  ✓ Kinematics: {results['kinematics']['metadata']['duration_s']:.1f}s, {results['kinematics']['metadata']['n_markers']} markers")
    except Exception as e:
        if verbose:
            print(f"  ✗ Kinematics: {e}")
    
    # Emg Data
    try:
        vicon_file = find_vicon_file(session_folder, block_num)
        if vicon_file and vicon_file.exists():
            emg = EMGData(str(vicon_file), verbose=False).load()
            results['emg'] = check_emg_quality(emg)
            emg_data = emg
            if verbose:
                print(f"  ✓ EMG: {results['emg']['metadata']['duration_s']:.1f}s, {results['emg']['metadata']['n_channels']} ch")
    except Exception as e:
        if verbose:
            print(f"  ✗ EMG: {e}")
    
    # Cross-Modal Alignment
    available = sum([d is not None for d in [task_data, eeg_data, eyes_data, kinematics_data, emg_data]])
    
    if available >= 2:
        try:
            results['crossmodal'] = check_crossmodal_alignment(
                task_data=task_data,
                eeg_data=eeg_data,
                eyes_data=eyes_data,
                kinematics_data=kinematics_data,
                emg_data=emg_data
            )
            if verbose:
                print(f"  ✓ Cross-modal alignment checked")
        except Exception as e:
            if verbose:
                print(f"  ✗ Cross-modal: {e}")
    
    return results

#==============================================================================================
#                                           MAIN
#==============================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run quality check pipeline on a session')
    parser.add_argument('session_folder', type=str, help='Path to session folder')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    run_quality_check(
        session_folder=args.session_folder,
        output_path=args.output,
        verbose=not args.quiet
    )

#==============================================================================================