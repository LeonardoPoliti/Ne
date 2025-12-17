#==============================================================================================================
#                                   SESSION PARSING - NeuroRobCoRe 
# -------------------------------------------------------------------------------------------------------------
# Functions to parse session folder names and find data files within session folders. 
# Specific to folder and file naming conventions used in the project - ask leonardo.politi3@unibo.it if needed
#==============================================================================================================

import re
from pathlib import Path
from typing import Tuple, Optional, List

#==============================================================================================================

def parse_session(folder_name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse session folder name: id_NN_date
    Pattern: any ID (1+ chars), 2-digit number (exactly 2, session number), date (any format, will be used for h5 filename)

    Returns: (participant_id, session_num, date_str) or (None, None, None) if no match
    """
    pattern = r'^(.+?)_(\d{2})_(.+)$'
    match = re.match(pattern, folder_name)
    
    if match:
        participant_id = match.group(1)
        session_num = int(match.group(2))
        date_str = match.group(3)
        return participant_id, session_num, date_str
    
    return None, None, None

#-------------------------------------------------------------------------------------------------

def find_task_blocks(session_folder: Path) -> List[Path]:
    """
    Find all task block folders in session/task/ directory.
    Task blocks are subfolders named: id_NN_date_time
    Returns: sorted list of block folders.
    """
    task_folder = session_folder / 'task'
    
    if not task_folder.exists():
        raise ValueError(f"No 'task' folder found in {session_folder}")
    
    # Find all subfolders (each is a block)
    block_folders = [f for f in sorted(task_folder.iterdir()) if f.is_dir()]
    
    if not block_folders:
        raise ValueError(f"No block folders found in {task_folder}")
    
    return block_folders

#-------------------------------------------------------------------------------------------------

def find_eyelink_file(session_folder: Path, block_num: int, participant_id: str) -> Optional[Path]:
    """
    Find EyeLink .asc file for a specific block.
    Filename format: IDNNMMDD.asc (8 chars: 2-letter ID prefix, 2-digit block number, 2-digit month, 2-digit day)
    Returns: Path to .asc file or None if not found.
    """
    eyelink_folder = session_folder / 'eyelink'
    
    if not eyelink_folder.exists():
        return None
    
    # Get 2-letter prefix (first, third) from 4-letter participant ID
    id_prefix = participant_id[0] + participant_id[2]
    
    # Look for matching files
    pattern = f"{id_prefix}{block_num:02d}*.asc"
    matches = list(eyelink_folder.glob(pattern))
    
    if matches:
        return matches[0]
    
    # Fallback: try any .asc file with block number
    for asc_file in eyelink_folder.glob("*.asc"):
        if f"{block_num:02d}" in asc_file.stem:
            return asc_file
    
    return None

#-------------------------------------------------------------------------------------------------

def find_unicorn_file(session_folder: Path, block_num: int) -> Optional[Path]:
    """
    Find Raw Unicorn EEG .csv file for a specific block.
    Filename format: id_task_NN_raw_date_time.csv (NN = block number, "raw" = raw data)
    Returns: Path to .csv file or None if not found. Prefers "_raw_" files, else any file with block number.
    """
    unicorn_folder = session_folder / 'unicorn'
    
    if not unicorn_folder.exists():
        return None
    
    # Look for raw file with block number
    for csv_file in unicorn_folder.glob("*_raw_*.csv"):
        # Extract block number from filename
        parts = csv_file.stem.split('_')
        if int(parts[2]) == block_num:
            return csv_file
        
    # Fallback: try any .csv file with block number
    for csv_file in unicorn_folder.glob("*.csv"):
        if f"_{block_num:02d}_" in csv_file.stem or f"_{block_num}_" in csv_file.stem:
            return csv_file                                 
    
    return None

#-------------------------------------------------------------------------------------------------

def find_vicon_file(session_folder: Path, block_num: int) -> Optional[Path]:
    """
    Find Vicon .c3d file for a specific block.
    Filename format: id_date_NN.c3d (NN = block number)
    Returns: Path to .c3d file or None if not found.
    """
    vicon_folder = session_folder / 'vicon'
    
    if not vicon_folder.exists():
        return None
    
    # Look for .c3d file with block number at end
    for c3d_file in vicon_folder.glob("*.c3d"):
        # Check if filename ends with block number
        stem = c3d_file.stem
        if stem.endswith(f"_{block_num:02d}") or stem.endswith(f"_{block_num}"):
            return c3d_file
    
    return None

#==============================================================================================================
