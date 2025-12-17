#=================================================================================================
#                    Quality Check Report Generator - NeuroRobCoRe
#=================================================================================================

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

#--------------------------------------------------------------------------------------

def _safe_get(data, *keys, default=None):
    """Safely traverse nested dictionary/object structure."""
    try:
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key, default)
            else:
                result = getattr(result, key, default)
            if result is None:
                return default
        return result
    except (KeyError, AttributeError, TypeError):
        return default

#--------------------------------------------------------------------------------------

def generate_session_report(output_path: str,
                            session_info: Dict[str, Any],
                            blocks_results: List[Dict[str, Any]]) -> str:
    """
    Generate comprehensive HTML quality check report for entire session.
    
    Args:
        output_path: Path to save HTML report
        session_info: Dictionary with session information
        blocks_results: List of dictionaries, each containing results for one block
    
    Returns:
        Path to generated HTML file
    """
    html = _build_session_report(session_info, blocks_results)
    output_path = Path(output_path)
    output_path.write_text(html, encoding='utf-8')
    return str(output_path)

#--------------------------------------------------------------------------------------

def _build_session_report(session_info: Dict, blocks_results: List[Dict]) -> str:
    """Build complete HTML report with improved layout and all visualizations."""
    
    # Build navigation for blocks
    block_nav = ""
    for i, block in enumerate(blocks_results):
        block_nav += f'<a href="#block-{i+1}" class="nav-item">Block {i+1}</a>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>QC Report - {session_info.get('participant_id', 'Unknown')}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
            margin: 0; 
            background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
            color: #212529;
        }}
        
        /* Header */
        .header {{ 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
            color: white; 
            padding: 25px; 
            position: sticky; 
            top: 0; 
            z-index: 100;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ 
            margin: 0 0 15px 0; 
            font-size: 2em;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        .session-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }}
        .session-info-item {{
            display: flex;
            flex-direction: column;
        }}
        .session-info-label {{
            font-size: 0.85em;
            opacity: 0.8;
            margin-bottom: 3px;
        }}
        .session-info-value {{
            font-size: 1.2em;
            font-weight: 500;
        }}
        
        /* Navigation */
        .nav {{ 
            display: flex; 
            gap: 10px; 
            flex-wrap: wrap; 
        }}
        .nav-item {{ 
            color: white; 
            text-decoration: none; 
            padding: 8px 20px; 
            background: rgba(255,255,255,0.15); 
            border-radius: 20px;
            transition: all 0.3s;
            font-weight: 500;
        }}
        .nav-item:hover {{ 
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }}
        
        /* Container */
        .container {{ 
            max-width: 1600px; 
            margin: 0 auto; 
            padding: 30px 20px; 
        }}
        
        /* Typography */
        h2 {{ 
            color: #1e3c72; 
            border-bottom: 3px solid #2a5298; 
            padding-bottom: 12px; 
            margin: 40px 0 20px 0;
            font-size: 1.8em;
            font-weight: 400;
        }}
        h3 {{ 
            color: #2a5298; 
            margin: 25px 0 15px 0;
            font-size: 1.4em;
            font-weight: 500;
        }}
        h4 {{ 
            color: #495057; 
            margin: 20px 0 10px 0;
            font-size: 1.1em;
            font-weight: 600;
        }}
        h5 {{ 
            color: #6c757d; 
            margin: 15px 0 8px 0;
            font-size: 1em;
            font-weight: 600;
        }}
        
        /* Sections */
        .section {{ 
            background: white; 
            padding: 25px; 
            margin: 20px 0; 
            border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }}
        
        /* Block sections with colored left border */
        .block-container {{
            margin: 40px 0;
            padding: 25px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 6px solid #2a5298;
        }}
        .block-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
        }}
        .block-title {{
            font-size: 1.6em;
            color: #1e3c72;
            font-weight: 500;
        }}
        .block-status {{
            display: flex;
            gap: 8px;
        }}
        .status-badge {{
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .status-ok {{ 
            background: #d4edda; 
            color: #155724; 
        }}
        .status-missing {{ 
            background: #f8d7da; 
            color: #721c24; 
        }}
        
        /* Modality sections within blocks */
        .modality-section {{
            margin: 25px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #6c757d;
        }}
        .modality-section.task {{ border-left-color: #28a745; }}
        .modality-section.eeg {{ border-left-color: #007bff; }}
        .modality-section.eyes {{ border-left-color: #17a2b8; }}
        .modality-section.kinematics {{ border-left-color: #ffc107; }}
        .modality-section.emg {{ border-left-color: #dc3545; }}
        .modality-section.crossmodal {{ border-left-color: #6f42c1; }}
        
        .modality-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .modality-icon {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .modality-icon.task {{ background: #28a745; }}
        .modality-icon.eeg {{ background: #007bff; }}
        .modality-icon.eyes {{ background: #17a2b8; }}
        .modality-icon.kinematics {{ background: #ffc107; }}
        .modality-icon.emg {{ background: #dc3545; }}
        .modality-icon.crossmodal {{ background: #6f42c1; }}
        
        /* Metadata grid */
        .metadata-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
            gap: 12px;
            margin: 15px 0;
        }}
        .metadata-item {{ 
            background: white; 
            padding: 12px; 
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        .metadata-label {{ 
            font-weight: 600; 
            color: #6c757d; 
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metadata-value {{ 
            color: #212529; 
            font-size: 1.15em; 
            margin-top: 5px;
            font-weight: 500;
        }}
        
        /* Tables */
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0; 
            font-size: 0.9em;
            background: white;
        }}
        th, td {{ 
            padding: 10px 12px; 
            text-align: left; 
            border: 1px solid #dee2e6; 
        }}
        th {{ 
            background: #2a5298; 
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:nth-child(even) {{ 
            background: #f8f9fa; 
        }}
        tr:hover {{
            background: #e9ecef;
        }}
        
        /* Status indicators */
        .warning {{ 
            color: #dc3545; 
            font-weight: bold; 
        }}
        .ok {{ 
            color: #28a745;
            font-weight: 600;
        }}
        .alert {{
            color: #fd7e14;
            font-weight: 600;
        }}
        
        /* Status boxes */
        .status-box {{
            padding: 12px 15px;
            border-radius: 6px;
            margin: 10px 0;
            font-weight: 500;
        }}
        .status-box.success {{
            background: #d4edda;
            color: #155724;
            border-left: 4px solid #28a745;
        }}
        .status-box.warning {{
            background: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
        }}
        .status-box.error {{
            background: #f8d7da;
            color: #721c24;
            border-left: 4px solid #dc3545;
        }}
        .status-box.info {{
            background: #d1ecf1;
            color: #0c5460;
            border-left: 4px solid #17a2b8;
        }}
        
        /* Plots */
        .plot {{ 
            width: 100%; 
            height: 350px; 
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .plot-small {{ 
            height: 280px; 
        }}
        .plot-thin {{
            height: 200px;
            max-width: 600px;
        }}
        .plot-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .plot-container-thin {{
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            max-width: 700px;
        }}
        .plot-container-centered {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 900px;
        }}
        .plots-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .plots-row .plot-container-thin {{
            max-width: 100%;
        }}
        
        /* Metric cards */
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            text-align: center;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #6c757d;
            font-weight: 600;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #212529;
        }}
        .metric-unit {{
            font-size: 0.6em;
            color: #6c757d;
            font-weight: 400;
        }}
        
        /* Collapsible sections */
        details {{ 
            margin: 15px 0;
            background: white;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        summary {{ 
            cursor: pointer; 
            color: #2a5298; 
            font-weight: 600;
            padding: 5px;
            user-select: none;
        }}
        summary:hover {{
            color: #1e3c72;
        }}
        details[open] summary {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        /* Code/pre blocks */
        pre {{ 
            background: #2c3e50; 
            color: #ecf0f1; 
            padding: 15px; 
            border-radius: 6px; 
            overflow-x: auto; 
            max-height: 400px;
            font-size: 0.85em;
            line-height: 1.5;
        }}
        
        /* Timestamp */
        .timestamp {{ 
            color: #6c757d; 
            font-size: 0.9em;
            text-align: right;
            margin-bottom: 20px;
        }}
        
        /* Summary table specific */
        .summary-table {{ 
            font-size: 0.95em; 
        }}
        .summary-table th {{ 
            background: #1e3c72; 
        }}
        .summary-table a {{
            color: #2a5298;
            text-decoration: none;
            font-weight: 600;
        }}
        .summary-table a:hover {{
            color: #1e3c72;
            text-decoration: underline;
        }}
        
        /* Print styles */
        @media print {{
            .header {{ position: relative; }}
            .nav {{ display: none; }}
            .block-container {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
<div class="header">
    <h1>📊 Quality Check Report</h1>
    <div class="session-info">
        <div class="session-info-item">
            <div class="session-info-label">Participant</div>
            <div class="session-info-value">{session_info.get('participant_id', 'N/A')}</div>
        </div>
        <div class="session-info-item">
            <div class="session-info-label">Session</div>
            <div class="session-info-value">{session_info.get('session_num', 'N/A')}</div>
        </div>
        <div class="session-info-item">
            <div class="session-info-label">Date</div>
            <div class="session-info-value">{session_info.get('date', 'N/A')}</div>
        </div>
        <div class="session-info-item">
            <div class="session-info-label">Blocks</div>
            <div class="session-info-value">{len(blocks_results)}</div>
        </div>
    </div>
    <div class="nav">
        <a href="#summary" class="nav-item">📋 Summary</a>
        {block_nav}
    </div>
</div>
<div class="container">
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    html += _build_session_summary(blocks_results)
    
    for i, block in enumerate(blocks_results):
        html += _build_block_section(i + 1, block)
    
    html += "</div></body></html>"
    return html

#--------------------------------------------------------------------------------------

def _build_session_summary(blocks_results: List[Dict]) -> str:
    """Build comprehensive session summary section."""
    html = '<h2 id="summary">📋 Session Summary</h2><div class="section">'
    html += '<table class="summary-table"><tr><th>Block</th><th>Task</th><th>EEG</th><th>Eyes</th><th>Kinematics</th><th>EMG</th><th>Duration</th><th>Trials</th></tr>'
    
    for i, block in enumerate(blocks_results):
        block_num = i + 1
        task_ok = '✓' if block.get('task') else '✗'
        eeg_ok = '✓' if block.get('eeg') else '✗'
        eyes_ok = '✓' if block.get('eyes') else '✗'
        kin_ok = '✓' if block.get('kinematics') else '✗'
        emg_ok = '✓' if block.get('emg') else '✗'
        
        duration = 'N/A'
        for mod in ['eeg', 'eyes', 'kinematics', 'emg']:
            if block.get(mod, {}).get('metadata', {}).get('duration_s'):
                duration = f"{block[mod]['metadata']['duration_s']:.1f}s"
                break
        
        trials = block.get('task', {}).get('trials', {}).get('n_trials', 'N/A')
        
        html += f'<tr><td><a href="#block-{block_num}">Block {block_num}</a></td>'
        html += f'<td class="{"ok" if task_ok=="✓" else "warning"}">{task_ok}</td>'
        html += f'<td class="{"ok" if eeg_ok=="✓" else "warning"}">{eeg_ok}</td>'
        html += f'<td class="{"ok" if eyes_ok=="✓" else "warning"}">{eyes_ok}</td>'
        html += f'<td class="{"ok" if kin_ok=="✓" else "warning"}">{kin_ok}</td>'
        html += f'<td class="{"ok" if emg_ok=="✓" else "warning"}">{emg_ok}</td>'
        html += f'<td>{duration}</td><td>{trials}</td></tr>'
    
    html += '</table></div>'
    return html

#--------------------------------------------------------------------------------------

def _build_block_section(block_num: int, block: Dict) -> str:
    """Build comprehensive section for a single block."""
    block_info = block.get('block_info', {})
    block_name = block_info.get('block_name', f'Block {block_num}')
    
    # Count available modalities
    modalities = []
    if block.get('task'): modalities.append('Task')
    if block.get('eeg'): modalities.append('EEG')
    if block.get('eyes'): modalities.append('Eyes')
    if block.get('kinematics'): modalities.append('Kinematics')
    if block.get('emg'): modalities.append('EMG')
    
    html = f'''<div class="block-container" id="block-{block_num}">
    <div class="block-header">
        <div class="block-title">Block {block_num}: {block_name}</div>
        <div class="block-status">'''
    
    # Status badges
    for mod in ['task', 'eeg', 'eyes', 'kinematics', 'emg']:
        status_class = 'status-ok' if block.get(mod) else 'status-missing'
        status_text = '✓' if block.get(mod) else '✗'
        mod_name = mod.capitalize()
        html += f'<span class="status-badge {status_class}">{mod_name} {status_text}</span>'
    
    html += '</div></div>'
    
    # Extract screen resolution from task metadata for eyes section
    screen_bounds = None
    if block.get('task') and block['task'].get('metadata'):
        task_meta = block['task']['metadata']
        content = task_meta.get('content', {})
        # Look in configuration section for screen_resolution
        config = content.get('configuration', {})
        if config.get('screen_resolution'):
            try:
                res_str = config['screen_resolution']
                if 'x' in str(res_str):
                    w, h = res_str.split('x')
                    screen_bounds = (0, int(w), 0, int(h))
            except:
                pass
    
    # Build each modality section
    if block.get('task'):
        html += _build_task_section(block['task'], block_num)
    
    if block.get('eeg'):
        html += _build_eeg_section(block['eeg'], block_num)
    
    if block.get('eyes'):
        html += _build_eyes_section(block['eyes'], block_num, screen_bounds=screen_bounds)
    
    if block.get('kinematics'):
        html += _build_kinematics_section(block['kinematics'], block_num)
    
    if block.get('emg'):
        html += _build_emg_section(block['emg'], block_num)
    
    if block.get('crossmodal'):
        html += _build_crossmodal_section(block['crossmodal'], block_num)
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------

def _build_task_section(results: Dict, block_num: int) -> str:
    """Build task data section with all visualizations."""
    html = '''<div class="modality-section task">
    <div class="modality-header">
        <div class="modality-icon task"></div>
        <h3>Task Data</h3>
    </div>'''
    
    # Metadata - SHOW ACTUAL CONTENT
    if results and 'metadata' in results and results['metadata']:
        meta = results['metadata']
        sections = meta.get('sections', [])
        content = meta.get('content', {})
        
        html += '<h4>📋 Session Metadata</h4>'
        
        if content:
            # Display content for each section
            for section in sections:
                section_data = content.get(section, {})
                if section_data:
                    html += f'<details><summary><strong>{section}</strong></summary>'
                    html += '<div class="metadata-grid">'
                    
                    if isinstance(section_data, dict):
                        for key, value in section_data.items():
                            # Format value nicely
                            if isinstance(value, float):
                                display_val = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                            elif isinstance(value, (list, tuple)):
                                display_val = ', '.join(str(v) for v in value[:10])
                                if len(value) > 10:
                                    display_val += f"... (+{len(value)-10} more)"
                            else:
                                display_val = str(value)
                            
                            html += f'''<div class="metadata-item">
                                <div class="metadata-label">{key}</div>
                                <div class="metadata-value" style="font-size: 0.9em;">{display_val}</div>
                            </div>'''
                    else:
                        html += f'<p>{section_data}</p>'
                    
                    html += '</div></details>'
        else:
            # Fallback to simple section list if no content
            html += '<div class="metric-cards">'
            for section in sections:
                html += f'''<div class="metric-card">
                    <div class="metric-label">{section}</div>
                    <div class="metric-value">✓</div>
                </div>'''
            html += '</div>'
    
    # Trials info
    trials = None
    if results and 'trials' in results:
        trials = results.get('trials')
    
    if trials:
        n_trials = trials.get('n_trials', 'N/A')
        is_seq = trials.get('is_sequential', None)
        
        html += '<h4>Trial Information</h4>'
        html += '<div class="metric-cards">'
        html += f'''<div class="metric-card">
            <div class="metric-label">Total Trials</div>
            <div class="metric-value">{n_trials}</div>
        </div>'''
        
        if is_seq is not None:
            seq_status = 'ok' if is_seq else 'warning'
            html += f'''<div class="metric-card">
                <div class="metric-label">Sequential</div>
                <div class="metric-value"><span class="{seq_status}">{'✓' if is_seq else '✗'}</span></div>
            </div>'''
        
        # Gaps
        gaps = trials.get('gaps', [])
        if gaps:
            html += f'''<div class="metric-card">
                <div class="metric-label">Gaps Detected</div>
                <div class="metric-value"><span class="warning">{len(gaps)}</span></div>
            </div>'''
        
        # Duplicates
        duplicates = trials.get('duplicates', [])
        if duplicates:
            html += f'''<div class="metric-card">
                <div class="metric-label">Duplicates</div>
                <div class="metric-value"><span class="warning">{len(duplicates)}</span></div>
            </div>'''
        
        # NaN count
        total_nan = trials.get('total_nan', 0)
        if total_nan > 0:
            html += f'''<div class="metric-card">
                <div class="metric-label">NaN Values</div>
                <div class="metric-value"><span class="alert">{total_nan}</span></div>
            </div>'''
        
        html += '</div>'
        
        # Trial duration histogram
        duration = trials.get('duration') if trials else None
        if duration and isinstance(duration, dict) and duration.get('values'):
            durations = _to_list(duration['values'], 10000)
            plot_id = f"task-dur-{block_num}"
            html += f'''<div class="plot-container">
                <h5>Trial Duration Distribution</h5>
                <div id="{plot_id}" class="plot plot-small"></div>
                <script>
                Plotly.newPlot('{plot_id}', [{{
                    x: {durations},
                    type: 'histogram',
                    marker: {{color: '#28a745'}},
                    nbinsx: 30
                }}], {{
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    xaxis: {{title: 'Duration (ms)'}},
                    yaxis: {{title: 'Count'}},
                    showlegend: false
                }});
                </script>
            </div>'''
            
            # Duration stats
            dur_stats = duration
            html += f'''<div class="status-box info">
                <strong>Duration Stats:</strong> 
                Mean={dur_stats.get('mean', 0):.1f}ms, 
                Median={dur_stats.get('median', 0):.1f}ms, 
                Std={dur_stats.get('std', 0):.1f}ms, 
                Range=[{dur_stats.get('min', 0):.1f}, {dur_stats.get('max', 0):.1f}]ms
            </div>'''
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------

def _build_eeg_section(results: Dict, block_num: int) -> str:
    """Build EEG section with all visualizations."""

    html = '''<div class="modality-section eeg">
    <div class="modality-header">
        <div class="modality-icon eeg"></div>
        <h3>EEG Data</h3>
    </div>'''
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        html += '<div class="metadata-grid">'
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Sampling Rate</div>
            <div class="metadata-value">{meta.get('sampling_rate', 0):.1f} Hz</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Channels</div>
            <div class="metadata-value">{meta.get('n_channels', 0)}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Samples</div>
            <div class="metadata-value">{meta.get('n_samples', 0):,}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Duration</div>
            <div class="metadata-value">{meta.get('duration_s', 0):.1f} s</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Trials</div>
            <div class="metadata-value">{meta.get('n_trials', 'N/A')}</div>
        </div>'''
        html += '</div>'
        
        # Channel names
        if meta.get('channel_names'):
            html += f"<p><strong>Channels:</strong> {', '.join(meta['channel_names'])}</p>"
    
    # Temporal Integrity
    if 'temporal' in results:
        temp = results['temporal']
        html += '<h4>⏱️ Temporal Integrity</h4>'
        
        # Sampling rate check
        expected_fs = temp.get('expected_fs', 250)
        actual_fs = temp.get('actual_fs', 0)
        fs_dev = temp.get('fs_deviation_pct', 0)
        
        fs_status = 'success' if fs_dev < 1 else ('warning' if fs_dev < 5 else 'error')
        html += f'''<div class="status-box {fs_status}">
            <strong>Sampling Rate:</strong> Expected={expected_fs}Hz, Actual={actual_fs:.2f}Hz, Deviation={fs_dev:.2f}%
        </div>'''
        
        # DT histogram
        if temp.get('dt', {}).get('values') is not None:
            dt_vals = _to_list(temp['dt']['values'], 50000)
            dt_stats = temp['dt']
            plot_id = f"eeg-dt-{block_num}"
            html += f'''<div class="plot-container-centered">
                <h5>Inter-Sample Time (ΔT) Distribution</h5>
                <div id="{plot_id}" style="width:700px; height:280px;"></div>
                <script>
                Plotly.newPlot('{plot_id}', [{{
                    x: {dt_vals},
                    type: 'histogram',
                    marker: {{color: '#007bff'}},
                    nbinsx: 50
                }}], {{
                    margin: {{t: 20, b: 35, l: 45, r: 15}},
                    width: 700,
                    height: 280,
                    xaxis: {{title: 'ΔT (ms)'}},
                    yaxis: {{title: 'Count'}},
                    showlegend: false,
                    shapes: [{{
                        type: 'line',
                        x0: {dt_stats.get('expected_ms', 4)},
                        x1: {dt_stats.get('expected_ms', 4)},
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {{color: 'red', width: 2, dash: 'dash'}}
                    }}]
                }}, {{responsive: false}});
                </script>
            </div>'''
            
            html += f'''<div class="status-box info">
                <strong>ΔT Stats:</strong> 
                Expected={dt_stats.get('expected_ms', 0):.2f}ms, 
                Median={dt_stats.get('median', 0):.2f}ms, 
                Mean={dt_stats.get('mean', 0):.2f}ms, 
                Std={dt_stats.get('std', 0):.2f}ms
            </div>'''
        
        # Duplicate timestamps
        n_dup = temp.get('duplicate_timestamps', 0)
        dup_pct = temp.get('duplicate_pct', 0)
        if n_dup > 0:
            html += f'''<div class="status-box warning">
                ⚠️ <strong>Duplicate Timestamps:</strong> {n_dup} ({dup_pct:.2f}%)
            </div>'''
    
    # Signal Validity
    if 'signal' in results:
        sig = results['signal']
        html += '<h4>📊 Signal Quality</h4>'
        
        # Invalid samples
        invalid_total = sig.get('invalid_samples_total', 0)
        invalid_pct = sig.get('invalid_pct_total', 0)
        
        status_class = 'success' if invalid_pct < 1 else ('warning' if invalid_pct < 10 else 'error')
        html += f'''<div class="status-box {status_class}">
            <strong>Invalid Samples:</strong> {invalid_total:,} ({invalid_pct:.2f}%)
        </div>'''
        
        # Invalid segments
        if sig.get('invalid_segments', {}).get('n_segments', 0) > 0:
            inv_seg = sig['invalid_segments']
            html += f'''<div class="status-box info">
                <strong>Invalid Segments:</strong> {inv_seg['n_segments']} segments
            </div>'''
            
            if inv_seg.get('stats'):
                stats = inv_seg['stats']
                html += f'''<p>Segment duration: Median={stats['median']:.1f}ms, 
                    Max={stats['max']:.1f}ms, Mean={stats['mean']:.1f}ms</p>'''
        
        # Channel correlation matrix - SQUARE
        if sig.get('correlation_matrix', {}).get('matrix'):
            corr_data = sig['correlation_matrix']
            matrix = corr_data['matrix']
            channels = corr_data.get('channel_names', [])
            n_channels = len(channels)
            
            # Calculate size for square matrix (min 400, max 600)
            plot_size = min(600, max(400, n_channels * 60))
            
            plot_id = f"eeg-corr-{block_num}"
            html += f'''<div class="plot-container-centered">
                <h5>Channel Correlation Matrix</h5>
                <div id="{plot_id}" style="width:{plot_size}px; height:{plot_size}px;"></div>
                <script>
                Plotly.newPlot('{plot_id}', [{{
                    z: {matrix},
                    x: {channels},
                    y: {channels},
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    reversescale: false,
                    zmid: 0,
                    zmin: -1,
                    zmax: 1,
                    colorbar: {{title: 'Corr', len: 0.9}}
                }}], {{
                    margin: {{t: 30, b: 80, l: 80, r: 60}},
                    width: {plot_size},
                    height: {plot_size},
                    xaxis: {{tickangle: -45, scaleanchor: 'y', scaleratio: 1}},
                    yaxis: {{autorange: 'reversed'}}
                }}, {{responsive: false}});
                </script>
            </div>'''
        
        # PSD plots
        if sig.get('psd'):
            html += '<details><summary><strong>Power Spectral Density (PSD)</strong></summary>'
            html += '<div class="plots-row">'
            
            for ch, psd_data in sig['psd'].items():
                if 'error' not in psd_data:
                    freqs = _to_list(psd_data['frequencies'], 5000)
                    power = _to_list(psd_data['power'], 5000)
                    
                    plot_id = f"eeg-psd-{ch}-{block_num}".replace(' ', '-')
                    html += f'''<div class="plot-container-thin">
                        <h5>{ch}</h5>
                        <div id="{plot_id}" style="height:160px;"></div>
                        <script>
                        Plotly.newPlot('{plot_id}', [{{
                            x: {freqs},
                            y: {power},
                            type: 'scatter',
                            mode: 'lines',
                            line: {{color: '#007bff'}}
                        }}], {{
                            margin: {{t: 15, b: 35, l: 50, r: 10}},
                            xaxis: {{title: 'Freq (Hz)', range: [0, 50]}},
                            yaxis: {{title: 'Power', type: 'log'}},
                            showlegend: false
                        }}, {{responsive: true}});
                        </script>
                    </div>'''
            
            html += '</div></details>'
        
        # DC offset
        if sig.get('dc_offset'):
            html += '<details><summary><strong>DC Offset per Channel</strong></summary>'
            html += '<table><tr><th>Channel</th><th>Mean (µV)</th><th>Min (µV)</th><th>Max (µV)</th></tr>'
            for ch, vals in sig['dc_offset'].items():
                if vals.get('mean') is not None:
                    html += f"<tr><td>{ch}</td><td>{vals['mean']:.2f}</td><td>{vals['min']:.2f}</td><td>{vals['max']:.2f}</td></tr>"
            html += '</table></details>'
        
        # Clipping
        if sig.get('clipping'):
            total_clip = sum(v.get('total_clipping_events', 0) for v in sig['clipping'].values() if isinstance(v, dict))
            if total_clip > 0:
                html += f'''<div class="status-box warning">
                    ⚠️ <strong>Clipping Detected:</strong> {total_clip} events across all channels
                </div>'''
    
    # Motion sensors
    if 'motion' in results:
        motion = results['motion']
        
        if motion.get('accelerometer'):
            html += '<details><summary><strong>Accelerometer</strong></summary>'
            html += '<div class="plots-row">'
            for axis, stats in motion['accelerometer'].items():
                vals = _to_list(stats.get('values'), 10000)
                plot_id = f"eeg-acc-{axis}-{block_num}".replace(' ', '-')
                
                # Center at zero with symmetric range based on data extent
                data_min = stats.get('min', -1)
                data_max = stats.get('max', 1)
                # Use the larger absolute value to create symmetric range
                abs_extent = max(abs(data_min), abs(data_max))
                padding = abs_extent * 0.1  # 10% padding
                x_range = [-(abs_extent + padding), abs_extent + padding]
                
                html += f'''<div class="plot-container-thin">
                    <h5>{axis}</h5>
                    <div id="{plot_id}" style="height:150px;"></div>
                    <script>
                    Plotly.newPlot('{plot_id}', [{{
                        x: {vals},
                        type: 'histogram',
                        marker: {{color: '#007bff'}},
                        nbinsx: 40
                    }}], {{
                        margin: {{t: 15, b: 50, l: 50, r: 10}},
                        xaxis: {{
                            title: '{axis}', 
                            titlefont: {{size: 11}}, 
                            range: {x_range},
                            zeroline: true,
                            zerolinewidth: 2,
                            zerolinecolor: '#999'
                        }},
                        yaxis: {{title: 'Count', titlefont: {{size: 11}}}},
                        showlegend: false
                    }}, {{responsive: true, displayModeBar: false}});
                    </script>
                </div>'''
            html += '</div></details>'
        
        if motion.get('gyroscope'):
            html += '<details><summary><strong>Gyroscope</strong></summary>'
            html += '<div class="plots-row">'
            for axis, stats in motion['gyroscope'].items():
                vals = _to_list(stats.get('values'), 10000)
                plot_id = f"eeg-gyr-{axis}-{block_num}".replace(' ', '-')
                
                # Center at zero with symmetric range based on data extent
                data_min = stats.get('min', -1)
                data_max = stats.get('max', 1)
                # Use the larger absolute value to create symmetric range
                abs_extent = max(abs(data_min), abs(data_max))
                padding = abs_extent * 0.1  # 10% padding
                x_range = [-(abs_extent + padding), abs_extent + padding]
                
                html += f'''<div class="plot-container-thin">
                    <h5>{axis}</h5>
                    <div id="{plot_id}" style="height:150px;"></div>
                    <script>
                    Plotly.newPlot('{plot_id}', [{{
                        x: {vals},
                        type: 'histogram',
                        marker: {{color: '#007bff'}},
                        nbinsx: 40
                    }}], {{
                        margin: {{t: 15, b: 50, l: 50, r: 10}},
                        xaxis: {{
                            title: '{axis}', 
                            titlefont: {{size: 11}}, 
                            range: {x_range},
                            zeroline: true,
                            zerolinewidth: 2,
                            zerolinecolor: '#999'
                        }},
                        yaxis: {{title: 'Count', titlefont: {{size: 11}}}},
                        showlegend: false
                    }}, {{responsive: true, displayModeBar: false}});
                    </script>
                </div>'''
            html += '</div></details>'
    
    # Triggers - simplified (just total events and trial count)
    if 'triggers' in results:
        trig = results['triggers']
        n_events = trig.get('n_events', 0)
        n_unique = len(trig.get('trigger_counts', {}))
        
        html += '<h4>🎯 Trigger Events</h4>'
        html += '<div class="metric-cards">'
        html += f'''<div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-value">{n_events}</div>
        </div>'''
        html += f'''<div class="metric-card">
            <div class="metric-label">Unique Triggers</div>
            <div class="metric-value">{n_unique}</div>
        </div>'''
        
        if trig.get('trial_state_info', {}).get('n_trials'):
            info = trig['trial_state_info']
            html += f'''<div class="metric-card">
                <div class="metric-label">Trials Identified</div>
                <div class="metric-value">{info['n_trials']}</div>
            </div>'''
        
        html += '</div>'
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------
def _build_eyes_section(results: Dict, block_num: int, screen_bounds: tuple = None) -> str:
    """Build eyes section with all visualizations.
    
    Args:
        results: Eyes QC results dictionary
        block_num: Block number
        screen_bounds: Optional tuple (x_min, x_max, y_min, y_max) from task metadata
    """

    html = '''<div class="modality-section eyes">
    <div class="modality-header">
        <div class="modality-icon eyes"></div>
        <h3>Eye Tracking Data</h3>
    </div>'''
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        html += '<div class="metadata-grid">'
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Sampling Rate</div>
            <div class="metadata-value">{meta.get('sampling_rate', 0):.1f} Hz</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Eye Tracked</div>
            <div class="metadata-value">{meta.get('eye_tracked', 'N/A')}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Duration</div>
            <div class="metadata-value">{meta.get('duration_s', 0):.1f} s</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Fixations</div>
            <div class="metadata-value">{meta.get('n_fixations', 0)}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Saccades</div>
            <div class="metadata-value">{meta.get('n_saccades', 0)}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Blinks</div>
            <div class="metadata-value">{meta.get('n_blinks', 0)}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Trials</div>
            <div class="metadata-value">{meta.get('n_trials', 'N/A')}</div>
        </div>'''
        html += '</div>'
    
    # Temporal integrity
    if 'temporal' in results:
        temp = results['temporal']
        html += '<h4>⏱️ Temporal Integrity</h4>'
        
        # Duration coverage
        if temp.get('duration_coverage'):
            cov = temp['duration_coverage']
            coverage_pct = cov.get('coverage_pct', 0)
            is_binocular = cov.get('is_binocular', False)
            status_class = 'success' if coverage_pct > 80 else ('warning' if coverage_pct > 50 else 'error')
            
            # Show per-eye coverage if binocular
            if is_binocular and cov.get('per_eye'):
                per_eye = cov['per_eye']
                html += f'''<div class="status-box {status_class}">
                    <strong>Event Coverage (binocular):</strong> {coverage_pct:.1f}% avg
                    (L: {per_eye.get('L', {}).get('coverage_pct', 0):.1f}%, R: {per_eye.get('R', {}).get('coverage_pct', 0):.1f}%)
                </div>'''
                
                html += '<div class="metric-cards">'
                for eye in ['L', 'R']:
                    eye_data = per_eye.get(eye, {})
                    html += f'''<div class="metric-card">
                        <div class="metric-label">{eye} Eye Coverage</div>
                        <div class="metric-value">{eye_data.get('coverage_pct', 0):.1f} <span class="metric-unit">%</span></div>
                    </div>'''
                html += '</div>'
            else:
                html += f'''<div class="status-box {status_class}">
                    <strong>Event Coverage:</strong> {coverage_pct:.1f}% of recording time
                </div>'''
                
                html += '<div class="metric-cards">'
                html += f'''<div class="metric-card">
                    <div class="metric-label">Fixation Time</div>
                    <div class="metric-value">{cov.get('fixation_ms', 0)/1000:.1f} <span class="metric-unit">s</span></div>
                </div>'''
                html += f'''<div class="metric-card">
                    <div class="metric-label">Saccade Time</div>
                    <div class="metric-value">{cov.get('saccade_ms', 0)/1000:.1f} <span class="metric-unit">s</span></div>
                </div>'''
                html += f'''<div class="metric-card">
                    <div class="metric-label">Blink Time</div>
                    <div class="metric-value">{cov.get('blink_ms', 0)/1000:.1f} <span class="metric-unit">s</span></div>
                </div>'''
                html += '</div>'
            
            # Show note if present
            if cov.get('note'):
                html += f'<p><small><em>{cov["note"]}</em></small></p>'
        
        # Overlapping events
        if temp.get('overlapping_events'):
            overlaps = temp['overlapping_events']
            total_overlaps = sum(overlaps.values())
            if total_overlaps > 0:
                html += f'''<div class="status-box warning">
                    ⚠️ <strong>Overlapping Events:</strong> {total_overlaps} detected
                    (Fix-Sacc: {overlaps.get('fixation_saccade', 0)}, 
                    Fix-Blink: {overlaps.get('fixation_blink', 0)}, 
                    Sacc-Blink: {overlaps.get('saccade_blink', 0)})
                </div>'''
    
    # Signal validity
    if 'signal' in results:
        sig = results['signal']
        html += '<h4>📊 Signal Quality</h4>'
        
        # NaN percentage
        if sig.get('nan_pct_gaze'):
            max_nan = max(sig['nan_pct_gaze'].values())
            status_class = 'success' if max_nan < 10 else ('warning' if max_nan < 30 else 'error')
            html += f'''<div class="status-box {status_class}">
                <strong>Data Loss (NaN):</strong> {', '.join(f'{k}: {v:.1f}%' for k, v in sig['nan_pct_gaze'].items())}
            </div>'''
        
        # Gap analysis
        if sig.get('gap_durations', {}).get('n_gaps', 0) > 0:
            gaps = sig['gap_durations']
            html += f"<p><strong>Gaps:</strong> {gaps['n_gaps']} detected"
            if gaps.get('stats'):
                html += f" (median: {gaps['stats']['median']:.1f}ms, max: {gaps['stats']['max']:.1f}ms)"
            html += "</p>"
        
        # Saccade/Fixation ratio
        if sig.get('saccade_fixation_ratio', {}).get('ratio') is not None:
            ratio_data = sig['saccade_fixation_ratio']
            ratio = ratio_data['ratio']
            status_class = 'success' if 0.8 <= ratio <= 1.2 else 'warning'
            html += f'''<div class="status-box {status_class}">
                <strong>Saccade/Fixation Ratio:</strong> {ratio:.2f} 
                ({ratio_data['n_saccades']} saccades / {ratio_data['n_fixations']} fixations)
            </div>'''
        
        # Velocity/Acceleration stats
        if sig.get('velocity_stats', {}).get('velocity'):
            vel_stats = sig['velocity_stats']['velocity']
            acc_stats = sig['velocity_stats'].get('acceleration', {})
            
            html += '<details><summary><strong>Velocity & Acceleration</strong></summary>'
            
            # Show capping warnings if any
            if vel_stats.get('n_capped', 0) > 0:
                html += f'''<div class="status-box warning">
                    ⚠️ <strong>{vel_stats['n_capped']}</strong> velocity samples exceeded 3000 px/s and were excluded from statistics
                </div>'''
            
            if acc_stats.get('n_capped', 0) > 0:
                html += f'''<div class="status-box warning">
                    ⚠️ <strong>{acc_stats['n_capped']}</strong> acceleration samples exceeded the 99.5th percentile and were excluded from statistics
                </div>'''
            
            # Velocity histogram
            vel_vals = _to_list(vel_stats.get('values'), 10000)
            plot_id = f"eyes-vel-{block_num}"
            html += f'''<div class="plot-container">
                <h5>Gaze Velocity Distribution</h5>
                <div id="{plot_id}" class="plot plot-small"></div>
                <script>
                Plotly.newPlot('{plot_id}', [{{
                    x: {vel_vals},
                    type: 'histogram',
                    marker: {{color: '#17a2b8'}},
                    nbinsx: 50
                }}], {{
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    xaxis: {{title: 'Speed (px/s)'}},
                    yaxis: {{title: 'Count'}},
                    showlegend: false
                }});
                </script>
            </div>'''
            
            html += f'''<div class="status-box info">
                <strong>Velocity:</strong> Median={vel_stats['median']:.1f} px/s, 
                Mean={vel_stats['mean']:.1f} px/s, Max={vel_stats['max']:.1f} px/s
            </div>'''
            
            # Acceleration histogram
            if acc_stats.get('values') is not None:
                acc_vals = _to_list(acc_stats.get('values'), 10000)
                plot_id = f"eyes-acc-{block_num}"
                html += f'''<div class="plot-container">
                    <h5>Gaze Acceleration Distribution</h5>
                    <div id="{plot_id}" class="plot plot-small"></div>
                    <script>
                    Plotly.newPlot('{plot_id}', [{{
                        x: {acc_vals},
                        type: 'histogram',
                        marker: {{color: '#17a2b8'}},
                        nbinsx: 50
                    }}], {{
                        margin: {{t: 30, b: 40, l: 50, r: 20}},
                        xaxis: {{title: 'Acceleration (px/s²)'}},
                        yaxis: {{title: 'Count'}},
                        showlegend: false
                    }});
                    </script>
                </div>'''
            
            html += '</details>'
        
        # Gaze heatmap - bounded to display area
        if sig.get('gaze_distribution'):
            gaze = sig['gaze_distribution']
            x_raw = gaze.get('x')
            y_raw = gaze.get('y')
            gaze_screen_bounds = gaze.get('screen_bounds')
            
            # Use screen_bounds from task metadata if available, otherwise from gaze data
            x_min, x_max = 0, 1280
            y_min, y_max = 0, 1024
            if screen_bounds:
                x_min, x_max, y_min, y_max = screen_bounds
            elif gaze_screen_bounds:
                x_min, x_max, y_min, y_max = gaze_screen_bounds
            
            # Convert to numpy and filter to bounds (np already imported at module level)
            x_arr = np.array(x_raw) if x_raw is not None else np.array([])
            y_arr = np.array(y_raw) if y_raw is not None else np.array([])
            
            if len(x_arr) > 0 and len(y_arr) > 0:
                # Count out-of-bounds data
                total_valid = len(x_arr)
                in_bounds_mask = (x_arr >= x_min) & (x_arr <= x_max) & (y_arr >= y_min) & (y_arr <= y_max)
                out_of_bounds_pct = (1 - np.sum(in_bounds_mask) / total_valid) * 100
                
                # Filter to display bounds for visualization
                x_bounded = x_arr[in_bounds_mask]
                y_bounded = y_arr[in_bounds_mask]
                
                x = _to_list(x_bounded, 10000)
                y = _to_list(y_bounded, 10000)
                
                plot_id = f"eyes-heat-{block_num}"
                
                # Show warning if significant out-of-bounds data
                if out_of_bounds_pct > 5:
                    html += f'''<div class="status-box warning">
                        ⚠️ <strong>{out_of_bounds_pct:.1f}%</strong> of gaze data falls outside display bounds [{x_min}, {x_max}] x [{y_min}, {y_max}]
                    </div>'''
                
                # Calculate aspect ratio for proper display shape
                aspect_ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 1.25
                plot_height = 525
                plot_width = int(plot_height * aspect_ratio)
                
                html += f'''<div class="plot-container-centered">
                    <h5>Gaze Position Heatmap (Display: {int(x_max-x_min)}x{int(y_max-y_min)})</h5>
                    <div id="{plot_id}" style="width:{plot_width}px; height:{plot_height}px;"></div>
                    <script>
                    Plotly.newPlot('{plot_id}', [{{
                        x: {x},
                        y: {y},
                        type: 'histogram2dcontour',
                        colorscale: 'Hot',
                        reversescale: false,
                        showscale: true,
                        ncontours: 20,
                        contours: {{
                            coloring: 'heatmap'
                        }}
                    }}], {{
                        margin: {{t: 30, b: 50, l: 60, r: 60}},
                        width: {plot_width},
                        height: {plot_height},
                        paper_bgcolor: '#fff',
                        plot_bgcolor: '#000',
                        xaxis: {{
                            title: {{text: 'X (pixels)', font: {{color: '#000'}}}},
                            range: [{x_min}, {x_max}],
                            constrain: 'domain',
                            tickfont: {{color: '#000'}},
                            gridcolor: '#000',
                            zerolinecolor: '#000',
                            showgrid: false
                        }},
                        yaxis: {{
                            title: {{text: 'Y (pixels)', font: {{color: '#000'}}}},
                            range: [{y_max}, {y_min}],
                            scaleanchor: 'x',
                            scaleratio: 1,
                            constrain: 'domain',
                            tickfont: {{color: '#000'}},
                            gridcolor: '#000',
                            zerolinecolor: '#000',
                            showgrid: false
                        }},
                        showlegend: false
                    }}, {{responsive: false}});
                    </script>
                </div>'''
    
    # Event consistency
    if 'events' in results:
        evt = results['events']
        html += '<details><summary><strong>Event Duration Statistics</strong></summary>'
        
        if evt.get('fixation_duration'):
            fix = evt['fixation_duration']
            html += f'''<div class="status-box info">
                <strong>Fixation Duration:</strong> Median={fix['median']:.1f}ms, 
                Mean={fix['mean']:.1f}ms, Range=[{fix['min']:.1f}, {fix['max']:.1f}]ms
            </div>'''
        
        if evt.get('saccade_duration'):
            sacc = evt['saccade_duration']
            html += f'''<div class="status-box info">
                <strong>Saccade Duration:</strong> Median={sacc['median']:.1f}ms, 
                Mean={sacc['mean']:.1f}ms, Range=[{sacc['min']:.1f}, {sacc['max']:.1f}]ms
            </div>'''
        
        if evt.get('saccade_amplitude'):
            amp = evt['saccade_amplitude']
            html += f'''<div class="status-box info">
                <strong>Saccade Amplitude:</strong> Median={amp['median']:.1f}°, 
                Mean={amp['mean']:.1f}°, Max={amp['max']:.1f}°
            </div>'''
        
        if evt.get('blink_duration'):
            blink = evt['blink_duration']
            html += f'''<div class="status-box info">
                <strong>Blink Duration:</strong> Median={blink['median']:.1f}ms, 
                Mean={blink['mean']:.1f}ms, Range=[{blink['min']:.1f}, {blink['max']:.1f}]ms
            </div>'''
        
        html += '</details>'
    
    # Trigger Events
    if 'triggers' in results:
        trig = results['triggers']
        n_events = trig.get('n_events', 0)
        n_unique = trig.get('n_unique_triggers', 0)
        
        html += '<h4>🎯 Trigger Events</h4>'
        html += '<div class="metric-cards">'
        html += f'''<div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-value">{n_events}</div>
        </div>'''
        html += f'''<div class="metric-card">
            <div class="metric-label">Unique Triggers</div>
            <div class="metric-value">{n_unique}</div>
        </div>'''
        
        if trig.get('trial_state_info', {}).get('n_trials'):
            info = trig['trial_state_info']
            html += f'''<div class="metric-card">
                <div class="metric-label">Trials Identified</div>
                <div class="metric-value">{info['n_trials']}</div>
            </div>'''
        
        html += '</div>'
    
    # Binocular
    if results.get('binocular'):
        bino = results['binocular']
        html += '<details><summary><strong>Binocular Consistency</strong></summary>'
        
        if bino.get('gaze_correlation'):
            corr = bino['gaze_correlation']
            html += f'''<div class="status-box info">
                <strong>Left/Right Gaze Correlation:</strong> X={corr['x']:.3f}, Y={corr['y']:.3f}
            </div>'''
        
        if bino.get('pupil_correlation') is not None:
            html += f'''<div class="status-box info">
                <strong>Left/Right Pupil Correlation:</strong> {bino['pupil_correlation']:.3f}
            </div>'''
        
        if bino.get('vergence'):
            verg = bino['vergence']
            html += f'''<div class="status-box info">
                <strong>Vergence (L-R difference):</strong> 
                X: mean={verg['x_diff_mean']:.1f}px, std={verg['x_diff_std']:.1f}px | 
                Y: mean={verg['y_diff_mean']:.1f}px, std={verg['y_diff_std']:.1f}px
            </div>'''
        
        html += '</details>'
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------
def _build_kinematics_section(results: Dict, block_num: int) -> str:
    """Build kinematics section with all visualizations."""

    html = '''<div class="modality-section kinematics">
    <div class="modality-header">
        <div class="modality-icon kinematics"></div>
        <h3>Kinematics Data</h3>
    </div>'''
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        html += '<div class="metadata-grid">'
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Sampling Rate</div>
            <div class="metadata-value">{meta.get('sampling_rate', 0):.1f} Hz</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Markers</div>
            <div class="metadata-value">{meta.get('n_markers', 'N/A')}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Frames</div>
            <div class="metadata-value">{meta.get('n_frames', 0):,}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Duration</div>
            <div class="metadata-value">{meta.get('duration_s', 0):.1f} s</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Trials</div>
            <div class="metadata-value">{meta.get('n_trials', 'N/A')}</div>
        </div>'''
        html += '</div>'
        
        if meta.get('marker_names'):
            html += f"<p><strong>Markers:</strong> {', '.join(meta['marker_names'])}</p>"
    
    # Signal validity
    if 'signal' in results:
        sig = results['signal']
        html += '<h4>📊 Signal Quality</h4>'
        
        # Missing frames
        if sig.get('missing_frames'):
            html += '<h5>Missing Frames per Marker</h5>'
            html += '<table><tr><th>Marker</th><th>Missing</th><th>%</th><th>Status</th></tr>'
            for marker, vals in sig['missing_frames'].items():
                pct = vals['pct_missing']
                status_class = 'ok' if pct < 5 else ('alert' if pct < 20 else 'warning')
                status_text = '✓' if pct < 5 else ('⚠' if pct < 20 else '✗')
                html += f"<tr><td>{marker}</td><td>{vals['n_missing']}</td><td class='{status_class}'>{pct:.1f}%</td><td>{status_text}</td></tr>"
            html += "</table>"
        
        # Gap analysis
        if sig.get('gap_analysis'):
            html += '<details><summary><strong>Gap Analysis per Marker</strong></summary>'
            for marker, gap_data in sig['gap_analysis'].items():
                if gap_data.get('n_gaps', 0) > 0:
                    html += f"<h5>{marker}</h5>"
                    html += f"<p><strong>Gaps:</strong> {gap_data['n_gaps']}"
                    if gap_data.get('stats'):
                        stats = gap_data['stats']
                        html += f" (median: {stats['median']:.1f}ms, max: {stats['max']:.1f}ms)"
                    html += "</p>"
            html += '</details>'
        
        # Residuals - CHECK IF THESE ARE ACTUALLY RESIDUALS
        if sig.get('residuals'):
            html += '<details><summary><strong>Marker Reconstruction Residuals</strong></summary>'
            html += '<div class="status-box info">Note: Residuals indicate reconstruction quality. Lower is better.</div>'
            html += '<table><tr><th>Marker</th><th>Median</th><th>Mean</th><th>Std</th><th>Max</th></tr>'
            for marker, vals in sig['residuals'].items():
                if 'error' not in vals:
                    # Check if values look suspiciously like coordinates (>10mm is unusual for residuals)
                    if vals['median'] > 10:
                        html += f"<tr style='background-color: #fff3cd;'><td>{marker}</td><td>{vals['median']:.2f}</td><td>{vals['mean']:.2f}</td><td>{vals['std']:.2f}</td><td>{vals['max']:.2f}</td><td class='warning'>⚠ High values!</td></tr>"
                    else:
                        html += f"<tr><td>{marker}</td><td>{vals['median']:.2f}</td><td>{vals['mean']:.2f}</td><td>{vals['std']:.2f}</td><td>{vals['max']:.2f}</td></tr>"
            html += '</table>'
            html += '<div class="status-box warning">⚠️ <strong>Warning:</strong> If residual values are >10mm, they may actually be Z-coordinates rather than residuals. Check the kinematics data class!</div>'
            html += '</details>'
    
    # Velocity/Acceleration - COMPACT GRID LAYOUT using individual plots in flexbox
    if 'temporal' in results:
        temp = results['temporal']
        
        # Collect all markers with valid data
        markers_with_data = [(m, s) for m, s in temp.items() if 'error' not in s and s.get('velocity', {}).get('values') is not None]
        
        if markers_with_data:
            html += '<details open><summary><strong>Marker Velocity & Acceleration</strong></summary>'
            
            # Build summary table first
            html += '<table><tr><th>Marker</th><th>Vel Median (mm/s)</th><th>Vel Max (mm/s)</th><th>Acc Median (mm/s²)</th><th>Acc Max (mm/s²)</th></tr>'
            for marker, stats in markers_with_data:
                vel = stats.get('velocity', {})
                acc = stats.get('acceleration', {})
                html += f"<tr><td>{marker}</td>"
                html += f"<td>{vel.get('median', 0):.1f}</td><td>{vel.get('max', 0):.1f}</td>"
                html += f"<td>{acc.get('median', 0):.1f}</td><td>{acc.get('max', 0):.1f}</td></tr>"
            html += '</table>'
            
            # Velocity histograms in a flex grid
            html += '<h5>Velocity Distributions (mm/s)</h5>'
            html += '<div class="plots-row">'
            
            for marker, stats in markers_with_data:
                vel = stats.get('velocity', {})
                if vel.get('values') is not None:
                    vel_vals = _to_list(vel.get('values'), 3000)
                    plot_id = f"kin-vel-{marker}-{block_num}".replace(' ', '-').replace('_', '-')
                    html += f'''<div class="plot-container-thin" style="flex: 1 1 200px; max-width: 280px;">
                        <h6 style="margin:0 0 5px 0; font-size:0.85em;">{marker}</h6>
                        <div id="{plot_id}" style="height:120px;"></div>
                        <script>
                        Plotly.newPlot('{plot_id}', [{{
                            x: {vel_vals},
                            type: 'histogram',
                            marker: {{color: '#ffc107'}},
                            nbinsx: 25
                        }}], {{
                            margin: {{t: 5, b: 25, l: 35, r: 5}},
                            xaxis: {{title: '', tickfont: {{size: 9}}}},
                            yaxis: {{title: '', tickfont: {{size: 9}}}},
                            showlegend: false
                        }}, {{responsive: true, displayModeBar: false}});
                        </script>
                    </div>'''
            
            html += '</div>'
            
            # Acceleration histograms in a flex grid
            html += '<h5>Acceleration Distributions (mm/s²)</h5>'
            html += '<div class="plots-row">'
            
            for marker, stats in markers_with_data:
                acc = stats.get('acceleration', {})
                if acc.get('values') is not None:
                    acc_vals = _to_list(acc.get('values'), 3000)
                    plot_id = f"kin-acc-{marker}-{block_num}".replace(' ', '-').replace('_', '-')
                    html += f'''<div class="plot-container-thin" style="flex: 1 1 200px; max-width: 280px;">
                        <h6 style="margin:0 0 5px 0; font-size:0.85em;">{marker}</h6>
                        <div id="{plot_id}" style="height:120px;"></div>
                        <script>
                        Plotly.newPlot('{plot_id}', [{{
                            x: {acc_vals},
                            type: 'histogram',
                            marker: {{color: '#fd7e14'}},
                            nbinsx: 25
                        }}], {{
                            margin: {{t: 5, b: 25, l: 35, r: 5}},
                            xaxis: {{title: '', tickfont: {{size: 9}}}},
                            yaxis: {{title: '', tickfont: {{size: 9}}}},
                            showlegend: false
                        }}, {{responsive: true, displayModeBar: false}});
                        </script>
                    </div>'''
            
            html += '</div>'
            html += '</details>'
    
    # Rigid body distances
    if results.get('rigid_body'):
        html += '<details><summary><strong>Rigid Body Inter-Marker Distances</strong></summary>'
        for group_name, group_data in results['rigid_body'].items():
            html += f"<h5>{group_name}: {', '.join(group_data['markers'])}</h5>"
            html += '<table><tr><th>Marker Pair</th><th>Mean (mm)</th><th>Std (mm)</th><th>CV (%)</th></tr>'
            for pair, dist in group_data['distances'].items():
                if 'error' not in dist:
                    cv = dist.get('cv', 0)
                    status_class = 'ok' if cv < 2 else ('alert' if cv < 5 else 'warning')
                    html += f"<tr><td>{pair}</td><td>{dist['mean']:.2f}</td><td>{dist['std']:.2f}</td><td class='{status_class}'>{cv:.2f}</td></tr>"
            html += '</table>'
        html += '</details>'
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------
def _build_emg_section(results: Dict, block_num: int) -> str:
    """Build EMG section with all visualizations."""

    html = '''<div class="modality-section emg">
    <div class="modality-header">
        <div class="modality-icon emg"></div>
        <h3>EMG Data</h3>
    </div>'''
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        html += '<div class="metadata-grid">'
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Sampling Rate</div>
            <div class="metadata-value">{meta.get('sampling_rate', 0):.1f} Hz</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Channels</div>
            <div class="metadata-value">{meta.get('n_channels', 'N/A')}</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Duration</div>
            <div class="metadata-value">{meta.get('duration_s', 0):.1f} s</div>
        </div>'''
        html += f'''<div class="metadata-item">
            <div class="metadata-label">Trials</div>
            <div class="metadata-value">{meta.get('n_trials', 'N/A')}</div>
        </div>'''
        html += '</div>'
        
        if meta.get('channel_names'):
            html += f"<p><strong>Channels:</strong> {', '.join(meta['channel_names'])}</p>"
    
    # Signal quality
    if 'signal' in results:
        sig = results['signal']
        html += '<h4>📊 Signal Quality</h4>'
        
        # DC offset
        if sig.get('dc_offset'):
            html += '<details><summary><strong>DC Offset per Channel</strong></summary>'
            html += '<table><tr><th>Channel</th><th>Mean (V)</th><th>Min (V)</th><th>Max (V)</th></tr>'
            for ch, vals in sig['dc_offset'].items():
                if 'error' not in vals:
                    html += f"<tr><td>{ch}</td><td>{vals['mean']:.4f}</td><td>{vals['min']:.4f}</td><td>{vals['max']:.4f}</td></tr>"
            html += '</table></details>'
        
        # PSD
        if sig.get('psd'):
            html += '<details><summary><strong>Power Spectral Density (PSD)</strong></summary>'
            html += '<div class="plots-row">'
            for ch, psd_data in sig['psd'].items():
                if 'error' not in psd_data:
                    freqs = _to_list(psd_data['frequencies'], 5000)
                    power = _to_list(psd_data['power'], 5000)
                    
                    plot_id = f"emg-psd-{ch}-{block_num}".replace(' ', '-')
                    html += f'''<div class="plot-container-thin">
                        <h5>{ch}</h5>
                        <div id="{plot_id}" style="height:160px;"></div>
                        <script>
                        Plotly.newPlot('{plot_id}', [{{
                            x: {freqs},
                            y: {power},
                            type: 'scatter',
                            mode: 'lines',
                            line: {{color: '#dc3545'}}
                        }}], {{
                            margin: {{t: 15, b: 35, l: 50, r: 10}},
                            xaxis: {{title: 'Freq (Hz)'}},
                            yaxis: {{title: 'Power', type: 'log'}},
                            showlegend: false
                        }}, {{responsive: true}});
                        </script>
                    </div>'''
            html += '</div></details>'
        
        # Clipping
        if sig.get('clipping'):
            total_clip = sum(v.get('total_clipping_events', 0) for v in sig['clipping'].values() if isinstance(v, dict))
            if total_clip > 0:
                html += f'''<div class="status-box warning">
                    ⚠️ <strong>Clipping Detected:</strong> {total_clip} events
                </div>'''
                
                html += '<details><summary><strong>Clipping Details</strong></summary>'
                html += '<table><tr><th>Channel</th><th>At Min</th><th>At Max</th><th>Total</th></tr>'
                for ch, clip in sig['clipping'].items():
                    if isinstance(clip, dict):
                        html += f"<tr><td>{ch}</td><td>{clip.get('clipping_at_min', 0)}</td><td>{clip.get('clipping_at_max', 0)}</td><td>{clip.get('total_clipping_events', 0)}</td></tr>"
                html += '</table></details>'
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------
def _build_crossmodal_section(results: Dict, block_num: int) -> str:
    """Build cross-modal alignment section."""

    html = '''<div class="modality-section crossmodal">
    <div class="modality-header">
        <div class="modality-icon crossmodal"></div>
        <h3>Cross-Modal Alignment</h3>
    </div>'''
    
    if 'error' in results:
        html += f'''<div class="status-box error">
            ❌ {results['error']}
        </div></div>'''
        return html
    
    # Consistency checks
    if 'consistency' in results and results.get('consistency'):
        cons = results['consistency']
        html += '<h4>🔗 Data Consistency</h4>'
        
        # Duration consistency
        if cons.get('durations', {}).get('values_s'):
            html += '<h5>Recording Durations</h5>'
            html += '<table><tr><th>Modality</th><th>Duration (s)</th></tr>'
            for mod, dur in cons['durations']['values_s'].items():
                html += f'<tr><td>{mod.capitalize()}</td><td>{dur:.2f}</td></tr>'
            html += '</table>'
            
            rng = cons['durations'].get('range', 0)
            status_class = 'success' if rng < 0.5 else ('warning' if rng < 2 else 'error')
            html += f'''<div class="status-box {status_class}">
                <strong>Duration Range:</strong> {rng:.2f}s 
                (min={cons['durations'].get('min', 0):.2f}s, max={cons['durations'].get('max', 0):.2f}s)
            </div>'''
        
        # Trial count consistency
        if cons.get('trial_counts', {}).get('values'):
            trial_counts = cons['trial_counts']['values']
            consistent = cons['trial_counts'].get('consistent', False)
            
            html += '<h5>Trial Counts</h5>'
            html += '<table><tr><th>Modality</th><th>Trials</th></tr>'
            for mod, count in trial_counts.items():
                html += f'<tr><td>{mod.capitalize()}</td><td>{count}</td></tr>'
            html += '</table>'
            
            status_class = 'success' if consistent else 'error'
            status_text = '✓ All modalities have same trial count' if consistent else '✗ Trial counts differ across modalities'
            html += f'''<div class="status-box {status_class}">
                {status_text}
            </div>'''
    
    # Event alignment
    if results.get('event_alignment'):
        evt_align = results['event_alignment']
        html += '<h4>⏱️ Timing Alignment</h4>'
        
        if evt_align.get('modalities_with_trial_times'):
            mods = evt_align['modalities_with_trial_times']
            html += f"<p><strong>Modalities with trial timing:</strong> {', '.join(mods)}</p>"
        
        # Offset analysis
        if evt_align.get('offset_analysis') and isinstance(evt_align['offset_analysis'], dict):
            offset_data = evt_align['offset_analysis']
            
            if 'note' not in offset_data:
                html += '<h5>Systematic Offset Analysis</h5>'
                
                for pair_name, analysis in offset_data.items():
                    if 'error' not in analysis:
                        html += f"<h6>{pair_name.replace('_', ' ').title()}</h6>"
                        
                        off_stats = analysis.get('offset_stats', {})
                        mean_off = off_stats.get('mean_ms', 0)
                        std_off = off_stats.get('std_ms', 0)
                        
                        # Determine status
                        if abs(mean_off) < 5 and std_off < 10:
                            status_class = 'success'
                        elif abs(mean_off) < 20 and std_off < 50:
                            status_class = 'warning'
                        else:
                            status_class = 'error'
                        
                        html += f'''<div class="status-box {status_class}">
                            <strong>Time Offset:</strong> {mean_off:.2f}ms ± {std_off:.2f}ms
                            (median={off_stats.get('median_ms', 0):.2f}ms, 
                            range=[{off_stats.get('min_ms', 0):.2f}, {off_stats.get('max_ms', 0):.2f}]ms)
                        </div>'''
                        
                        # Drift warning
                        drift = analysis.get('drift_estimate', {})
                        if drift.get('has_drift'):
                            html += f'''<div class="status-box warning">
                                ⚠️ <strong>Clock Drift Detected:</strong> slope deviation = {drift['slope_deviation']:.6f}
                            </div>'''
                        
                        # Regression stats
                        reg = analysis.get('regression', {})
                        html += f'''<p><small>Regression: slope={reg.get('slope', 0):.4f}, 
                            R²={reg.get('r_squared', 0):.4f}, 
                            n_events={analysis.get('n_matched_events', 0)}</small></p>'''
    
    html += '</div>'
    return html

#--------------------------------------------------------------------------------------
def _to_list(data, max_len: int = 10000) -> list:
    """Convert numpy array or list to JSON-serializable list, truncating if needed."""

    if data is None:
        return []
    
    if isinstance(data, np.ndarray):
        data = data.flatten()
        if len(data) > max_len:
            # Sample evenly
            indices = np.linspace(0, len(data)-1, max_len, dtype=int)
            data = data[indices]
        return data.tolist()
    elif isinstance(data, list):
        if len(data) > max_len:
            step = len(data) // max_len
            data = data[::step]
        return data
    else:
        return list(data)

# #=================================================================================================