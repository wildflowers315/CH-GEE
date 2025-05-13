import os
import subprocess
from pathlib import Path

# Set parameters
aoi_path = '../downloads/new_aoi.geojson'
if not os.path.exists(aoi_path):
    raise FileNotFoundError(f"AOI file not found at {aoi_path}")

# Create output directory
output_dir = 'chm_outputs'
os.makedirs(output_dir, exist_ok=True)

# Build command with all your parameters
cmd = [
    'python', 'chm_main.py',
    '--aoi', aoi_path,
    '--year', '2022',
    '--start-date', '01-01',
    '--end-date', '12-31',
    '--gedi-start-date', '2022-01-01',
    '--gedi-end-date', '2022-12-31',
    '--clouds-th', '70',
    '--quantile', 'rh98',
    '--model', 'RF',
    '--num-trees-rf', '500',
    '--min-leaf-pop-rf', '5',
    '--bag-frac-rf', '0.5',
    '--max-nodes-rf', '1000',
    '--output-dir', output_dir,
    '--export-training',
    '--export-predictions',
    '--scale', '30'
]

# Convert all arguments to strings
cmd = [str(arg) for arg in cmd]

# Run the command
print("Running canopy height model...")
subprocess.run(cmd, check=True)
print("Processing complete!")