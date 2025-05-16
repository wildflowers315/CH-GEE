import os
import subprocess
from pathlib import Path

# Set parameters
aoi_path = '../downloads/new_aoi.geojson'
if not os.path.exists(aoi_path):
    raise FileNotFoundError(f"AOI file not found at {aoi_path}")

# Create output directories
output_dir = 'chm_outputs'
eval_dir = os.path.join(output_dir, 'evaluation')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Build command for model training and prediction
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
    '--scale', '30',
    '--ndvi-threshold', '0.35',
    '--mask-type', 'ALL',
]

# Convert all arguments to strings
cmd = [str(arg) for arg in cmd]

# Run the model training and prediction
print("Running canopy height model...")
subprocess.run(cmd, check=True)

# Run evaluation with PDF report generation
eval_cmd = [
    'python', 'evaluate_predictions.py',
    '--pred', os.path.join(output_dir, 'chm_export_1747130473.tif'),
    '--ref', os.path.join(output_dir, 'dchm_09id4.tif'),
    '--output', eval_dir,
    '--pdf',
    '--training', os.path.join(output_dir, 'training_data.csv'),
    '--merged', os.path.join(output_dir, 'chm_export_1747162160.tif')
]

# print("\nRunning evaluation...")
# subprocess.run([str(arg) for arg in eval_cmd], check=True)
# print("All processing complete!")