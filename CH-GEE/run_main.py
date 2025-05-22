import os
import subprocess
from pathlib import Path
import time

from utils import get_latest_file
from combine_heights import combine_heights_with_training

# Set parameters
def main(type: str):

    aoi_path = '../downloads/new_aoi.geojson'
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found at {aoi_path}")

    # Create output directories
    output_dir = 'chm_outputs'
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Build command for GEE model training and prediction
    gee_cmd = [
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
        # '--export-predictions',
        '--scale', '10',
        # '--resample', 'bicubic',
        '--ndvi-threshold', '0.35',
        '--mask-type', 'ALL',
    ]
    if type == 'data_preparation':
        # Convert all arguments to strings
        gee_cmd = [str(arg) for arg in gee_cmd]
        # Run the GEE model training and prediction
        print("Running GEE canopy height model...")
        subprocess.run(gee_cmd, check=True)
        # # Wait for downloaded files
        # print("Waiting for GEE exports to complete...")
        # time.sleep(60)  # Wait for files to be downloaded
    else:
        # Get the most recent training data, stack, and mask files
        training_file = get_latest_file(output_dir, 'training_data')
        stack_file = get_latest_file(output_dir, 'stack')
        mask_file = get_latest_file(output_dir, 'forestMask')  # Forest mask is optional

    ref_file = os.path.join(output_dir, 'dchm_09id4.tif')
    
    if type == 'height_analysis':
        combine_heights_with_training(output_dir, ref_file)

    if type =='train_predict':
        try:
            # Build command for local model training and prediction
            train_cmd = [
                'python', 'train_predict_map.py',
                '--training-data', training_file,
                '--stack', stack_file,
                '--mask', mask_file,  # Used as both quality mask and forest mask
                '--output-dir', output_dir,
                # '--output-filename', 'local_canopy_height_predictions.tif',
                '--test-size', '0.2',
                '--apply-forest-mask'  # Add flag to indicate mask should be used as forest mask
            ]
            # Run local training and prediction
            print("\nRunning local model training and prediction...")
            subprocess.run([str(arg) for arg in train_cmd], check=True)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure all required files have been exported from GEE before running local processing.")
    
    if type == 'evaluate':
        pred_file = get_latest_file(output_dir, 'predictCH') 
        # Run evaluation with PDF report generation
        eval_cmd = [
            'python', 'evaluate_predictions.py',
            '--pred', pred_file,
            '--ref', ref_file,
            '--output', eval_dir,
            '--pdf',
            '--training', training_file,
            '--merged', stack_file,
            '--forest-mask', mask_file
        ]
        print("\nRunning evaluation...")
        subprocess.run([str(arg) for arg in eval_cmd], check=True)
        print("All processing complete!")

if __name__ == "__main__":
    # Example usage
    main('data_preparation')
    # main('height_analysis')
    # main('train_predict')
    # main('evaluate')
    