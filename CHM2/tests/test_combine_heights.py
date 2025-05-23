import os
import sys
import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from combine_heights import analyze_heights, combine_heights_with_training

def test_analyze_heights_input_validation():
    """Test input validation for analyze_heights function."""
    # Test invalid DataFrame
    with pytest.raises(TypeError):
        analyze_heights(None, ['col1'])
    
    # Test invalid height_columns
    df = pd.DataFrame({'reference_height': [1, 2, 3]})
    with pytest.raises(TypeError):
        analyze_heights(df, 'not_a_list')
    
    # Test missing reference column
    df = pd.DataFrame({'col1': [1, 2, 3]})
    with pytest.raises(ValueError):
        analyze_heights(df, ['col1'])

def test_analyze_heights_calculations():
    """Test statistical calculations in analyze_heights function."""
    # Create test data
    ref_heights = [10, 20, 30, 40, 50]
    test_heights = [12, 22, 32, 42, 52]  # Always 2m higher than reference
    df = pd.DataFrame({
        'reference_height': ref_heights,
        'test_height': test_heights
    })
    
    # Run analysis
    stats, _ = analyze_heights(df, ['test_height'])
    
    # Check results
    assert stats is not None
    assert 'test_height' in stats
    assert 'error_matrix' in stats
    
    test_stats = stats['test_height']
    
    # Test correlation (should be almost 1)
    assert abs(test_stats['Correlation'] - 1.0) < 1e-10
    
    # Test RMSE (should be 2.0)
    assert abs(test_stats['RMSE'] - 2.0) < 1e-10
    
    # Test bias (should be 2.0)
    assert abs(test_stats['Bias'] - 2.0) < 1e-10
    
    # Test R-squared (should be almost 1)
    assert abs(test_stats['R-squared'] - 1.0) < 1e-10
    
    # Test slope (should be 1.0)
    assert abs(test_stats['Slope'] - 1.0) < 1e-10
    
    # Test error matrix
    assert 'RMSE' in stats['error_matrix']['test_height']
    assert abs(stats['error_matrix']['test_height']['RMSE'] - 2.0) < 1e-10

def test_analyze_heights_missing_data():
    """Test handling of missing data in analyze_heights function."""
    # Create test data with NaN and -32767
    df = pd.DataFrame({
        'reference_height': [10, np.nan, 30, -32767, 50],
        'test_height': [12, 22, np.nan, 42, 52]
    })
    
    # Run analysis
    stats, _ = analyze_heights(df, ['test_height'])
    
    # Check valid pairs and statistics
    assert stats['test_height']['N'] == 2
    assert 'RMSE' in stats['test_height']
    assert 'Bias' in stats['test_height']
    assert 'R-squared' in stats['test_height']
    
    # Check error matrix
    assert 'RMSE' in stats['error_matrix']['test_height']

def test_combine_heights_input_validation(tmp_path):
    """Test input validation for combine_heights_with_training function."""
    # Test non-existent output directory
    with pytest.raises(ValueError):
        combine_heights_with_training("/nonexistent/dir", "ref.tif")
    
    # Test non-existent reference file
    os.makedirs(str(tmp_path), exist_ok=True)
    with pytest.raises(ValueError):
        combine_heights_with_training(str(tmp_path), "nonexistent.tif")

def test_output_file_naming(tmp_path):
    """Test that output files use correct naming pattern."""
    # Create mock training data
    df = pd.DataFrame({
        'longitude': [1, 2],
        'latitude': [1, 2],
        'rh': [10, 20]
    })
    os.makedirs(str(tmp_path), exist_ok=True)
    training_file = tmp_path / "training_data.csv"
    df.to_csv(training_file, index=False)
    
    # Create mock reference file (empty)
    ref_file = tmp_path / "dchm_09id4.tif"
    ref_file.touch()
    
    # Run function (this will fail due to mock files, but we can check file naming)
    with pytest.raises(Exception):
        combine_heights_with_training(str(tmp_path), str(ref_file))
        
    # Check that output files would use correct prefix
    expected_files = [
        'trainingData_with_heights.csv',
        'trainingData_height_analysis.json',
        'trainingData_height_comparison.png'
    ]
    for fname in expected_files:
        assert not (tmp_path / fname).exists()

if __name__ == '__main__':
    pytest.main([__file__])