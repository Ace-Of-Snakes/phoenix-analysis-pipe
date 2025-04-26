import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

def prepare_data(json_data):
    """Convert JSON data to pandas DataFrame and prepare for analysis."""
    # Convert JSON to DataFrame
    data = []
    for key, value in json_data.items():
        data.append({
            'teff': float(value['teff']),
            'logg': float(value['logg']),
            'quality_shifted': float(value['Quality_shifted']),
            'quality_unshifted': float(value['Quality_unshifted'])
        })
    
    df = pd.DataFrame(data)
    return df
def normalize_datasets(old_df, new_df, quality_column='quality_shifted'):
    """Normalize both datasets to make them comparable."""
    # Create scalers
    teff_scaler = MinMaxScaler()
    logg_scaler = MinMaxScaler()
    quality_scaler = MinMaxScaler()
    
    # Combine datasets for scaling
    combined_teff = np.concatenate([old_df['teff'], new_df['teff']])
    combined_logg = np.concatenate([old_df['logg'], new_df['logg']])
    combined_quality = np.concatenate([old_df[quality_column], new_df[quality_column]])
    
    # Fit scalers
    teff_scaler.fit(combined_teff.reshape(-1, 1))
    logg_scaler.fit(combined_logg.reshape(-1, 1))
    quality_scaler.fit(combined_quality.reshape(-1, 1))
    
    # Transform both datasets
    old_normalized = pd.DataFrame({
        'teff': teff_scaler.transform(old_df['teff'].values.reshape(-1, 1)).flatten(),
        'logg': logg_scaler.transform(old_df['logg'].values.reshape(-1, 1)).flatten(),
        'quality': quality_scaler.transform(old_df[quality_column].values.reshape(-1, 1)).flatten()
    })
    
    new_normalized = pd.DataFrame({
        'teff': teff_scaler.transform(new_df['teff'].values.reshape(-1, 1)).flatten(),
        'logg': logg_scaler.transform(new_df['logg'].values.reshape(-1, 1)).flatten(),
        'quality': quality_scaler.transform(new_df[quality_column].values.reshape(-1, 1)).flatten()
    })
    
    return old_normalized, new_normalized, (teff_scaler, logg_scaler, quality_scaler)

def calculate_shift_vector(old_df, new_df, quality_column='quality_shifted'):
    """Calculate the shift vector between datasets."""
    # Find minima in both datasets
    old_min_idx = old_df[quality_column].idxmin()
    new_min_idx = new_df[quality_column].idxmin()
    
    # Calculate shift vector
    shift = {
        'teff': new_df.loc[new_min_idx, 'teff'] - old_df.loc[old_min_idx, 'teff'],
        'logg': new_df.loc[new_min_idx, 'logg'] - old_df.loc[old_min_idx, 'logg'],
        'quality': new_df.loc[new_min_idx, quality_column] - old_df.loc[old_min_idx, quality_column]
    }
    
    return shift

def find_optimal_transformation(old_df, new_df, quality_column='quality_shifted'):
    """Find optimal transformation parameters between datasets with improved numerical stability."""
    def objective(params):
        try:
            scale_teff, scale_logg, shift_teff, shift_logg = params
            
            # Ensure parameters are within reasonable bounds
            if not (0.5 <= scale_teff <= 2.0 and 0.5 <= scale_logg <= 2.0):
                return 1e10
                
            # Apply transformation to old dataset
            transformed_old = old_df.copy()
            transformed_old['teff'] = old_df['teff'] * scale_teff + shift_teff
            transformed_old['logg'] = old_df['logg'] * scale_logg + shift_logg
            
            # Check if transformed points are within reasonable ranges
            teff_min, teff_max = min(new_df['teff']) - 500, max(new_df['teff']) + 500
            logg_min, logg_max = min(new_df['logg']) - 0.5, max(new_df['logg']) + 0.5
            
            if not (transformed_old['teff'].between(teff_min, teff_max).all() and 
                   transformed_old['logg'].between(logg_min, logg_max).all()):
                return 1e10
            
            # Interpolate transformed old quality values at new data points
            transformed_quality = griddata(
                (transformed_old['teff'], transformed_old['logg']),
                transformed_old[quality_column],
                (new_df['teff'], new_df['logg']),
                method='linear',  # Changed to linear for better stability
                fill_value=np.nan
            )
            
            # Calculate error only for valid points
            valid_mask = ~np.isnan(transformed_quality)
            if not np.any(valid_mask):
                return 1e10
                
            error = np.mean((transformed_quality[valid_mask] - 
                           new_df[quality_column].values[valid_mask])**2)
            
            # Smooth penalty terms
            scale_penalty = ((scale_teff - 1.0)**2 + (scale_logg - 1.0)**2) * 0.1
            shift_penalty = (abs(shift_teff/1000) + abs(shift_logg)) * 0.1
            
            total_error = error + scale_penalty + shift_penalty
            
            return float(total_error) if np.isfinite(total_error) else 1e10
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10
    
    # Initial guess based on data ranges
    teff_ratio = (new_df['teff'].max() - new_df['teff'].min()) / (old_df['teff'].max() - old_df['teff'].min())
    logg_ratio = (new_df['logg'].max() - new_df['logg'].min()) / (old_df['logg'].max() - old_df['logg'].min())
    teff_shift = new_df['teff'].mean() - old_df['teff'].mean()
    logg_shift = new_df['logg'].mean() - old_df['logg'].mean()
    
    x0 = [
        min(max(teff_ratio, 0.5), 2.0),
        min(max(logg_ratio, 0.5), 2.0),
        teff_shift,
        logg_shift
    ]
    
    # Set bounds for the optimization
    bounds = [
        (0.5, 2.0),     # scale_teff
        (0.5, 2.0),     # scale_logg
        (-500, 500),    # shift_teff
        (-0.5, 0.5)     # shift_logg
    ]
    
    # Try different optimization methods
    methods = ['Nelder-Mead', 'L-BFGS-B', 'SLSQP']
    best_result = None
    best_score = np.inf
    
    for method in methods:
        try:
            if method == 'Nelder-Mead':
                result = minimize(objective, x0, method=method, 
                               options={'maxiter': 1000, 'xatol': 1e-8})
            else:
                result = minimize(objective, x0, method=method, bounds=bounds,
                               options={'maxiter': 1000, 'ftol': 1e-8})
                
            if result.fun < best_score and np.all(np.isfinite(result.x)):
                best_result = result
                best_score = result.fun
                
        except Exception as e:
            print(f"Optimization failed with method {method}: {e}")
            continue
    
    if best_result is None:
        raise Exception("All optimization methods failed")
        
    return best_result.x

def apply_transformation(df, transformation_params, inverse=False):
    """Apply or inverse-apply transformation to dataset."""
    scale_teff, scale_logg, shift_teff, shift_logg = transformation_params
    
    if inverse:
        # Inverse transformation
        transformed = df.copy()
        transformed['teff'] = (df['teff'] - shift_teff) / scale_teff
        transformed['logg'] = (df['logg'] - shift_logg) / scale_logg
    else:
        # Forward transformation
        transformed = df.copy()
        transformed['teff'] = df['teff'] * scale_teff + shift_teff
        transformed['logg'] = df['logg'] * scale_logg + shift_logg
    
    return transformed

def analyze_shifted_data(old_df, new_df, quality_column='quality_shifted'):
    """Main function to analyze and compare shifted datasets."""
    # Normalize datasets
    old_norm, new_norm, scalers = normalize_datasets(old_df, new_df, quality_column)
    
    # Calculate basic shift vector
    shift_vector = calculate_shift_vector(old_df, new_df, quality_column)
    
    # Find optimal transformation
    transform_params = find_optimal_transformation(old_df, new_df, quality_column)
    
    # Apply transformation to get corrected prediction
    old_min_idx = old_df[quality_column].idxmin()
    old_minimum = {
        'teff': old_df.loc[old_min_idx, 'teff'],
        'logg': old_df.loc[old_min_idx, 'logg'],
        'quality': old_df.loc[old_min_idx, quality_column]
    }
    
    # Transform old minimum to new space
    transformed_minimum = {
        'teff': old_minimum['teff'] * transform_params[0] + transform_params[2],
        'logg': old_minimum['logg'] * transform_params[1] + transform_params[3],
    }
    
    results = {
        'shift_vector': shift_vector,
        'transformation_params': {
            'scale_teff': transform_params[0],
            'scale_logg': transform_params[1],
            'shift_teff': transform_params[2],
            'shift_logg': transform_params[3]
        },
        'old_minimum': old_minimum,
        'predicted_new_minimum': transformed_minimum
    }
    
    return results

import json

with open('results_oldh5.json') as f:
    old_data = json.load(f)

with open('results_cleaned_h5.json') as f:
    new_data = json.load(f)

# Convert your JSON data to DataFrames
old_df = prepare_data(old_data)
new_df = prepare_data(new_data)

# Analyze the shift
results = analyze_shifted_data(old_df, new_df)

# Get the corrected prediction for the new dataset minimum
predicted_minimum = results['predicted_new_minimum']

# Print the results
print(f"Shift vector: {results['shift_vector']}")

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def apply_shift_vector(old_minimum, shift_vector, data_bounds):
    """
    Apply shift vector to predict new minimum points and validate results.
    
    Parameters:
    - old_minimum: dict with 'teff', 'logg', 'quality' values
    - shift_vector: dict with 'teff', 'logg', 'quality' shifts
    - data_bounds: dict with 'teff_min', 'teff_max', 'logg_min', 'logg_max'
    
    Returns:
    - dict with predicted new minimum and validation info
    """
    # Apply shift vector to get predicted new minimum
    predicted_minimum = {
        'teff': old_minimum['teff'] + shift_vector['teff'],
        'logg': old_minimum['logg'] + shift_vector['logg'],
        'predicted_quality': old_minimum['quality'] + shift_vector['quality']
    }
    
    # Validate if prediction is within bounds
    is_valid = (
        data_bounds['teff_min'] <= predicted_minimum['teff'] <= data_bounds['teff_max'] and
        data_bounds['logg_min'] <= predicted_minimum['logg'] <= data_bounds['logg_max']
    )
    
    # Calculate confidence score based on distance from data boundaries
    teff_range = data_bounds['teff_max'] - data_bounds['teff_min']
    logg_range = data_bounds['logg_max'] - data_bounds['logg_min']
    
    teff_center_dist = abs(predicted_minimum['teff'] - (data_bounds['teff_max'] + data_bounds['teff_min'])/2)
    logg_center_dist = abs(predicted_minimum['logg'] - (data_bounds['logg_max'] + data_bounds['logg_min'])/2)
    
    confidence_score = 1.0 - (
        (teff_center_dist / teff_range + logg_center_dist / logg_range) / 2
    )
    
    return {
        'predicted_minimum': predicted_minimum,
        'is_valid': is_valid,
        'confidence_score': confidence_score
    }

def generate_search_grid(predicted_minimum, grid_size=5, teff_step=50, logg_step=0.05):
    """
    Generate a grid of points around the predicted minimum for fine-tuning.
    """
    teff_values = np.linspace(
        predicted_minimum['teff'] - (grid_size//2) * teff_step,
        predicted_minimum['teff'] + (grid_size//2) * teff_step,
        grid_size
    )
    
    logg_values = np.linspace(
        predicted_minimum['logg'] - (grid_size//2) * logg_step,
        predicted_minimum['logg'] + (grid_size//2) * logg_step,
        grid_size
    )
    
    return np.meshgrid(teff_values, logg_values)

def find_refined_minimum(df, predicted_minimum, quality_column='quality_shifted'):
    """
    Refine the predicted minimum using local interpolation.
    """
    # Generate fine grid around predicted minimum
    teff_grid, logg_grid = generate_search_grid(predicted_minimum)
    
    # Interpolate quality values on fine grid
    quality_grid = griddata(
        (df['teff'], df['logg']),
        df[quality_column],
        (teff_grid, logg_grid),
        method='cubic'
    )
    
    # Find minimum point on fine grid
    if np.any(~np.isnan(quality_grid)):
        min_idx = np.nanargmin(quality_grid)
        refined_minimum = {
            'teff': teff_grid.flat[min_idx],
            'logg': logg_grid.flat[min_idx],
            'quality': np.nanmin(quality_grid)
        }
    else:
        refined_minimum = predicted_minimum
    
    return refined_minimum


# Your existing minimum point from the old dataset
old_minimum = {
    'teff': 8441.508863600222,  # Your old minimum teff
    'logg': 4.448355653132949,  # Your old minimum logg
    'quality': 0.0012651018962123028  # Your old minimum quality
}

# Define data bounds for your new dataset
data_bounds = {
    'teff_min': 8100,  # Minimum teff in new dataset
    'teff_max': 8800,  # Maximum teff in new dataset
    'logg_min': 3.0,   # Minimum logg in new dataset
    'logg_max': 5.0    # Maximum logg in new dataset
}

# Apply the shift vector
result = apply_shift_vector(old_minimum, results['shift_vector'], data_bounds)

if result['is_valid']:
    print("Predicted new minimum:")
    print(f"Teff: {result['predicted_minimum']['teff']:.2f}")
    print(f"Logg: {result['predicted_minimum']['logg']:.2f}")
    print(f"Predicted Quality: {result['predicted_minimum']['predicted_quality']:.6f}")
    print(f"Confidence Score: {result['confidence_score']:.2f}")
    
    # If you have your new dataset loaded in a DataFrame (new_df)
    # You can refine the prediction:
    refined_minimum = find_refined_minimum(new_df, result['predicted_minimum'])
    print("\nRefined minimum after local search:")
    print(f"Teff: {refined_minimum['teff']:.2f}")
    print(f"Logg: {refined_minimum['logg']:.2f}")
    print(f"Quality: {refined_minimum['quality']:.6f}")
else:
    print("Warning: Predicted minimum is outside valid data bounds!")