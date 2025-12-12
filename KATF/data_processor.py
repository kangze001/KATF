import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def length_align(X, seq_len):
    """
    Aligns the length of sequences (X) to the specified sequence length (seq_len).
    It truncates longer sequences and zero-pads shorter ones.
    
    Parameters:
    X (ndarray): Input sequences.
    seq_len (int): Desired sequence length.

    Returns:
    ndarray: Aligned sequences.
    """
    input_len = X.shape[-1]
    
    if seq_len < input_len:
        # Truncate the sequence
        X = X[..., :seq_len]
        
    if seq_len > input_len:
        # Calculate padding length and pad with zeros
        padding_num = seq_len - input_len
        # Create pad_width: (0, 0) for all but the last dimension
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)
        
    return X

def process_MSTF_multi_scale(index, sequence, scale_params, adaptive_scales=False):
    """
    Processes a single sequence to generate Multi-Scale Time-based Amplitude 
    (MSTF/TAM) features.
    
    Parameters:
    index (int): Sequence index.
    sequence (ndarray): Input sequence (time values).
    scale_params (list): List of scale parameters, where each element is (max_time, bins).
    adaptive_scales (bool): Not used in this version but kept for signature consistency.
    
    Returns:
    tuple: (index, features)
    """
    
    total_bins = sum(bins for _, bins in scale_params)
    # Features matrix: 2 rows per scale (positive/negative), total_bins columns
    features = np.zeros((2 * len(scale_params), total_bins))
    
    bin_offset = 0
    
    for scale_idx, (max_time, bins) in enumerate(scale_params):
        temp_feature = np.zeros((2, bins))
        
        for pack in sequence:
            if pack == 0: # Skip zero-padded values
                continue
            
            # 0 for positive flow (forward), 1 for negative flow (backward)
            sign = 0 if pack > 0 else 1 
            abs_time = abs(pack)
            
            # Calculate bin index (linear interpolation)
            if abs_time >= max_time:
                bin_idx = bins - 1 # Assign to the last bin
            else:
                # Map abs_time from [0, max_time) to bin_idx in [0, bins-1)
                bin_idx = int(abs_time * (bins - 1) / max_time)
            
            temp_feature[sign, bin_idx] += 1
        
        # Insert current scale's features into the final matrix
        row_offset = scale_idx * 2
        
        # The assignment below assumes that the features are concatenated row-wise.
        # However, the original code shows concatenation in the column dimension (bin dimension)
        # while keeping the 2 feature rows (positive/negative) per scale.
        
        # Correctly insert features into the TAM structure:
        features[row_offset:row_offset+2, bin_offset:bin_offset+bins] = temp_feature
        
        # Update bin offset
        bin_offset += bins
    
    return index, features

def extract_MSTF(sequences, scale_configs=None, adaptive_scales=False, num_workers=30):
    """
    Extracts Multi-Scale TAM features from a batch of sequences using multiprocessing.
    
    Parameters:
    sequences (ndarray): Input sequences array (Num_Sequences, Length).
    scale_configs (list): List of scale configurations [(max_time, bins), ...].
    adaptive_scales (bool): Flag for adaptive scaling (currently unused).
    num_workers (int): Number of parallel processes.
    
    Returns:
    ndarray: Extracted TAM features (Num_Sequences, Num_Features, Total_Bins).
    """
    # Set default scale configuration
    if scale_configs is None:
        scale_configs = [(15, 1500), (15, 300), (15, 150)]
    
    num_sequences = sequences.shape[0]
    
    # Calculate total feature dimension
    total_bins = sum(bins for _, bins in scale_configs)
    num_features = 2 * len(scale_configs) # 2 rows (positive/negative) per scale
    
    # Initialize feature tensor
    TAM = np.zeros((num_sequences, num_features, total_bins))
    
    # Use multiprocessing to process sequences in parallel
    max_workers = min(num_workers, num_sequences)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index in range(num_sequences):
            future = executor.submit(
                process_MSTF_multi_scale,
                index, 
                sequences[index],
                scale_configs,
                adaptive_scales
            )
            futures.append(future)
        
        # Use tqdm to track progress of completed futures
        with tqdm(total=len(futures), desc="Extracting TAM Features") as pbar:
            for future in as_completed(futures):
                index, result = future.result()
                TAM[index] = result
                pbar.update(1)
    
    return TAM