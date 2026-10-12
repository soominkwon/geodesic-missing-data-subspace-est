import numpy as np


def closest_nonzero_indices(mat):
    # Get the number of rows and columns
    rows, cols = mat.shape
    
    # Initialize the output matrix with zeros
    output = np.zeros_like(mat)
    
    for i in range(rows):
        nonzero_indices = np.where(mat[i] != 0)[0]
        if len(nonzero_indices) == 0:
            continue  # If no non-zero elements in the row, skip to the next row
        
        for j in range(cols):
            if mat[i, j] == 1:
                output[i, j] = j
            else:
                # Find the closest nonzero index
                closest_index = nonzero_indices[np.abs(nonzero_indices - j).argmin()]
                output[i, j] = closest_index
    
    return output

def kspace_data_share(Ys, masks, dims):
    """
    Function to compute k-space data sharing via nearest neighbors.
    """

    # Initialize
    x, y = dims
    masks_concat = np.concatenate(masks, axis=1) # Concatenate masks together
    Ys_concat = np.concatenate(Ys, axis=1) # Concatenate observed data together
    missing_idx_mat = np.zeros((y, masks_concat.shape[1]))

    # Make matrix of missing row indices
    for i in range(masks_concat.shape[1]):
        mask = np.reshape(masks_concat[:, i], (x, y), order='F')
        missing_idx_mat[:, i] = mask[:, 0]

    assert np.any(np.all(missing_idx_mat == 0, axis=1)) == False # Make sure there is at least one neighbor

    fill_idx_mat = closest_nonzero_indices(missing_idx_mat) # Get indices for data sharing

    # Fill in missing using nearest neighbors
    shared_Y = np.zeros_like(Ys_concat, dtype=np.complex64)

    for i in range(Ys_concat.shape[1]):
        Y = np.reshape(Ys_concat[:, i], (x,y), order='F')
        fill_idx_row = fill_idx_mat[:, i]

        for j in range(fill_idx_row.shape[0]):
            Y = np.reshape(Ys_concat[:, i], (x,y), order='F')
            curr_Y = np.reshape(Ys_concat[:, int( fill_idx_row[j] )], (x,y), order='F')
            Y[j] = curr_Y[j]
        
        shared_Y[:, i] = np.reshape(Y, (-1, ), order='F')

    return shared_Y # Imputed data, but concatenated