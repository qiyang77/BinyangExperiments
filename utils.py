import numpy as np
import xarray as xr
from scipy.ndimage import rotate as scipy_rotate
from typing import List, Union
from scipy.spatial import cKDTree

def make_da(
            x_coords: List[Union[int, float]],
            y_coords: List[Union[int, float]],
            values: List[Union[int, float]],
            resolution: Union[int, float] = 30,
            max_dist: Union[int, float] = 10,
            name: str = "variable"
                ) -> xr.DataArray:
    """
    Grids sparse points using Nearest Neighbor, but sets cells to NaN 
    if the nearest point is further away than `max_dist`.
    """
    # 1. Setup the source points (Your real data)
    source_points = np.column_stack((x_coords, y_coords))
    source_values = np.array(values)

    # 2. Define the Target Grid
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # We use 'ceil' to ensure we cover the whole extent
    new_x = np.arange(x_min, x_max + resolution, resolution)
    new_y = np.arange(y_min, y_max + resolution, resolution)
    
    grid_x, grid_y = np.meshgrid(new_x, new_y)
    
    # Flatten the grid to (N, 2) for querying the Tree
    target_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # 3. Build the Tree and Query
    # cKDTree is very fast for spatial lookups
    tree = cKDTree(source_points)
    
    # query() returns:
    # dists: distance to the nearest neighbor
    # idxs: index of that neighbor in source_points
    dists, idxs = tree.query(target_points, k=1)

    # 4. Map values and Filter by Max Distance
    # Map the nearest neighbor values to the grid
    grid_values = source_values[idxs]
    
    # Where distance is too large, set to NaN
    grid_values[dists > max_dist] = np.nan

    # 5. Reshape back to grid shape (Y, X)
    final_data = grid_values.reshape(grid_y.shape)

    # 6. Create DataArray
    da = xr.DataArray(
        data=final_data,
        coords={"y": new_y, "x": new_x},
        dims=("y", "x"),
        name=name
    )

    return da

def valid2All(valid_indices, data_values):
    grid_num = 1715
    assert len(valid_indices) == len(data_values)
    full_data = np.full(grid_num, np.nan)
    for i, val in enumerate(data_values):
        if i < len(valid_indices):
            idx = valid_indices[i]
            if idx < grid_num:
                full_data[idx] = val
    return full_data

def get_processed_da(df_source, date_col, valid_idx, df_coords):
    data = np.array(df_source[date_col].tolist())
    data[data < 0] = np.nan 
    data_grids = valid2All(valid_idx, data)
    data_grids = np.array(data_grids)[(df_coords['idx']-1).tolist()]
    assert len(data_grids) == len(df_coords), 'grids are not aligned'
    da = make_da(
        x_coords=df_coords['x'],
        y_coords=df_coords['y'],
        values=data_grids
    )
    return da

def rotate_and_crop(da, angle=45):
    """
    Rotates the image array and crops NaNs to reduce whitespace.
    """
    # 1. Rotate the numpy array (order=0 means nearest neighbor, avoids NaN smearing)
    # We use cval=np.nan to fill the new background with NaNs
    rotated_data = scipy_rotate(da.values, angle, reshape=True, order=0, cval=np.nan)
    
    # 2. Crop to valid data (Remove all-NaN rows/cols)
    # Check where data is NOT nan
    mask = ~np.isnan(rotated_data)
    
    if not np.any(mask):
        return rotated_data # Return as is if empty
        
    # Find indices of valid data
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Slice the array
    cropped_data = rotated_data[rmin:rmax+1, cmin:cmax+1]
    
    return cropped_data