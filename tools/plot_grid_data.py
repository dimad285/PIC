#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def read_grid_data(filename):
    """
    Read grid data from a text file created by the Grid2D.save_to_txt method.
    
    Parameters:
    ----------
    filename : str
        Path to the input text file.
        
    Returns:
    -------
    data : numpy.ndarray
        Array containing all data from the file.
    header : list
        List of column names.
    grid_shape : tuple
        Tuple containing (nx, ny) dimensions of the grid.
    """
    # Read header to get column names
    with open(filename, 'r') as f:
        header_line = f.readline().strip()
    
    # Check if header starts with a comment character
    if header_line.startswith('#'):
        header_line = header_line[1:].strip()
    
    # Split header into column names
    header = header_line.split()
    
    # Read data from file
    data = np.loadtxt(filename)
    
    # Determine grid dimensions from unique x and y values
    unique_x = np.unique(data[:, 0])
    unique_y = np.unique(data[:, 1])
    nx = len(unique_x)
    ny = len(unique_y)
    
    return data, header, (nx, ny)

def reshape_to_grid(data, column_idx, grid_shape):
    """
    Reshape a column of data into a 2D grid.
    
    Parameters:
    ----------
    data : numpy.ndarray
        Array containing all data from the file.
    column_idx : int
        Index of the column to reshape.
    grid_shape : tuple
        Tuple containing (nx, ny) dimensions of the grid.
        
    Returns:
    -------
    grid : numpy.ndarray
        2D array containing the reshaped data.
    x_grid : numpy.ndarray
        2D array of x coordinates.
    y_grid : numpy.ndarray
        2D array of y coordinates.
    """
    nx, ny = grid_shape
    
    # Extract x, y, and target data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, column_idx]
    
    # Reshape data to 2D grid
    x_grid = x.reshape(ny, nx)
    y_grid = y.reshape(ny, nx)
    z_grid = z.reshape(ny, nx)
    
    return z_grid, x_grid, y_grid

def plot_field_3d_surface(data, header, grid_shape, field_name, output_dir=None, cmap='viridis'):
    """
    Create a 3D surface plot of a scalar field.
    
    Parameters:
    ----------
    data : numpy.ndarray
        Array containing all data from the file.
    header : list
        List of column names.
    grid_shape : tuple
        Tuple containing (nx, ny) dimensions of the grid.
    field_name : str
        Name of the field to plot.
    output_dir : str, optional
        Directory to save the plot. If None, the plot is displayed instead.
    cmap : str
        Colormap to use for the plot.
    """
    # Check if field_name is in header
    if field_name not in header:
        print(f"Error: Field '{field_name}' not found in data. Available fields are: {header[2:]}")
        return
    
    # Get column index for the field
    column_idx = header.index(field_name)
    
    # Reshape data to 2D grid
    field_grid, x_grid, y_grid = reshape_to_grid(data, column_idx, grid_shape)
    
    # Create figure with 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(x_grid, y_grid, field_grid, cmap=cmap, 
                           linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=field_name)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(field_name)
    ax.set_title(f"3D Surface Plot of {field_name}")
    
    # Adjust the view angle
    ax.view_init(elev=30, azim=45)
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{field_name}_3d_surface.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_field(data, header, grid_shape, field_name, output_dir=None, plot_type='contourf'):
    """
    Plot a field from the grid data.
    
    Parameters:
    ----------
    data : numpy.ndarray
        Array containing all data from the file.
    header : list
        List of column names.
    grid_shape : tuple
        Tuple containing (nx, ny) dimensions of the grid.
    field_name : str
        Name of the field to plot.
    output_dir : str, optional
        Directory to save the plot. If None, the plot is displayed instead.
    plot_type : str
        Type of plot to create ('contourf', 'contour', 'surface').
    """
    # Check if field_name is in header
    if field_name not in header:
        print(f"Error: Field '{field_name}' not found in data. Available fields are: {header[2:]}")
        return
    
    # For 3D surface plot, use the specialized function
    if plot_type == 'surface':
        plot_field_3d_surface(data, header, grid_shape, field_name, output_dir)
        return
    
    # Get column index for the field
    column_idx = header.index(field_name)
    
    # Reshape data to 2D grid
    field_grid, x_grid, y_grid = reshape_to_grid(data, column_idx, grid_shape)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create plot based on plot_type
    if plot_type == 'contourf':
        contour = plt.contourf(x_grid, y_grid, field_grid, 100, cmap='viridis')
        plt.colorbar(contour, label=field_name)
    elif plot_type == 'contour':
        contour = plt.contour(x_grid, y_grid, field_grid, 20, colors='k')
        plt.clabel(contour, inline=True, fontsize=8)
        plt.contourf(x_grid, y_grid, field_grid, 100, cmap='viridis', alpha=0.7)
        plt.colorbar(label=field_name)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{field_name} - Grid Plot")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{field_name}_{plot_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_vector_field(data, header, grid_shape, x_component, y_component, output_dir=None, plot_type='quiver'):
    """
    Plot a vector field from the grid data.
    
    Parameters:
    ----------
    data : numpy.ndarray
        Array containing all data from the file.
    header : list
        List of column names.
    grid_shape : tuple
        Tuple containing (nx, ny) dimensions of the grid.
    x_component : str
        Name of the x component field.
    y_component : str
        Name of the y component field.
    output_dir : str, optional
        Directory to save the plot. If None, the plot is displayed instead.
    plot_type : str
        Type of plot to create ('quiver' or 'streamplot').
    """
    # Check if fields are in header
    if x_component not in header or y_component not in header:
        print(f"Error: Fields '{x_component}' or '{y_component}' not found in data. Available fields are: {header[2:]}")
        return
    
    # Get column indices for the fields
    x_idx = header.index(x_component)
    y_idx = header.index(y_component)
    
    # Reshape data to 2D grid
    x_grid, _, _ = reshape_to_grid(data, 0, grid_shape)  # Just to get the grid structure
    y_grid, _, _ = reshape_to_grid(data, 1, grid_shape)
    u_grid, _, _ = reshape_to_grid(data, x_idx, grid_shape)
    v_grid, _, _ = reshape_to_grid(data, y_idx, grid_shape)
    
    # Calculate magnitude for color mapping
    magnitude = np.sqrt(u_grid**2 + v_grid**2)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create plot based on plot_type
    if plot_type == 'quiver':
        # Reduce number of arrows for better visibility
        skip = (slice(None, None, 3), slice(None, None, 3))
        q = plt.quiver(x_grid[skip], y_grid[skip], u_grid[skip], v_grid[skip], 
                       magnitude[skip], cmap='viridis', scale=30, pivot='mid')
        plt.colorbar(q, label='Magnitude')
        
    elif plot_type == 'streamplot':
        # Streamplot requires evenly spaced grid
        x_unique = np.unique(x_grid[0, :])
        y_unique = np.unique(y_grid[:, 0])
        strm = plt.streamplot(x_unique, y_unique, u_grid, v_grid, 
                             color=magnitude, cmap='viridis', linewidth=1, density=1.5)
        plt.colorbar(strm.lines, label='Magnitude')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{x_component}/{y_component} - Vector Field")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{x_component}_{y_component}_{plot_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot grid data from text file.')
    parser.add_argument('filename', help='Path to the input text file')
    parser.add_argument('--field', '-f', help='Name of the field to plot')
    parser.add_argument('--x-component', help='X component of vector field')
    parser.add_argument('--y-component', help='Y component of vector field')
    parser.add_argument('--output-dir', '-o', help='Directory to save plots')
    parser.add_argument('--plot-type', '-p', 
                        choices=['contourf', 'contour', 'quiver', 'streamplot', 'surface'], 
                        default='contourf', help='Type of plot to create')
    parser.add_argument('--list-fields', '-l', action='store_true', help='List available fields in the file')
    parser.add_argument('--colormap', '-c', default='viridis', help='Colormap to use for plots')
    
    args = parser.parse_args()
    
    # Read data from file
    data, header, grid_shape = read_grid_data(args.filename)
    
    # List available fields
    if args.list_fields:
        print(f"Available fields in {args.filename}:")
        for field in header[2:]:  # Skip x and y columns
            print(f"  - {field}")
        return
    
    # Plot vector field if x and y components are provided
    if args.x_component and args.y_component:
        if args.plot_type in ['quiver', 'streamplot']:
            plot_vector_field(data, header, grid_shape, args.x_component, args.y_component, 
                              args.output_dir, args.plot_type)
        else:
            print(f"Warning: Plot type '{args.plot_type}' not valid for vector fields. Using 'quiver' instead.")
            plot_vector_field(data, header, grid_shape, args.x_component, args.y_component, 
                              args.output_dir, 'quiver')
    
    # Plot scalar field if field is provided
    elif args.field:
        plot_field(data, header, grid_shape, args.field, args.output_dir, args.plot_type)
    
    # If no specific field is provided, plot all scalar fields
    else:
        for field in header[2:]:  # Skip x and y columns
            if args.plot_type == 'surface':
                plot_field_3d_surface(data, header, grid_shape, field, args.output_dir, args.colormap)
            else:
                plot_field(data, header, grid_shape, field, args.output_dir, 'contourf')

if __name__ == '__main__':
    main()