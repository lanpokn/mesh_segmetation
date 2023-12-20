import open3d as o3d
from mesh_io import *
# Specify the path to your PLY file
ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"

# Call the function to read the PLY file as a mesh
mesh = read_ply_as_mesh(ply_file)

# Specify the triangle indices for each group
triangle_indices_group = [['0', '1', '2'], [30, 40, 50], [600, 700, 800]]  # Example groups of triangle indices

# Call the function to render K segmentation with different colors for each group
mesh_with_colors = render_K_segmentation(mesh, triangle_indices_group)


# Visualize the mesh with colors
o3d.visualization.draw_geometries([mesh_with_colors])