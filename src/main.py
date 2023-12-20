import open3d as o3d
import time
from mesh_io import *
from segmentation import MeshSeg
# Specify the path to your PLY file
def test():
    ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"

    # Call the function to read the PLY file as a mesh
    mesh = read_ply_as_mesh(ply_file)

    # Specify the triangle indices for each group
    triangle_indices_group = [['0', '1', '2'], [30, 40, 50], [600, 700, 800]]  # Example groups of triangle indices

    # Call the function to render K segmentation with different colors for each group
    mesh_with_colors = render_K_segmentation(mesh, triangle_indices_group)


    # Visualize the mesh with colors
    o3d.visualization.draw_geometries([mesh_with_colors])
def main(ply_file):
    # Start the timer
    start_time = time.time()

    # Call the function to read the PLY file as a mesh
    # already cost 28 seconds, may try tqdm
    mesh = read_ply_as_mesh(ply_file)
    normal= get_normal_from_ply(ply_file)
    #show_mesh_and_normal(mesh,normal)
    mesh_seg_horse = MeshSeg(mesh,normal)
    triangle_indices_list =  mesh_seg_horse.Segementation_recursion(1)
    total_items = sum(len(sublist) for sublist in triangle_indices_list)

    print(f"The 2D list has {total_items} items.")
    # Calculate the total time taken
    total_time = time.time() - start_time
    print("Total time taken:", total_time, "seconds")
    mesh_with_colors = render_K_segmentation(mesh, triangle_indices_list)


    # Visualize the mesh with colors
    o3d.visualization.draw_geometries([mesh_with_colors])
main("C:/Users/hhq/Desktop/mesh_segmentation/data/cat.ply")