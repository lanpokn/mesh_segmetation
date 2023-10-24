#input,output,render
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def read_ply_as_mesh(ply_file):
    # Read the PLY file as a mesh
    mesh = o3d.io.read_triangle_mesh(ply_file)
    return mesh

#only used in 2-seg
def assign_triangle_colors(mesh, triangle_indices, color):
    # Create a default color array for all vertices
    default_color = [0.5, 0.5, 0.5]  # Default color (gray)
    vertex_colors = np.tile(default_color, (len(mesh.vertices), 1))

    # Assign the specified color to the vertices of the specified triangles
    for triangle_index in triangle_indices:
        vertex_indices = mesh.triangles[triangle_index]
        vertex_colors[vertex_indices] = color

    # Assign the vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh
def hsv_to_rgb(h, s, v):
    # Convert HSV to RGB
    if s == 0.0:
        return v, v, v
    i = int(h * 6.)
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def generate_color_palette(K):
    # Generate a color palette of length K
    color_palette = []
    for i in range(K):
        hue = i * (360 / K)  # Vary the hue value for each color
        rgb = hsv_to_rgb(hue / 360, 1, 1)  # Convert HSV to RGB
        color_palette.append(rgb)

    return color_palette

#mesh is the totol mesh data
#triangle_indices_group have K triangle_indices，K is unknown
def render_K_segmentation(mesh, triangle_indices_group):
    # Generate a color palette of length K
    K = len(triangle_indices_group)
    color_palette = generate_color_palette(K)

    # Create an array to store the vertex colors
    vertex_colors = np.zeros((len(mesh.vertices), 3))

    # Assign colors to each group of triangles
    for i, indices in enumerate(triangle_indices_group):
        color = color_palette[i % K]  # Cycle through the color palette
        for index in indices:
            vertex_indices = mesh.triangles[index]
            vertex_colors[vertex_indices] = color

    # Assign the vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Visualize the mesh with different colors for each group
    #o3d.visualization.draw_geometries([mesh])
    return mesh
