#input,output,render
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt 
import openmesh as om
def read_ply_as_mesh(ply_file):
    # Read the PLY file as a mesh
    mesh = o3d.io.read_triangle_mesh(ply_file)
    return mesh
def read_ply_as_openmesh(ply_file):
    # Create an OpenMesh mesh
    mesh = om.TriMesh()

    # Read the PLY file
    mesh = om.read_trimesh(ply_file)

    return mesh
def get_normal_from_ply(ply_file):
    mesh= read_ply_as_openmesh(ply_file)
    mesh.request_face_normals()
    mesh.update_normals()
    normal = mesh.face_normals()
    return normal
def show_mesh_and_normal(mesh, normal):
    # 创建mesh和法向量的可视化对象
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = mesh.vertices
    mesh_o3d.triangles = mesh.triangles
    mesh_o3d.triangle_normals = o3d.utility.Vector3dVector(normal)
    mesh_o3d.paint_uniform_color([0.5, 0.5, 0.5])
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.compute_triangle_normals()
    # 将法向量可视化为黑色箭头
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []
    for i in range(len(mesh_o3d.triangles)):
        triangle = mesh_o3d.triangles[i]
        normal = mesh_o3d.triangle_normals[i]
        center = (mesh_o3d.vertices[triangle[0]] + mesh_o3d.vertices[triangle[1]] + mesh_o3d.vertices[triangle[2]]) / 3
        points.append(center)
        points.append(center + 0.1 * normal)
        lines.append([2 * i, 2 * i + 1])
        colors.append([0, 0, 0])
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 可视化
    o3d.visualization.draw_geometries([mesh_o3d, line_set])


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
def render_K_segmentation(mesh_om, triangle_indices_group):
    # Generate a color palette of length K
    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_om.points())
    mesh.triangles = o3d.utility.Vector3iVector(mesh_om.face_vertex_indices())
    
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
# def render_K_segmentation_om(mesh, triangle_indices_group):
#     # Generate a color palette of length K
#     K = len(triangle_indices_group)
#     color_palette = generate_color_palette(K)

#     # Create an array to store the vertex colors
#     vertex_colors = np.zeros((len(mesh.vertices), 3))

#     # Assign colors to each group of triangles
#     for i, indices in enumerate(triangle_indices_group):
#         color = color_palette[i % K]  # Cycle through the color palette
#         for index in indices:
#             vertex_indices = mesh.triangles[index]
#             vertex_colors[vertex_indices] = color

#     # Assign the vertex colors to the mesh
#     mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

#     # Visualize the mesh with different colors for each group
#     #o3d.visualization.draw_geometries([mesh])
#     return mesh
if __name__ == "__main__":
    ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"
    mesh = read_ply_as_openmesh(ply_file)
    mesh.request_face_normals()
    mesh.update_normals()
    normal = mesh.face_normals()
    print(normal[1][0])