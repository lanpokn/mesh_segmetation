import open3d as o3d
import numpy as np
import networkx as nx
import time
from mesh_io import *
from joblib import Parallel, delayed, parallel_backend
import math
#openmesh is only used to find normal!not change too much 
class MeshSeg:
    def __init__(self, mesh,normal = []):
        self.mesh = mesh
        self.normal = normal
        self.neighbor_triangles_index = self.find_neighboring_triangles()
        self.dual_graph = self.generate_dual_graph()
    
    #find neighbor is a costly process, so I use joblib to accelerate
    #it seems that I can't use joblib in dual graph, to be solved
    def find_neighboring_triangles(self):
        triangles = np.asarray(self.mesh.triangles)
        neighboring_triangles = []

        def process_triangle(i):
            vertices = set(triangles[i])
            current_neighboring_triangles = []
            for j, triangle in enumerate(triangles):
                if i == j:
                    continue
                shared_vertices = vertices.intersection(set(triangle))
                if len(shared_vertices) == 2:
                    current_neighboring_triangles.append(j)
            return current_neighboring_triangles

        num_cores = -1  # Use all available cores
        neighboring_triangles = Parallel(n_jobs=num_cores)(delayed(process_triangle)(i) for i in range(len(triangles)))

        return neighboring_triangles
    
    # dual graph, vertex i is same as triangle i, they have the same number
    def generate_dual_graph(self):
        triangles = np.asarray(self.mesh.triangles)
        G = nx.Graph()
        for i in range(len(triangles)):
            G.add_node(i)
        for i, triangle in enumerate(triangles):
            neighboring_triangles = self.neighbor_triangles_index[i]
            for neighbor_triangle_index in neighboring_triangles:
                neighbor_triangle = triangles[neighbor_triangle_index]
                distance = self.compute_distance(i, neighbor_triangle_index)
                G.add_edge(i, neighbor_triangle_index, weight=distance)
        return G

    #@staticmethod
    def compute_distance(self,triangle_index, neighbor_triangle_index,sigma = 0.5,n = 0.01):
        #get alpha
        cos_alpha = 0
        cos_alpha +=self.normal[triangle_index][0]*self.normal[neighbor_triangle_index][0]
        cos_alpha +=self.normal[triangle_index][1]*self.normal[neighbor_triangle_index][1]
        cos_alpha +=self.normal[triangle_index][2]*self.normal[neighbor_triangle_index][2]
        alpha = math.acos(cos_alpha)
        #get average point of triangle
        triangle = self.mesh.triangles[triangle_index]
        average_vertex = (mesh.vertices[triangle[0]] + mesh.vertices[triangle[1]] + mesh.vertices[triangle[2]]) / 3
        #get average point of neighbor triangle
        neighbor_triangle = self.mesh.triangles[neighbor_triangle_index]
        average_vertex_neighbor = (mesh.vertices[neighbor_triangle[0]] + mesh.vertices[neighbor_triangle[1]] + mesh.vertices[neighbor_triangle[2]]) / 3
        #TODO
        convex = 1
        #judge convex or concave:if concave,normal should intersect with another triangle
        if alpha == 0:
            convex = 1
        else:
            plane_normal = self.normal[triangle_index]
            # Get the ray_origin
            ray_origin = average_vertex
            #get ray_direction
            ray_direction = self.normal[triangle_index]
            #get plane_point
            plane_point = average_vertex_neighbor
            t = np.dot(plane_normal, plane_point - ray_origin) / np.dot(plane_normal, ray_direction)
            if t>0 :
                convex = 0
            else:
                convex = 1
        return 1
    def Segmentation_Base(self,group_indices):
        new_group_indices =[] 
        #determin K num by 
        return new_group_indices

    def Segmentation(self, triangle_indices_group):
        # Segment the mesh based on the provided triangle indices group
        new_groups = []
        # to avoid repetitive computation, generate dual mesh first

        for group_indices in triangle_indices_group:
            # Perform segmentation on each group of triangles
            # ... (your implementation here)
            new_group_indices = self.Segmentation_Base(group_indices)
            # Add the new groups to the list
            new_groups.extend(new_group_indices)

        return new_groups

#used for test
if __name__ == "__main__":
    # Create a more complicated mesh
    # vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]])
    # triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # # Create an instance of MeshSeg and pass the mesh
    # mesh_seg = MeshSeg(mesh)

    # # Access the dual graph
    # dual_graph = mesh_seg.dual_graph

    # # Print the nodes, edges, and weights of the dual graph
    # for u, v, data in dual_graph.edges(data=True):
    #     weight = data['weight']
    #     print(f"Edge ({u}, {v}), Weight: {weight}")
    
    ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"

    # Start the timer
    start_time = time.time()

    # Call the function to read the PLY file as a mesh
    # already cost 28 seconds, may try tqdm
    mesh = read_ply_as_mesh(ply_file)
    normal= get_normal_from_ply(ply_file)
    #show_mesh_and_normal(mesh,normal)
    mesh_seg_horse = MeshSeg(mesh,normal)

    # Calculate the total time taken
    total_time = time.time() - start_time

    # Print the total time
    print("Total time taken:", total_time, "seconds")
    