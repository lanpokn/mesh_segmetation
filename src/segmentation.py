import open3d as o3d
import numpy as np
import networkx as nx
import time
from mesh_io import *
class MeshSeg:
    def __init__(self, mesh):
        self.mesh = mesh
        #neighbor_triangles_index is adjacency list
        self.neighbor_triangles_index = self.find_neighboring_triangles()
        #dual graph: vertex i is tri i in mesh, edge i to j is dist from i to j
        self.dual_graph = self.generate_dual_graph()
    def find_neighboring_triangles(self):
        triangles = np.asarray(self.mesh.triangles)
        neighboring_triangles = []
        
        for i, current_triangle in enumerate(triangles):
            vertices = set(current_triangle)
            current_neighboring_triangles = []
            
            for j, triangle in enumerate(triangles):
                if i == j:
                    continue
                
                shared_vertices = vertices.intersection(set(triangle))
                if len(shared_vertices) == 2:
                    current_neighboring_triangles.append(j)
            
            neighboring_triangles.append(current_neighboring_triangles)
        
        return neighboring_triangles
    def generate_dual_graph(self):
        # Extract vertex positions and triangle indices from the original mesh
        triangles = np.asarray(self.mesh.triangles)
        # Create a new NetworkX graph for the dual representation
        G = nx.Graph()
        # Add vertices (triangles) to the graph
        for i in range(len(triangles)):
            G.add_node(i)
        # Compute the edge weights based on the distances between neighboring triangles
        for i, triangle in enumerate(triangles):
            # Get the neighboring triangles for the current triangle
            neighboring_triangles = self.neighbor_triangles_index[i]
            for neighbor_triangle_index in neighboring_triangles:
                # Compute the distance between the current triangle and its neighbor
                neighbor_triangle = triangles[neighbor_triangle_index]
                distance = self.compute_distance(triangle, neighbor_triangle)
                # Add an edge between the current triangle and its neighbor with the computed distance as the weight
                G.add_edge(i, neighbor_triangle_index, weight=distance)
        return G
    #each triangle is a combine of three point
    #there will be two points is the same in neaby, otherwise the algor is wrong    
    def compute_distance(self,triangle,neighbor_triangle):
        #debug
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
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]])
    triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Create an instance of MeshSeg and pass the mesh
    mesh_seg = MeshSeg(mesh)

    # Access the dual graph
    dual_graph = mesh_seg.dual_graph

    # Print the nodes, edges, and weights of the dual graph
    for u, v, data in dual_graph.edges(data=True):
        weight = data['weight']
        print(f"Edge ({u}, {v}), Weight: {weight}")
    
    ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"

    # Start the timer
    start_time = time.time()

    # Call the function to read the PLY file as a mesh
    # already cost 28 seconds, may try tqdm
    mesh = read_ply_as_mesh(ply_file)
    mesh_seg_horse = MeshSeg(mesh)

    # Calculate the total time taken
    total_time = time.time() - start_time

    # Print the total time
    print("Total time taken:", total_time, "seconds")
    