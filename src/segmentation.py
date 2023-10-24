import open3d as o3d
import numpy as np
import networkx as nx

class MeshSeg:
    def generate_dual_graph(mesh):
        # Extract vertex positions and triangle indices from the original mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Create a new NetworkX graph for the dual representation
        G = nx.Graph()
        
        # Add vertices (triangles) to the graph
        for i in range(len(triangles)):
            G.add_node(i)
        
        # Compute the edge weights based on the distances between neighboring triangles
        for i, triangle in enumerate(triangles):
            #TODO
            # distance should be right, and G's i Node is also mesh.triangle[i]!
            
            # Compute the distance between the neighboring triangles
            distance = compute_distance(neighbor_triangle, next_neighbor_triangle, vertices)
            
            # Add an edge between the current triangle and its neighbor with the computed distance as the weight
            G.add_edge(i, neighbor_triangle, weight=distance)
        
        return G
    def Segmentation_Base(mesh,group_indices):
        new_group_indices =[] 
        #determin K num by 
        return new_group_indices

    def Segmentation(mesh, triangle_indices_group):
        # Segment the mesh based on the provided triangle indices group
        new_groups = []
        # to avoid repetitive computation, generate dual mesh first

        for group_indices in triangle_indices_group:
            # Perform segmentation on each group of triangles
            # ... (your implementation here)
            new_group_indices = Segmentation_Base(mesh,group_indices)
            # Add the new groups to the list
            new_groups.extend(new_group_indices)

        return new_groups