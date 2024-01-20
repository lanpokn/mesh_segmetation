#hhq, 2023-12
#TODO the pachage is a little ugly, try to abstract?
import open3d as o3d
import numpy as np
import igraph as ig
import time
from mesh_io import *
from joblib import Parallel, delayed, parallel_backend
import math
import random
#openmesh is only used to find normal!not change too much 
class MeshSeg:
    def __init__(self, mesh,normal = []):
        self.mesh = mesh
        self.normal = normal
        self.neighbor_triangles_index = self.find_neighboring_triangles()
        self.dual_graph = self.generate_dual_graph()
        self.firsttime = False
    
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
    def generate_dual_graph(self, sigma=0.1):
        triangles = np.asarray(self.mesh.triangles)
        G = ig.Graph()
        ang_dist_list = []
        geo_dist_list = []
        ang_dist_graph = [[0 for _ in range(len(triangles))] for _ in range(len(triangles))]
        geo_dist_graph = [[0 for _ in range(len(triangles))] for _ in range(len(triangles))]
        
        for i in range(len(triangles)):
            G.add_vertex(str(i))
        
        for i, triangle in enumerate(triangles):
            neighboring_triangles = self.neighbor_triangles_index[i]
            for neighbor_triangle_index in neighboring_triangles:
                ang_dist, geo_dist = self.compute_distance(i, neighbor_triangle_index)
                ang_dist_graph[i][neighbor_triangle_index] = ang_dist
                geo_dist_graph[i][neighbor_triangle_index] = geo_dist
                ang_dist_list.append(ang_dist)
                geo_dist_list.append(geo_dist)

        # Enumerate again to give value to G
        avg_ang_dist = (sum(ang_dist_list)) / (len(ang_dist_list))
        avg_geo_dist = (sum(geo_dist_list)) / (len(geo_dist_list))

        for i, triangle in enumerate(triangles):
            neighboring_triangles = self.neighbor_triangles_index[i]
            for neighbor_triangle_index in neighboring_triangles:
                distance = sigma * ang_dist_graph[i][neighbor_triangle_index] / avg_ang_dist
                distance += (1 - sigma) * geo_dist_graph[i][neighbor_triangle_index] / avg_geo_dist
                G.add_edge(str(i), str(neighbor_triangle_index), weight=distance)

        return G
    def draw_dual_graph(self):
        ig.plot(self.dual_graph, layout="kk", vertex_size=2, vertex_label=None)
        plt.show()
        return


    #@staticmethod
    def compute_distance(self,triangle_index, neighbor_triangle_index,n = 0.1):
        #get alpha
        cos_alpha = 0
        cos_alpha +=self.normal[triangle_index][0]*self.normal[neighbor_triangle_index][0]
        cos_alpha +=self.normal[triangle_index][1]*self.normal[neighbor_triangle_index][1]
        cos_alpha +=self.normal[triangle_index][2]*self.normal[neighbor_triangle_index][2]
        #alpha = math.acos(cos_alpha)
        #get average point of triangle
        triangle = self.mesh.triangles[triangle_index]
        average_vertex = (self.mesh.vertices[triangle[0]] + self.mesh.vertices[triangle[1]] + self.mesh.vertices[triangle[2]]) / 3
        #get average point of neighbor triangle
        neighbor_triangle = self.mesh.triangles[neighbor_triangle_index]
        average_vertex_neighbor = (self.mesh.vertices[neighbor_triangle[0]] + self.mesh.vertices[neighbor_triangle[1]] + self.mesh.vertices[neighbor_triangle[2]]) / 3
        #TODO
        convex = 1
        #judge convex or concave:if concave,normal should intersect with another triangle
        if cos_alpha == 1:
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
        if convex == 1:
            ang_dist = n*(1-cos_alpha)
        else:
            ang_dist = 1*(1-cos_alpha)
        geo_dist = 0
        geo_dist +=average_vertex[0]*average_vertex_neighbor[0]
        geo_dist +=average_vertex[1]*average_vertex_neighbor[1]
        geo_dist +=average_vertex[2]*average_vertex_neighbor[2]
        return ang_dist,geo_dist
    def Segmentation_Base(self,triangle_indices):
        new_group_indices =[] 
        #determin K num by 
        seed_node_list = self.Determine_kseed(triangle_indices)
        new_group_indices = self.Calculate_groups(seed_node_list,triangle_indices)
        return new_group_indices

    def Calculate_groups(self,seed_node_list,triangle_indices, e=0):
        #bug, after generate sub
        G = self.dual_graph.subgraph(triangle_indices)
        # Create a list to store the groups
        new_group_indices = []
        #add seed to new_group_indices
        #TODO change to numpy may accelerate?
        for i in range(0,len(seed_node_list)):
            temp = [seed_node_list[i]]
            new_group_indices.append(temp) 
        # Initialize the 3D list for fuzzy_group_indices
        fuzzy_group_indices = [[[] for _ in range(len(seed_node_list))] for _ in range(len(seed_node_list))]
    
        # Iterate over each node in the graph
        for node in G.vs:
            # Calculate the distances from the node to each seed node
            # when
            distance = np.zeros(len(seed_node_list))
            for i in range(0,len(seed_node_list)):
                distance[i] = G.distances(node['name'], seed_node_list[i])[0][0]

            # Find the indices that would sort the 'distance' array
            sorted_indices = np.argsort(distance)

            # Find the minimum distance and its corresponding index
            min_distance = distance[sorted_indices[0]]
            #avoid finding it self
            if min_distance == 0:
                continue
            min_index = sorted_indices[0]
            probabilty = (1.0 / min_distance) / (np.sum(1.0 / distance))
            # Find the second minimum distance and its corresponding index
            second_min_index = sorted_indices[1]
            #find threshold
            threshold = 1/len(seed_node_list) + e
            if probabilty > threshold:
                new_group_indices[min_index].append(node['name'])
            else:
                fuzzy_group_indices[min_index][second_min_index].append(node['name'])
                fuzzy_group_indices[second_min_index][min_index].append(node['name'])
        
        # Apply Maximum Flow Minimum Cut for the fuzzy part
        def is_connected(original_graph, group1_name_list, group2_name_list):
            # Create a subgraph containing group1 vertices
            # subgraph_group1 = original_graph.subgraph(group1_name_list)
            # subgraph_group2 = original_graph.subgraph(group2_name_list)
            # Check if group1 is connected to the i-th new group
            if any(original_graph.are_connected(vertex, other_vertex) for vertex in group1_name_list for other_vertex in group2_name_list):
                return True

            # If no connection is found
            return False
        #TODO, mincut otfen fail, need further investigate
        for i in range(len(seed_node_list) - 1):
            for j in range(i + 1, len(seed_node_list)):
                # Create a directed graph based on the edges between the two groups
                if fuzzy_group_indices[i][j] == []:
                    continue
                cut_graph = G.subgraph(fuzzy_group_indices[min_index][second_min_index])
                if cut_graph.ecount()<1:
                    print("No edge in fuzzy part, give all fuzzy part to one side")
                    new_group_indices[i].extend(fuzzy_group_indices[i][j])
                    continue
                # Calculate the maximum flow minimum cut
                cut = cut_graph.mincut()
                if not cut.partition:
                    print("No valid mincut found, give all fuzzy part to one side")
                    new_group_indices[i].extend(fuzzy_group_indices[i][j])
                    # Handle this situation as needed
                else:
                    # Get the source and sink sets from the mincut result
                    source_set = cut_graph.vs.select(cut.partition[0])
                    sink_set = cut_graph.vs.select(cut.partition[1])

                    # Update the fuzzy_group_indices based on the minimum cut
                    group1 = [v['name'] for v in source_set]
                    group2 = [v['name'] for v in sink_set]
                    # Update the fuzzy_group_indices based on the minimum cut

                    # Update new_group_indices
                    # here is to judge which part to conect 
                    if is_connected(G,group1,new_group_indices[i]):
                        new_group_indices[i].extend(group1)
                        new_group_indices[j].extend(group2)
                    else:
                        new_group_indices[i].extend(group2)
                        new_group_indices[j].extend(group1)
        return new_group_indices
    #in first iterate,
    #probabilty is just average of 1/weight
    def Segmentation(self, triangle_indices_list):
        # Segment the mesh based on the provided triangle indices group
        new_indices_list = []

        # to avoid repetitive computation, generate dual mesh first

        for triangle_indices in triangle_indices_list:
            # Perform segmentation on each group of triangles
            new_group_indices = self.Segmentation_Base(triangle_indices)
            # in second iter, everyting goes wrong, but is in Segmentation_Base
            if new_group_indices is not None:
                new_indices_list.extend(new_group_indices)
            else:
                print("error,new_group_indices is None")
        return new_indices_list
    #return a list , each element is a list of triangle index
    def Segementation_recursion(self,recursion_time = 2):
        triangle_indices_list = []
        triangle_indice = self.get_all_triangle_indices()
        triangle_indices_list.append(triangle_indice)
        for i in range(0,recursion_time):
            triangle_indices_list = self.Segmentation(triangle_indices_list)
        return triangle_indices_list

    #determine K seeds,K no more than 10
    def Determine_kseed(self, triangle_indices,max_seed = 5):
        if self.firsttime == False:
            self.firsttime = True
        else:
            max_seed = 2
        distance_list = []
        node_list = []
        G_sub = self.dual_graph.subgraph(triangle_indices)
        
        first_node, _ = self.find_extreme_nodes(G_sub)
        node_list.append(first_node)

        for i in range(0, max_seed-1):
            node, distance = self.find_extreme_nodes(G_sub, node_list, find_min=False)
            node_list.append(node)
            distance_list.append(distance)

        # Find the longest descent
        largest_descent_index = max_seed
        optimal_distance = float('-inf')

        for i in range(len(distance_list),1):
            d_diff = distance_list[i] - distance_list[i - 1]
            if d_diff < optimal_distance:
                optimal_distance = d_diff
                largest_descent_index = i+1

        # Ensure the node list has unique elements
        node_list = list(set(node_list[:largest_descent_index]))

        # Return initial seeds
        return node_list
    def get_all_triangle_indices(self):
        # Get the triangle indices
        triangle_indices = []
        for i in range (0,len(self.mesh.triangles)):
            triangle_indices.append(i)
        # Print the triangle indices
        return triangle_indices
            

            
    
    #use it to find max/min node in dual graph
    #if target list is none, then compute average distance with all nodes
    #return optimal_node, distance
    def find_extreme_nodes(self, G, target_node_list=None, find_min=True):
        optimal_node = None
        if find_min:
            optimal_distance = float('inf')
        else:
            optimal_distance = float('-inf')

        if target_node_list is None:
            target_node_list = random.sample(range(G.vcount()), k=4)

        for node in G.vs:
            total_distance = 0
            for target in target_node_list:
                #temp = G.distances(node['name'], target)
                total_distance += G.distances(node['name'], target)[0][0]

            if find_min:
                if total_distance < optimal_distance:
                    optimal_distance = total_distance
                    optimal_node = node['name']
            else:
                if total_distance > optimal_distance:
                    optimal_distance = total_distance
                    optimal_node = node['name']

        return optimal_node, optimal_distance
#used for test
if __name__ == "__main__":
    ply_file = "C:/Users/hhq/Desktop/mesh_segmentation/data/horse.ply"

    # Start the timer
    start_time = time.time()

    # Call the function to read the PLY file as a mesh
    # already cost 28 seconds, may try tqdm
    mesh = read_ply_as_mesh(ply_file)
    normal= get_normal_from_ply(ply_file)
    #show_mesh_and_normal(mesh,normal)
    mesh_seg_horse = MeshSeg(mesh,normal)
    triangle_indices_list =  mesh_seg_horse.Segementation_recursion(2)
    total_items = sum(len(sublist) for sublist in triangle_indices_list)

    print(f"The 2D list has {total_items} items.")
    # Calculate the total time taken
    total_time = time.time() - start_time
    print("Total time taken:", total_time, "seconds")
    mesh_with_colors = render_K_segmentation(mesh, triangle_indices_list)


    # Visualize the mesh with colors
    o3d.visualization.draw_geometries([mesh_with_colors])
    # mesh_seg_horse.draw_dual_graph()