# principle explanation


The project's main algorithmic principles are derived from the paper "Hierarchical Mesh Decomposition using Fuzzy Cluster." The existing principles from the paper will not be reiterated here. The project accomplishes **hierarchical k-way decomposition**, where the hierarchy is purely a coding-level technique, to be elaborated in the next section.

In the specific implementation, certain detailed issues arose, prompting explanations at the theoretical level. Firstly, the project is **Python-based** and utilizes **open3d** for point cloud reading and display. The initial generation of the dual graph involved using the **networkx** library for graph storage. However, during subsequent testing, the Dijkstra algorithm from networkx proved too slow for large graphs. As a result, the project transitioned to using the **igraph library**, ensuring acceptable time complexity.

Simultaneously, the code incorporates optimizations for speed. For instance, the generation of the dual graph involves a highly repetitive and parallelizable task. The project leverages **CPU-level parallel processing** to accelerate this task, achieving a speedup of approximately 3-4 times. Additionally, determining the point closest to all meshes in a dual graph, especially in large models, is a computationally complex task. To address this, the algorithm abstracts the distance to a few random points, ensuring a manageable time complexity.


Finally, in the original paper, there were differing criteria for determining convexity and concavity when calculating face-to-face distance weights. However, **open3d lacks the capability to assess convexity and concavity** (essentially designed for reading and displaying). To address this issue, I initially **utilized openmesh to generate standard normals** (ensuring the normals point outward). Then, for adjacent faces, I computed the t-value when the ray of one face's normal vector (o+td) intersects with another face's normal vector. If t is positive, it indicates concavity; otherwise, it is convex. This approach ensures the overall correctness of the algorithm.

It's worth noting that when applying min-cut to fuzzy regions, various failures often occur. The likely reason is that the original paper's k-segmentation cannot guarantee connectivity, leading to situations in fuzzy regions that are unsuitable for minimum cut. This significantly impacts the algorithm's stability. The best solution to this problem is setting e=0, i.e., disregarding the computation of fuzzy regions. Through experimental validation, it has been confirmed that the segmentation results under this condition are also excellent. Therefore, unless otherwise specified, **the default is to set e=0**.

# Explanation of Code Structure

The code is organized into various folders to manage different aspects of the project:

1. **data Folder:**
   - Stores the input PLY data.
2. **fig Folder:**
   - Stores the output results.
3. **doc Folder:**
   - Contains various relevant documents.
4. **src Folder (Important):**
   - Holds the source code files.
5. **src Folder Contents:**
   - **mesh_io.py:**
     - Responsible for reading and displaying point clouds.
   - **main.py:**
     - Execution example. Modify the PLY file path as needed.
   - **segmentation.py (Most Important):**
     - Contains the main logic for segmentation.
6. **Functions in segmentation.py:**
   - **`Segmentation_recursion(self, recursion_time=2):`**
     - Entry function for segmentation. `recursion_time` controls the number of recursive iterations. Setting it to 1 runs the segmentation without recursion, and it returns data for displaying the segmented mesh.
   - **`Segmentation(self, triangle_indices_list):`**
     - Wrapper function that segments all sub-meshes in the provided list and returns a finer-grained mesh.
   - **`Segmentation_Base(self, triangle_indices):`**
     - The lowest-level segmentation function. Takes a mesh as input and returns its segmentation.
     - Core logic involves determining k seed nodes and calculating groups based on these seeds.
7. **Convexity/Concavity Determination Code:**
   - Inspired by computer graphics principles, the code checks convexity/concavity based on the angle cosine and ray calculations.
   - The determination involves checking if the cosine of the angle is equal to 1. If so, the region is convex; otherwise, it uses ray calculations for further assessment.
8. **Verification:**
   - The correctness of the normal vectors is validated through graphical representations, ensuring they point outward.

The code structure facilitates a clear and modular organization, making it easier to understand and maintain. For detailed implementation, refer to the source code.![normal_is_right](.\fig\normal_is_right.jpg)

# 运行结果分析

Time：

2000mesh: 10s（K=6）

4000mesh:30-40s

Result：

no recursive：

![body_mesh](.\fig\body_mesh.jpg)

![cat](.\fig\cat.jpg)

![screwdriver](.\fig\screwdriver.jpg)

recursive：

![recursive_cat](.\fig\recursive_cat.jpg)



![bunny](.\fig\bunny.jpg)

