# Change log

1. use openmesh to generate normal,because it's hard to use original data to get proper normal vector
2. use igraph to replace networkx, because networkx is too slow
3. use true name intead index, beacause after generate subgraph. index will be rearrange

# TODO

1.

C:\Users\hhq\Desktop\mesh_segmentation\src\segmentation_new.py:51: DeprecationWarning: You are using integers as vertex names. This is discouraged because most igraph functions interpret integers as vertex _IDs_ and strings as vertex names. For sake of consistency, convert your vertex names to strings before assigning them. Future versions from igraph 0.11.0 will disallow integers as vertex names.

  G.add_vertex(i)

use string rather than integer has vertex name is better

2. learn how to generate normals

# NOTE

最大流最小割只是用了min_cut库一步解决，而且用最大流最小割时割图经常出现不连通等导致不valid的情况，而且我很难理解非要搞个最大流最小割有什么意义，我感觉纯是论文为了显得高大上凑数的，带验证，总之我还是把e设置为0了。

可能最大流最小割的意义只有在特别窄的地方能感觉出来

最大流最小割问题很大。。。在复杂的图形中不能保证割是存在的，学习一下最大流最小割，抢救不了就不抢救了