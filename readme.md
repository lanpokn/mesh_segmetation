# Change log

1. use openmesh to generate normal,because it's hard to use original data to get proper normal vector
2. use igraph to replace networkx, because networkx is too slow

# TODO

C:\Users\hhq\Desktop\mesh_segmentation\src\segmentation_new.py:51: DeprecationWarning: You are using integers as vertex names. This is discouraged because most igraph functions interpret integers as vertex _IDs_ and strings as vertex names. For sake of consistency, convert your vertex names to strings before assigning them. Future versions from igraph 0.11.0 will disallow integers as vertex names.

  G.add_vertex(i)

use string rather than integer has vertex name is better