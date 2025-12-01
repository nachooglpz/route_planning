import osmnx as ox
import pyproj
import time
from kd_tree import KDTree

# ox.settings.log_console = True

# get projection coordinates from graph
def project_point(lat, lon, G_proj):
    proj_crs = G_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

# americana neighbourhood coordinates (centered at the templo expiatorio)
coordinates = 20.67407230013045, -103.35893745618219

G = ox.graph_from_point(coordinates, dist=1000, network_type='drive')

# get the coordinates
G_proj = ox.project_graph(G)

# extract nodes and coordinates
nodes = G_proj.nodes(data=True)
points = [(data['x'], data['y'], node_id) for node_id, data in nodes]

# print(points)

# build the KD-Tree
start_time = time.time()
kdtree = KDTree(points)
end_time = time.time()

print(f"Time to build the KD-Tree: {1000 * (end_time - start_time):.6f} ms\n")


locations = [
    (20.672415270083093, -103.35694735274907), (20.67451623525472, -103.3590330613026),
    (20.673038501017935, -103.36236405280633), (20.67188488248092, -103.36308214722402),
    (20.671211252700303, -103.35984558966872), (20.67192895162325, -103.35399825383365),
    (20.673332861627674, -103.35823740397204), (20.678432156484185, -103.35787404828217),
    (20.677790032478438, -103.3635397062817), (20.67580698534928, -103.36374157056464),
    (20.67308733571283, -103.36355316390105), (20.670846106321502, -103.35950914913484),
    (20.668902999119315, -103.35878597031072), (20.668317496492215, -103.36115451137914),
    (20.667883089880263, -103.36176683309122), (20.669475908116052, -103.3661943901561),
    (20.668380453873596, -103.36359706956144), (20.674581627982185, -103.35591276844035),
    (20.675185992803588, -103.35320105811411), (20.672957385547754, -103.35063738151716)
    ]


for lat, lon in locations:
    x, y = project_point(lat, lon, G_proj)
    start_time = time.time()
    point, dist, id = kdtree.nearest_neighbor((x, y), return_distance=True)
    nearest_node = nodes[id]
    end_time = time.time()
    print(f"Nearest node to ({lat},{lon}) is {nearest_node} with distance {dist}, search time: {1000 * (end_time - start_time):.6f} ms\n")