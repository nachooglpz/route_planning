import osmnx as ox
import pyproj

# get projection coordinates from graph
def project_point(lat, lon, G_proj):
    proj_crs = G_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

# americana neighbourhood coordinates (centered at the templo expiatorio)
coordinates = 20.67407230013045, -103.35893745618219

# get the projection map
G_proj = ox.project_graph(ox.graph_from_point(coordinates, dist=7000, network_type='drive'))

# extract nodes and coordinates
nodes = G_proj.nodes(data=True)
points = [(data['x'], data['y'], node_id) for node_id, data in nodes]