from simpleai.search import astar, SearchProblem
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from kd_tree import KDTree
import pyproj

# search problem definition
class RoutePlanning(SearchProblem):
    def __init__(self, i_id, e_id, G):
        SearchProblem.__init__(self, i_id)
        self.goal = e_id
        self.G = G

    def actions(self, state):
        return list(self.G.neighbors(state))
    
    def result(self, state, action):
        return action
    
    def is_goal(self, state):
        return state == self.goal
    
    def cost(self, state, action, state2):
        data = list(self.G.get_edge_data(state, state2).values())[0]
        return data.get('length', 1)
    
    def heuristic(self, state):
        # using euclidean distance
        x1 = self.G.nodes[state]['x']
        y1 = self.G.nodes[state]['y']

        x2 = self.G.nodes[self.goal]['x']
        y2 = self.G.nodes[self.goal]['y']

        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# get projection coordinates from graph
def project_point(lat, lon, G_proj):
    proj_crs = G_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

# get the closest hospital
def get_closest_hosp(tree, coords, G_proj):
    x, y = project_point(coords[0], coords[1], G_proj)

    return tree.nearest_neighbor(x, y)

# americana neighbourhood coordinates (centered at the templo expiatorio)
coordinates = 20.67407230013045, -103.35893745618219

# get the projection map
G_proj = ox.project_graph(ox.graph_from_point(coordinates, dist=7000, network_type='drive'))

# extract nodes and coordinates
nodes = G_proj.nodes(data=True)
points = [(data['x'], data['y'], node_id) for node_id, data in nodes]

kdtree = KDTree(points) # build the kd_tree

# hospital coords i found in maps
hospitals = [(20.673211891402076, -103.35950088419128), (20.676163033971346, -103.35825633925893),
             (20.67750809352649, -103.35795593186148), (20.6777088476619, -103.35804176254646),
             (20.680017501136977, -103.36085271747983), (20.670120156735344, -103.3553166382388),
             (20.672188012231366, -103.35396480495025), (20.681723874771457, -103.35780572808602),
             (20.686220578866905, -103.34448051424168), (20.674516825501843, -103.36656045852163),
             (20.678230807210603, -103.37102365414098), (20.676825527519473, -103.36632442413793),
             (20.66313340746032, -103.36291265412089), (20.68574522688616, -103.36847274639096), 
             (20.678317540315987, -103.37769954502716), (20.67813686233053, -103.3778497487259),
             (20.68726227514667, -103.33390048641502), (20.672085468860526, -103.32508138308421)]

# projection points for hospital coords
hosp_proj_points = [project_point(lat, lon, G_proj) for (lat, lon) in hospitals]

# get the hospital nodes
hosp_nodes = [node_id for (_, _, node_id) in  (kdtree.nearest_neighbor((x, y), return_distance=True) for (x, y) in hosp_proj_points)]

hosp_kd = KDTree([(x, y, i) for i, (x, y) in enumerate(hosp_proj_points)])

# assign each node its closest hospital
node_to_hospital = {}
for node_id, data in G_proj.nodes(data=True):
    x, y = data['x'], data['y']
    (_, hosp_idx) = hosp_kd.nearest_neighbor((x, y))
    node_to_hospital[node_id] = hosp_idx

# some random points that i selected on maps to test getting the nearest hospital
rand_points = [(20.672459116830847, -103.35685651020869), (20.671063821847405, -103.36298267544693),
               (20.668032273434655, -103.3669630735779), (20.67362931100878, -103.35180607317065),
               (20.66939283815133, -103.3507874753673), (20.678193252711818, -103.35206868064547),
               (20.67870696676613, -103.3616657833699), (20.67717326468566, -103.36527066524614)]

# get the route to the nearest hospital
for (xp, yp) in rand_points:
    # project the coords
    p = project_point(xp, yp, G_proj)
    _, i_id = kdtree.nearest_neighbor(p)
    nearest_hosp = hosp_nodes[node_to_hospital[i_id]]
    res = astar(RoutePlanning(i_id, nearest_hosp, G_proj))

    print(f"ROUTE FOR COORDS: {xp}, {yp}:")

    path_nodes = [state for (_, state) in res.path()]
    # for node in path_nodes:
    #     print(node)

    # get the coords of the nodes
    path_coords = [(G_proj.nodes[node]['x'], G_proj.nodes[node]['y']) for node in path_nodes]

    # plot the graph
    fig, ax = ox.plot_graph(G_proj, show=False, close=False, node_size=10, edge_color='lightgray')
    
    # plot the path
    xs, ys = zip(*path_coords)
    ax.plot(xs, ys, color='red', linewidth=2, label='Route')
    
    # mark start and hospital
    ax.scatter(G_proj.nodes[i_id]['x'], G_proj.nodes[i_id]['y'], c='blue', s=50, label='Start')
    ax.scatter(G_proj.nodes[nearest_hosp]['x'], G_proj.nodes[nearest_hosp]['y'],
               c='green', s=50, label='Hospital')
    
    ax.legend()
    plt.show()