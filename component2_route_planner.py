from simpleai.search import breadth_first, depth_first, uniform_cost, iterative_limited_depth_first, astar, SearchProblem
import osmnx as ox
import pyproj

# Search problem definition
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

# americana neighbourhood coordinates (centered at the templo expiatorio)
coordinates = 20.67407230013045, -103.35893745618219

# get the projection map
G_proj = ox.project_graph(ox.graph_from_point(coordinates, dist=7000, network_type='drive'))

# extract nodes and coordinates
nodes = G_proj.nodes(data=True)
points = [(data['x'], data['y'], node_id) for node_id, data in nodes]