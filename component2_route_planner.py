from simpleai.search import breadth_first, depth_first, uniform_cost, iterative_limited_depth_first, astar, SearchProblem
import osmnx as ox
from kd_tree import KDTree
import pyproj, time

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

kdtree = KDTree(points) # build the kd_tree

print('------------------------------------')
print('SHORT RANGE DISTANCES')
print('------------------------------------')

short_range_points = [(20.672415270083093, -103.35694735274907), (20.67451623525472, -103.3590330613026),
                      (20.673038501017935, -103.36236405280633), (20.67188488248092, -103.36308214722402),
                      (20.671211252700303, -103.35984558966872)
                    ]

astar_times = []
bfs_times = []
# dfs_times = []
ucs_times = []
iddfs_times = []
for i, (lat, lon) in enumerate(short_range_points):
    # take initial vertex
    x, y = project_point(lat, lon, G_proj)
    i_point, i_id = kdtree.nearest_neighbor((x, y))

    # take final vertex
    x, y = project_point(coordinates[0], coordinates[1], G_proj)
    e_point, e_id = kdtree.nearest_neighbor((x, y))

    # measure times
    
    # bfs time
    # print('trying bfs')
    start_time = time.time()
    bfs_solution = breadth_first(RoutePlanning(i_id, e_id, G_proj))
    end_time = time.time()
    bfs_times.append(end_time - start_time)
    #print('bfs finished')

    # print('trying dfs')
    # # dfs time
    # start_time = time.time()
    # dfs_solution = depth_first(RoutePlanning(i_id, e_id, G_proj))
    # end_time = time.time()
    # dfs_times.append(end_time - start_time)
    # print('dfs finished')

    # ucs time
    # print('trying ucs')
    start_time = time.time()
    ucs_solution = uniform_cost(RoutePlanning(i_id, e_id, G_proj))
    end_time = time.time()
    ucs_times.append(end_time - start_time)
    # print('ucs finished')

    # iddfs time
    # print('trying iddfs')
    start_time = time.time()
    iddfs_solution = iterative_limited_depth_first(RoutePlanning(i_id, e_id, G_proj))
    end_time = time.time()
    iddfs_times.append(end_time - start_time)
    # print('iddfs finished')

    # astar time
    # print('trying astart')
    start_time = time.time()
    astar_solution = astar(RoutePlanning(i_id, e_id, G_proj))
    end_time = time.time()
    astar_times.append(end_time - start_time)
    # print('astar finished')

    # print(f"bfs time: {bfs_times[i]:.6f}, dfs time: {dfs_times[i]:.6f}, ucs time: {ucs_times[i]:.6f}, iddfs time: {iddfs_times[i]:.6f}, astar time: {astar_times[i]:.6f}.")
    print(f"bfs time: {1000 * bfs_times[i]:.6f} ms, ucs time: {1000 * ucs_times[i]:.6f} ms, iddfs time: {1000 * iddfs_times[i]:.6f} ms, astar time: {1000 * astar_times[i]:.6f} ms.")

print(f"bfs average time: {1000 * (sum(bfs_times) / len(bfs_times)):.6f} ms")
# print(f"dfs average time: {(sum(dfs_times) / len(dfs_times)):.6f} s")
print(f"ucs average time: {1000 * (sum(ucs_times) / len(ucs_times)):.6f} ms")
print(f"iddfs average time: {1000 * (sum(iddfs_times) / len(iddfs_times)):.6f} ms")
print(f"astar average time: {1000 * (sum(astar_times) / len(astar_times)):.6f} ms")