from simpleai.search import breadth_first, depth_first, uniform_cost, iterative_limited_depth_first, astar, SearchProblem
import osmnx as ox
from kd_tree import KDTree
import pyproj, time
from func_timeout import func_timeout, FunctionTimedOut

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

# americana neighbourhood coordinates (centered at the templo expiatorio)
coordinates = 20.67407230013045, -103.35893745618219

# get the projection map
G_proj = ox.project_graph(ox.graph_from_point(coordinates, dist=7000, network_type='drive'))

# extract nodes and coordinates
nodes = G_proj.nodes(data=True)
points = [(data['x'], data['y'], node_id) for node_id, data in nodes]

kdtree = KDTree(points) # build the kd_tree

TIMEOUT = 10

print('------------------------------------')
print('SHORT RANGE DISTANCES')
print('------------------------------------')

short_range_points = [(20.672415270083093, -103.35694735274907), (20.67451623525472, -103.3590330613026),
                      (20.673038501017935, -103.36236405280633), (20.67188488248092, -103.36308214722402),
                      (20.671211252700303, -103.35984558966872)]

astar_times = []
bfs_times = []
dfs_times = []
ucs_times = []
iddfs_times = []

x, y = project_point(coordinates[0], coordinates[1], G_proj)
e_point, e_id = kdtree.nearest_neighbor((x, y))

for i, (lat, lon) in enumerate(short_range_points):

    x, y = project_point(lat, lon, G_proj)
    i_point, i_id = kdtree.nearest_neighbor((x, y))

    rp = RoutePlanning(i_id, e_id, G_proj)

    # BFS
    start_time = time.time()
    try:
        bfs_solution = func_timeout(TIMEOUT, breadth_first, args=(rp,))
    except FunctionTimedOut:
        bfs_solution = None
    bfs_times.append(time.time() - start_time)

    # DFS
    start_time = time.time()
    try:
        dfs_solution = func_timeout(TIMEOUT, depth_first, args=(rp,))
    except FunctionTimedOut:
        bfs_solution = None
    dfs_times.append(time.time() - start_time)

    # UCS
    start_time = time.time()
    try:
        ucs_solution = func_timeout(TIMEOUT, uniform_cost, args=(rp,))
    except FunctionTimedOut:
        ucs_solution = None
    ucs_times.append(time.time() - start_time)

    # IDDFS
    start_time = time.time()
    try:
        iddfs_solution = func_timeout(TIMEOUT, iterative_limited_depth_first, args=(rp,))
    except FunctionTimedOut:
        iddfs_solution = None
    iddfs_times.append(time.time() - start_time)

    # ASTAR
    start_time = time.time()
    try:
        astar_solution = func_timeout(TIMEOUT, astar, args=(rp,))
    except FunctionTimedOut:
        astar_solution = None
    astar_times.append(time.time() - start_time)

    print(
        f"bfs: {1000 * bfs_times[i]:.3f} ms, "
        f"dfs: {1000 * dfs_times[i]:.3f} ms, "
        f"ucs: {1000 * ucs_times[i]:.3f} ms, "
        f"iddfs: {1000 * iddfs_times[i]:.3f} ms, "
        f"astar: {1000 * astar_times[i]:.3f} ms"
    )

print(f"bfs avg:   {1000 * (sum(bfs_times) / len(bfs_times)):.3f} ms")
print(f"dfs avg:   {1000 * (sum(dfs_times) / len(dfs_times)):.3f} ms")
print(f"ucs avg:   {1000 * (sum(ucs_times) / len(ucs_times)):.3f} ms")
print(f"iddfs avg: {1000 * (sum(iddfs_times) / len(iddfs_times)):.3f} ms")
print(f"astar avg: {1000 * (sum(astar_times) / len(astar_times)):.3f} ms")

print('------------------------------------')
print('MID RANGE DISTANCES')
print('------------------------------------')

mid_range_points = [(20.68003419605609, -103.34978764582817), (20.65861100742661, -103.35914622492686),
                    (20.670817895239068, -103.37691317671931), (20.660220704521567, -103.36591689816883),
                    (20.67170125043079, -103.34258090216602)]

astar_times = []
bfs_times = []
dfs_times = []
ucs_times = []
iddfs_times = []

x, y = project_point(coordinates[0], coordinates[1], G_proj)
e_point, e_id = kdtree.nearest_neighbor((x, y))

for i, (lat, lon) in enumerate(mid_range_points):

    x, y = project_point(lat, lon, G_proj)
    i_point, i_id = kdtree.nearest_neighbor((x, y))

    rp = RoutePlanning(i_id, e_id, G_proj)

    # BFS
    start_time = time.time()
    try:
        bfs_solution = func_timeout(TIMEOUT, breadth_first, args=(rp,))
    except FunctionTimedOut:
        bfs_solution = None
    bfs_times.append(time.time() - start_time)

    # DFS
    start_time = time.time()
    try:
        dfs_solution = func_timeout(TIMEOUT, depth_first, args=(rp,))
    except FunctionTimedOut:
        dfs_solution = None
    dfs_times.append(time.time() - start_time)

    # UCS
    start_time = time.time()
    try:
        ucs_solution = func_timeout(TIMEOUT, uniform_cost, args=(rp,))
    except FunctionTimedOut:
        ucs_solution = None
    ucs_times.append(time.time() - start_time)

    # IDDFS
    start_time = time.time()
    try:
        iddfs_solution = func_timeout(TIMEOUT, iterative_limited_depth_first, args=(rp,))
    except FunctionTimedOut:
        iddfs_solution = None
    iddfs_times.append(time.time() - start_time)

    # ASTAR
    start_time = time.time()
    try:
        astar_solution = func_timeout(TIMEOUT, astar, args=(rp,))
    except FunctionTimedOut:
        astar_solution = None
    astar_times.append(time.time() - start_time)

    # print the output for this node nicely
    parts = [
    f"bfs: {bfs_times[i]:.6f} s" if bfs_solution else "",
    f"dfs: {dfs_times[i]:.6f} s" if dfs_solution else "",
    f"ucs: {ucs_times[i]:.6f} s" if ucs_solution else "",
    f"iddfs: {iddfs_times[i]:.6f} s" if iddfs_solution else "",
    f"astar: {astar_times[i]:.6f} s" if astar_solution else "",
    ]
    output = ", ".join(filter(None, parts))
    print(output)


print(f"bfs avg:   {(sum(bfs_times) / len(bfs_times)):.6f} s")
print(f"dfs avg:   {(sum(dfs_times) / len(dfs_times)):.6f} s")
print(f"ucs avg:   {(sum(ucs_times) / len(ucs_times)):.6f} s")
print(f"iddfs avg: {(sum(iddfs_times) / len(iddfs_times)):.6f} s")
print(f"astar avg: {(sum(astar_times) / len(astar_times)):.6f} s")

print('------------------------------------')
print('LARGE RANGE DISTANCES')
print('------------------------------------')

large_range_points = [(20.69381511190883, -103.35149193509012), (20.676533932372752, -103.32373059890227),
                    (20.659550509767612, -103.3458862806719), (20.658677119431214, -103.37492739973364),
                    (20.67607236812044, -103.3235355446329)]

TIMEOUT = 0.5

astar_times = []
bfs_times = []
dfs_times = []
ucs_times = []
iddfs_times = []

x, y = project_point(coordinates[0], coordinates[1], G_proj)
e_point, e_id = kdtree.nearest_neighbor((x, y))

for i, (lat, lon) in enumerate(large_range_points):

    x, y = project_point(lat, lon, G_proj)
    i_point, i_id = kdtree.nearest_neighbor((x, y))

    rp = RoutePlanning(i_id, e_id, G_proj)

    # BFS
    start_time = time.time()
    try:
        bfs_solution = func_timeout(TIMEOUT, breadth_first, args=(rp,))
    except FunctionTimedOut:
        bfs_solution = None
    bfs_times.append(time.time() - start_time)

    # DFS
    start_time = time.time()
    try:
        dfs_solution = func_timeout(TIMEOUT, depth_first, args=(rp,))
    except FunctionTimedOut:
        dfs_solution = None
    dfs_times.append(time.time() - start_time)

    # UCS
    start_time = time.time()
    try:
        ucs_solution = func_timeout(TIMEOUT, uniform_cost, args=(rp,))
    except FunctionTimedOut:
        ucs_solution = None
    ucs_times.append(time.time() - start_time)

    # IDDFS
    start_time = time.time()
    try:
        iddfs_solution = func_timeout(TIMEOUT, iterative_limited_depth_first, args=(rp,))
    except FunctionTimedOut:
        iddfs_solution = None
    iddfs_times.append(time.time() - start_time)

    # ASTAR
    start_time = time.time()
    astar_solution = astar(rp)
    astar_times.append(time.time() - start_time)

    # print the output for this node nicely
    parts = [
    f"bfs: {bfs_times[i]:.6f} s" if bfs_solution else "",
    f"dfs: {dfs_times[i]:.6f} s" if dfs_solution else "",
    f"ucs: {ucs_times[i]:.6f} s" if ucs_solution else "",
    f"iddfs: {iddfs_times[i]:.6f} s" if iddfs_solution else "",
    f"astar: {astar_times[i]:.6f} s" if astar_solution else "",
    ]
    output = ", ".join(filter(None, parts))
    print(output)


print(f"bfs avg:   {(sum(bfs_times) / len(bfs_times)):.6f} s")
print(f"dfs avg:   {(sum(dfs_times) / len(dfs_times)):.6f} s")
print(f"ucs avg:   {(sum(ucs_times) / len(ucs_times)):.6f} s")
print(f"iddfs avg: {(sum(iddfs_times) / len(iddfs_times)):.6f} s")
print(f"astar avg: {(sum(astar_times) / len(astar_times)):.6f} s")