"""
================================================================
CLASS FOR A KD-TREE
================================================================
"""

class KDNode:
    def __init__(self, point, data=None, left=None, right=None):
        self.point = point  # tuple: (x, y)
        self.data = data    # optional, e.g., node ID
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, points=None):
        """
        poiints is in the form (x, y) or (x, y, data)
        """
        self.root = None
        if points:
            self.root = self.build_kd_tree(points, depth=0)
    
    def build_kd_tree(self, points, depth):
        if not points:
            return None
        
        # alternate the dimension: 0 for x, 1 for y
        k = 2
        axis = depth % k
        
        # sort points and choose median
        points.sort(key=lambda x: x[axis])
        median_idx = len(points) // 2
        median_point = points[median_idx]
        
        # if points include extra data (x, y, data)
        if len(median_point) == 3:
            point = (median_point[0], median_point[1])
            data = median_point[2]
        else:
            point = median_point
            data = None
        
        return KDNode(
            point=point,
            data=data,
            left=self.build_kd_tree(points[:median_idx], depth + 1),
            right=self.build_kd_tree(points[median_idx + 1:], depth + 1)
        )
    
    def nearest_neighbor(self, target, return_distance=False):
        """
        find the nearest neighbor to target (x, y)
        """
        best = [None, float('inf')]  # [node, distance]

        def search_kd_tree(node, depth):
            if node is None:
                return
            
            axis = depth % 2
            point = node.point
            dist = ((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2) ** 0.5 # euclidean dist
            
            if dist < best[1]:
                best[0] = node
                best[1] = dist
            
            # choose branch to search first
            diff = target[axis] - point[axis]
            close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
            
            search_kd_tree(close, depth + 1)
            
            # check if we need to search the away branch
            if abs(diff) < best[1]:
                search_kd_tree(away, depth + 1)
        
        search_kd_tree(self.root, 0)
        if return_distance:
            return best[0].point, best[1], best[0].data
        else:
            return best[0].point, best[0].data
