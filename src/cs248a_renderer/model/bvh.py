import logging
from dataclasses import dataclass, field
from multiprocessing import Value
from typing import Dict, List, Tuple, Callable
import numpy as np
from pyglm import glm
from pyglm.glm import vec1, vec3
import slangpy as spy
from collections import deque

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []

        # TODO: Student implementation starts here.
        bounding_box = None
        for p in primitives:
            if bounding_box is None:
                bounding_box = p.bounding_box
            else:
                bounding_box = BoundingBox3D.union(bounding_box, p.bounding_box)

        self.nodes, primitives = self.generate_bvh(0, len(primitives), 0, bounding_box, primitives, max_nodes, min_prim_per_node, num_thresholds, on_progress)
        # TODO: Student implementation ends here.

    def generate_bvh(self, start_index, end_index, current_number_nodes, bounding_box: BoundingBox3D, primitives: List[Primitive], max_nodes: int, min_prim_per_node: int, num_thresholds: int, on_progress: Callable[[int, int], None] | None = None):
        
        queue = deque()
        nodes = []
    
        node_primitives_map = {}
        
        root_node = BVHNode(bounding_box, left=-1, right=-1, prim_left=0, prim_right=len(primitives), depth=0)
        nodes.append(root_node)
        node_primitives_map[0] = primitives
        queue.append((0, 0, len(primitives), 0, bounding_box, primitives, max_nodes - 1))
        node_count = 1
        
        while queue and node_count < max_nodes:
            node_idx, start_idx, end_idx, current_depth, bounding_box, node_primitives, available_nodes = queue.popleft()
            # print(f"Processing node {node_idx} at depth {current_depth}, primitives: {len(node_primitives)}", flush=True)
            if len(node_primitives) <= min_prim_per_node or available_nodes < 2:
                continue
            
            min_cost = float('inf')
            min_left_bounding_box = None
            min_right_bounding_box = None
            min_left_primitives = []
            min_right_primitives = []
            
            for axis in range(3):
                if axis == 0:
                    normal = vec3(1, 0, 0)
                elif axis == 1:
                    normal = vec3(0, 1, 0)
                else:
                    normal = vec3(0, 0, 1)
                split_offset = (bounding_box.max[axis] - bounding_box.min[axis]) / num_thresholds
                for threshold in range(num_thresholds):
                    point = bounding_box.min + normal * threshold * split_offset
                    left_primitives, right_primitives, left_bounding_box, right_bounding_box = self.split_primitives(node_primitives, point, normal)
                    if len(left_primitives) == 0 or len(right_primitives) == 0:
                        continue
                    cost = left_bounding_box.area * len(left_primitives) + right_bounding_box.area * len(right_primitives)
                    if cost < min_cost:
                        min_cost = cost
                        min_left_bounding_box = left_bounding_box
                        min_right_bounding_box = right_bounding_box
                        min_left_primitives = left_primitives
                        min_right_primitives = right_primitives
            
            if len(min_left_primitives) == 0 or len(min_right_primitives) == 0:
                continue
            
            # print(f"Best split: {len(min_left_primitives)} left, {len(min_right_primitives)} right, cost: {min_cost}", flush=True)
            
            left_idx = len(nodes)
            right_idx = len(nodes) + 1
            
            left_start = start_idx
            left_end = start_idx + len(min_left_primitives)
            right_start = left_end
            right_end = end_idx
            
            left_node = BVHNode(min_left_bounding_box, left=-1, right=-1, 
                               prim_left=left_start, prim_right=left_end,
                               depth=current_depth + 1)
            right_node = BVHNode(min_right_bounding_box, left=-1, right=-1,
                                prim_left=right_start, prim_right=right_end,
                                depth=current_depth + 1)
            
            nodes.append(left_node)
            nodes.append(right_node)
            
            node_primitives_map[left_idx] = min_left_primitives
            node_primitives_map[right_idx] = min_right_primitives
            
            nodes[node_idx].left = left_idx
            nodes[node_idx].right = right_idx
            nodes[node_idx].prim_left = 0
            nodes[node_idx].prim_right = 0

            queue.append((left_idx, left_start, left_end, 
                         current_depth + 1, min_left_bounding_box, min_left_primitives, available_nodes - 2))
            queue.append((right_idx, right_start, right_end,
                         current_depth + 1, min_right_bounding_box, min_right_primitives, available_nodes - 2))
            
            node_count += 2
            
            on_progress(node_count, max_nodes)
        
        temp_primitives = [None] * len(primitives)
        
        def collect_primitives(node_idx):
            node = nodes[node_idx]
            if node.left == -1 and node.right == -1:
                if node_idx in node_primitives_map:
                    prims = node_primitives_map[node_idx]
                    for i, prim in enumerate(prims):
                        temp_primitives[node.prim_left + i] = prim
            else:
                if node.left != -1:
                    collect_primitives(node.left)
                if node.right != -1:
                    collect_primitives(node.right)
        
        collect_primitives(0)
        
        for i, prim in enumerate(temp_primitives):
            if prim is not None:
                primitives[i] = prim
        
        print(f"BVH construction complete: {len(nodes)} nodes created", flush=True)
        print(f"Primitives reordered in-place: {len(primitives)}", flush=True)
        
        leaf_count = sum(1 for n in nodes if n.left == -1 and n.right == -1)
        print(f"Leaf nodes: {leaf_count}", flush=True)
        
        return nodes, primitives



    def split_primitives(self, primitives, split_point: vec3, normal: vec3):
        left_primitives = []
        left_bounding_box = None
        right_primitives = []
        right_bounding_box = None
        for p in primitives:
            bounding_box = p.bounding_box
            v = bounding_box.center - split_point
            dot_product = glm.dot(v, normal)
            if (dot_product > 0):
                if left_bounding_box is None:
                    left_bounding_box = bounding_box
                else:
                    left_bounding_box = BoundingBox3D.union(left_bounding_box, bounding_box)
                left_primitives.append(p)
            else:
                if right_bounding_box is None:
                    right_bounding_box = bounding_box
                else:
                    right_bounding_box = BoundingBox3D.union(right_bounding_box, bounding_box)
                right_primitives.append(p)
        if left_bounding_box is None:
            left_bounding_box = BoundingBox3D(min=split_point, max=split_point)
        if right_bounding_box is None:
            right_bounding_box = BoundingBox3D(min=split_point, max=split_point)
        # print(f"left bounding box center {left_bounding_box.center}", flush=True)
        # print(f"right bounding box center {right_bounding_box.center}", flush=True)
        # print(f"left primitives {len(left_primitives)}", flush=True)
        # print(f"right primitives {len(right_primitives)}", flush=True)
        return left_primitives, right_primitives, left_bounding_box, right_bounding_box

def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
