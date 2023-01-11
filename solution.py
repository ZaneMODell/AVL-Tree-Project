"""
Project 7
CSE 331 SS22 (Onsay)
Zane O'Dell
solution.py
"""
import math
import queue
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="bst_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the BSTree , properly handling the case of root = None. Recall that the height of
        an empty subtree is -1.
        :param: root: Node: The root Node of the subtree being measured.
        :return: Height of the subtree at root, i.e., the number of levels of Nodes below this Node. The height of a
        leaf Node is 0, and the height of a None-type is -1.
        """
        if root is None:
            return -1

        else:
            return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Insert a node with val into the subtree rooted at root, returning the root node of the balanced subtree after
        insertion. If val already exists in the tree, do nothing.
        Should update size and origin attributes of the object if necessary and correctly set parent/child pointers when
        inserting a new Node

        :param:root: Node: The root Node of the subtree in which to insert val.
        :param:val: T: The value to be inserted in the
         subtree rooted at root.
        :return: None
        """
        if self.origin is None:
            self.origin = Node(val)
            self.size = 1

        elif root is None:
            return

        elif root.value == val:
            return

        else:
            if val < root.value:
                if root.left is None:
                    root.left = Node(val)
                    self.size += 1
                else:
                    self.insert(root.left, val)

            elif val > root.value:
                if root.right is None:
                    root.right = Node(val)
                    self.size += 1
                else:
                    self.insert(root.right, val)
            root.height = 1 + max(self.height(root.right), self.height(root.left))

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Remove the node with value val from the subtree rooted at root, and return the root of the subtree following
        removal. If val does not exist in the BST tree, do nothing. Should update size and origin attributes of the
        object and correctly update parent/child pointers of Node objects as necessary.
        Should update the height attribute on all Node objects affected by the removal (ancestor nodes directly above on
        path to origin).
        :param:root: Node: The root Node of the subtree from which to delete val.
        :param:val: T: The value to be deleted from the subtree rooted at root.
        :return:: Root of new subtree after removal (could be the original root).
        """
        # handle empty and recursive left/right cases
        if root is None:
            return None
        elif val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # handle actual deletion step on this root
            if root.left is None:
                # pull up right child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.right
                if root.right is not None:
                    root.right.parent = root.parent
                self.size -= 1
                return root.right
            elif root.right is None:
                # pull up left child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.left
                if root.left is not None:
                    root.left.parent = root.parent
                self.size -= 1
                return root.left
            else:
                # two children: swap with predecessor and delete predecessor
                predecessor = root.left
                while predecessor.right is not None:
                    predecessor = predecessor.right
                root.value = predecessor.value
                root.left = self.remove(root.left, predecessor.value)
        # update height and rebalance every node that was traversed in recursive deletion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Find and return the Node with the value val in the subtree rooted at root. If val does not exist in the subtree
        rooted at root, return the Node below which val would be inserted as a child.
        :param:root: Node: The root Node of the subtree in which to search for val.
        :param:val: T: The value being searched in the subtree rooted at root.
        :return: Node object containing val if it exists, else the Node object below which val would be inserted as a
        child.
        """
        if root is None:
            return None

        if val == root.value:
            return root
        elif val < root.value:
            if root.left:
                return self.search(root.left, val)
            else:
                return root
        elif val > root.value:
            if root.right:
                return self.search(root.right, val)
            else:
                return root


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def visualize(self, filename="avl_tree_visualization.svg"):
        """
        Generates an svg image file of the binary tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + \
                          max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Remove the node with `value` from the subtree rooted at `root` if it exists.
        Return the root node of the balanced subtree following removal.

        :param root: root node of subtree from which to remove.
        :param val: value to be removed from subtree.
        :return: root node of balanced subtree.
        """
        # handle empty and recursive left/right cases
        if root is None:
            return None
        elif val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # handle actual deletion step on this root
            if root.left is None:
                # pull up right child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.right
                if root.right is not None:
                    root.right.parent = root.parent
                self.size -= 1
                return root.right
            elif root.right is None:
                # pull up left child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.left
                if root.left is not None:
                    root.left.parent = root.parent
                self.size -= 1
                return root.left
            else:
                # two children: swap with predecessor and delete predecessor
                predecessor = self.max(root.left)
                root.value = predecessor.value
                root.left = self.remove(root.left, predecessor.value)

        # update height and rebalance every node that was traversed in recursive deletion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return self.rebalance(root)

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree , properly handling the case of root = None. Recall that the height of
        an empty subtree is -1.
        :param: root: Node: The root Node of the subtree being measured.
        :return: Height of the subtree at root, i.e., the number of levels of Nodes below this Node. The height of a
        leaf Node is 0, and the height of a None-type is -1.
        """
        if root is None:
            return -1

        else:
            return root.height

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a right rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull left child up and shift left-right child across tree, update parent
        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        # left child has been pulled up to new root -> push old root down right, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + \
                          max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Computer the balance factor of the subtree rooted at root

        :param root: root node of unbalanced subtree to be rotated.
        :return: int representing the balance factor of the root
        """
        left_height = 0
        right_height = 0
        if root is None:
            return left_height - right_height

        if root.left is None:
            left_height = -1
        else:
            left_height = root.left.height
        if root.right is None:
            right_height = -1
        else:
            right_height = root.right.height

        return left_height - right_height

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Rebalance the subtree rooted at root (if necessary) and return the new root of the resulting subtree.
        :param: root: Node of the subtree to be rebalanced
        :return: Node: Root of the new subtree after rebalancing
        """
        if root is None:
            return None
        balance_factor = self.balance_factor(root)
        if abs(balance_factor) > 1:
            if balance_factor > 1:
                if self.balance_factor(root.left) < 0:
                    self.left_rotate(root.left)
                return self.right_rotate(root)

            else:
                if self.balance_factor(root.right) > 0:
                    self.right_rotate(root.right)
                return self.left_rotate(root)
        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        Insert a node with val into the subtree rooted at root, returning the root node of the balanced subtree after
        insertion.
        :param:root: Node: The root Node of the subtree in which to insert val.
        :param:val: T: The value to be inserted in the subtree rooted at root.
        :return: Root of new subtree after insertion and rebalancing (could be the original root).
        """
        if root is None:
            self.origin = Node(val)
            self.size = 1
            return self.origin

        elif root.value == val:
            return root

        else:
            if val < root.value:
                if root.left is None:
                    root.left = Node(val)
                    root.left.parent = root
                    self.size += 1
                    # return
                else:
                    self.insert(root.left, val)

            elif val > root.value:
                if root.right is None:
                    root.right = Node(val)
                    root.right.parent = root
                    self.size += 1
                else:
                    self.insert(root.right, val)
            root.height = 1 + max(self.height(root.right), self.height(root.left))
            return self.rebalance(root)

    def min(self, root: Node) -> Optional[Node]:
        """
        Returns the Node with the smallest value in the subtree beginning at root
        :param: root: root of the subtree
        :return: Node with the smallest value
        """
        if root is None:
            return None
        if root.left is None:
            return root
        else:
            return self.min(root.left)

    def max(self, root: Node) -> Optional[Node]:
        """
        Returns the Node with the largest value in the subtree beginning at root
        :param: root: root of the subtree
        :return: Node with the largest value
        """
        if root is None:
            return None
        if root.right is None:
            return root
        else:
            return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Find and return the Node with the value val in the subtree rooted at root. If val does not exist in the subtree
        rooted at root, return the Node below which val would be inserted as a child.
        :param:root: Node: The root Node of the subtree in which to search for val.
        :param:val: T: The value being searched in the subtree rooted at root.
        :return: Node object containing val if it exists, else the Node object below which val would be inserted as a
        child.
        """
        if root is None:
            return None

        if val == root.value:
            return root
        elif val < root.value:
            if root.left:
                return self.search(root.left, val)
            else:
                return root
        elif val > root.value:
            if root.right:
                return self.search(root.right, val)
            else:
                return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder (left, current, right) traversal of the subtree rooted at root using a Python generator.
        :param: root: Node in which the subtree traversal begins at
        :return: Generator yielding each of the node objects of the subtree traversed
        """
        if root:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Implementing this "dunder" method allows you to use an AVL tree class object anywhere you can use an iterable,
        e.g., inside of a for node in tree expression.
        :return: A generator that iterates over the inorder traversal of the tree
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a preorder (current, left, right) traversal of the subtree rooted at root using a Python generator.
        :param:root: Node: The root Node of the subtree currently being traversed.
        :return: Generator object which yields Node objects only (no None-type yields). Once all nodes of the tree have
        been yielded, a StopIteration exception is raised.
        """
        if not root:
            return
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)


    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a postorder (left, right, current) traversal of the subtree rooted at root using a Python generator.
        :param:root: Node: The root Node of the subtree currently being traversed.
        :return: Generator object which yields Node objects only (no None-type yields). Once all nodes of the tree have
        been yielded, a StopIteration exception is raised.
        """
        if not root:
            return
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a levelorder (breadth-order) traversal of the subtree rooted at root using a Python generator.
        :param:root: Node: The root Node of the subtree currently being traversed.
        :return: Generator object which yields Node objects only (no None-type yields). Once all nodes of the tree have
        been yielded, a StopIteration exception is raised.
        """
        q = queue.SimpleQueue()
        if root:
            q.put(root)
        while not q.empty():
            pop = q.get()
            if pop.left:
                q.put(pop.left)
            if pop.right:
                q.put(pop.right)

            yield pop



####################################################################################################


class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        pprinted_dict = json.dumps(self.dictionary, indent=2)
        return f"key: {self.key} dict:{self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return repr(self)

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return repr(self)

    def visualize(self, filename: str = "nnc_visualization.svg") -> str:
        svg_string = svg(self.tree.origin, 48, nnc_mode=True)
        print(svg_string, file=open(filename, 'w'))
        return svg_string

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Fits the one-dimensional NearestNeighborClassifier to data by:
        Rounding each x value to the number of digits specified by self.resolution to obtain a key,
        Searching for the Node in self.tree with this key, which is guaranteed to exist by construction of self.tree in
        self.__init__().
        Accessing the AVLWrappedDictionary object stored in node.value, and
        Updating the dictionary of this node.value by incrementing the number of times class y ended up at this node,
        i.e., the number of times class y was associated to key.
        :param:data: List[Tuple[float, str]]: A list of (x: float, y: str) pairs associating feature x values in the
        range [0, 1] to target y values. Provides the information necessary for our classifier to learn.
        :return: None.
        """

        for item in data:
            round_num = item[0]
            class_y = item[1]
            key = round(round_num, self.resolution)

            temp_wrapped = AVLWrappedDictionary(key)
            search_node = self.tree.search(self.tree.origin, temp_wrapped)

            if search_node:
                if class_y in search_node.value.dictionary:
                    search_node.value.dictionary[class_y] += 1
                else:
                    search_node.value.dictionary[class_y] = 1

        """Resolution = 2 -> approximate to the hundrenths place

tree.keys = [0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.0]
100 nodes

def fit(self, data: List[Tuple[float, str]]) -> None
	for d in data:
		# find node in the tree that corresponds to this data point
		# 	do that by rounding the 'key' of the datapoint (d[0])
		# 	to the resolution
		# 	i.e. if d = 0.18 and resolution = 1 -> round(d[0], resolution) = 0.1
		#	or if resolution = 2 -> round(d[0], resolution) = 0.18
		
		# since we found the node in the tree, we can increment its night/day count
		# based on this datapoints second value
		
predict(self, x: float, delta: float) -> str
	# retrieve all nodes within delta of the key (x)
	# sum up the dictionary counts 
	# return max count between night/day
	
0.3 , 0.2, resolution = 0.1
"""

    def predict(self, x: float, delta: float) -> str:
        """
        Predicts the class label of a single x value by:
        Rounding x to the number of digits specified by self.resolution to obtain a key,
        Searching for all Node objects in self.tree whose key is within ± delta of this key,
        Accessing the AVLWrappedDictionary object stored in all such node.values, and
        Taking the most common y label across all dictionaries stored in these node.values.
        Note that this process effectively predicts the class y based on the most common y observed in training data
        close to this x
        If no data in self.tree from the training set has a key within key ± delta, return None.
        Time / Space: O(k log n) / O(1)
        Here, k = (delta*10**resolution + 1) is the number of neighbors being searched in self.tree, and each search is
        an O(log n) operation.
        :param:x: float: Feature value in range [0, 1] with unknown class to be predicted.
        :param:delta: float: Width of interval to search across for neighbors of x.
        :return: str of the predicted class label y
        """
        class_dict = {}
        lower_bound = x - delta
        upper_bound = x + delta

        step = (1/(10 ** self.resolution))
        start_step = lower_bound
        while start_step <= upper_bound:
            temp_wrapped = AVLWrappedDictionary(start_step)
            search_node = self.tree.search(self.tree.origin, temp_wrapped)
            for key in search_node.value.dictionary.keys():
                if key not in class_dict:
                    class_dict[key] = search_node.value.dictionary[key]
                else:
                    class_dict[key] += search_node.value.dictionary[key]

            start_step += step
        if len(class_dict) == 0:
            return None

        max_value = max(class_dict.values())
        for item in class_dict.keys():
            if class_dict[item] == max_value:
                return item


def compare_times(structure, sizes, trail):
    """
    Comparing time on provide data structures in the worst case of BST tree
    :param structure: provided data struchtures
    :param sizes: size of test input
    :param trail: number of trail to test
    :return: None
    """
    import sys
    import time
    result = {}
    sys.stdout.write('\r')
    sys.stdout.write('Start...\n')
    total = len(sizes) * len(structure)
    count = 0
    for algorithm, value in structure.items():
        ImplementedTree = value
        if algorithm not in result:
            result[algorithm] = []
        for size in sizes:
            sum_times = 0
            for _ in range(trail):
                tree = ImplementedTree()
                start = time.perf_counter()
                for i in range(size):
                    tree.insert(tree.origin, i)
                for i in range(size, -1, -1):
                    tree.remove(tree.origin, i)
                end = time.perf_counter()
                sum_times += (end - start)
            count += 1
            result[algorithm].append(sum_times / trail)
            sys.stdout.write("[{:<20s}] {:d}%\n".format('=' * ((count * 20) // total),
                                                        count * 100 // total))
            sys.stdout.flush()
    return result

def compare_times(structure: dict, sizes: List[int], trial: int) -> dict:
    """
    Comparing time on provide data structures in the worst case of BST tree
    :param structure: provided data structures
    :param sizes: size of test input
    :param trial: number of trials to test
    :return: dict with list of average times per input size for each algorithm
    """
    import sys
    import time
    result = {}
    sys.stdout.write('\r')
    sys.stdout.write('Start...\n')
    total = len(sizes) * len(structure)
    count = 0
    for algorithm, value in structure.items():
        ImplementedTree = value
        if algorithm not in result:
            result[algorithm] = []
        for size in sizes:
            sum_times = 0
            for _ in range(trial):
                tree = ImplementedTree()
                start = time.perf_counter()
                for i in range(size):
                    tree.insert(tree.origin, i)
                for i in range(size, -1, -1):
                    tree.remove(tree.origin, i)
                end = time.perf_counter()
                sum_times += (end - start)
            count += 1
            result[algorithm].append(sum_times / trial)
            sys.stdout.write("[{:<20s}] {:d}%\n".format('=' * ((count * 20) // total),
                                                        count * 100 // total))
            sys.stdout.flush()
    return result


def plot_time_comparison():
    """
    Use compare_times to make a time comparison of normal binary search tree and AVL tree
    in a worst case scenario.
    Requires matplotlib. Comment this out if you do not wish to install matplotlib.
    """
    import matplotlib.pyplot as plt
    import sys
    sys.setrecursionlimit(2010)
    structures = {
        "bst": BinarySearchTree,
        "avl": AVLTree
    }
    sizes = [4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 300, 500, 1000, 2000]
    trials = 5
    data = compare_times(structures, sizes, trials)

    plt.style.use('seaborn-colorblind')
    plt.figure(figsize=(12, 8))

    for structure in structures:
        plt.plot(sizes, data[structure], label=structure)
    plt.legend()
    plt.xlabel("Input Size")
    plt.ylabel("Time to Sort (sec)")
    plt.title("BST vs AVL")
    plt.show()


_SVG_XML_TEMPLATE = """
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .value {{
        font: 300 16px monospace;
        text-align: center;
        dominant-baseline: middle;
        text-anchor: middle;
    }}
    .dict {{
        font: 300 16px monospace;
        dominant-baseline: middle;
    }}
    .node {{
        fill: lightgray;
        stroke-width: 1;
    }}
</style>
<g stroke="#000000">
{body}
</g>
</svg>
"""

_NNC_DICT_BOX_TEXT_TEMPLATE = """<text class="dict" y="{y}" xml:space="preserve">
    <tspan x="{label_x}" dy="1.2em">{label}</tspan>
    <tspan x="{bracket_x}" dy="1.2em">{{</tspan>
    {values}
    <tspan x="{bracket_x}" dy="1.2em">}}</tspan>
</text>
"""


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


def svg(root: Node, node_radius: int = 16, nnc_mode=False) -> str:
    """
    Taken from: https://github.com/joowani/binarytree

    Generate SVG XML.
    :param root: Generate SVG for tree rooted at root
    :param node_radius: Node radius in pixels (default: 16).
    :type node_radius: int
    :return: Raw SVG XML.
    :rtype: str
    """
    tree_height = root.height
    scale = node_radius * 3
    xml = deque()
    nodes_for_nnc_visualization: list[AVLWrappedDictionary] = []

    def scale_x(x: int, y: int) -> float:
        diff = tree_height - y
        x = 2 ** (diff + 1) * x + 2 ** diff - 1
        return 1 + node_radius + scale * x / 2

    def scale_y(y: int) -> float:
        return scale * (1 + y)

    def add_edge(parent_x: int, parent_y: int, node_x: int, node_y: int) -> None:
        xml.appendleft(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'.format(
                x1=scale_x(parent_x, parent_y),
                y1=scale_y(parent_y),
                x2=scale_x(node_x, node_y),
                y2=scale_y(node_y),
            )
        )

    def add_node(node_x: int, node_y: int, node: Node) -> None:
        x, y = scale_x(node_x, node_y), scale_y(node_y)
        xml.append(f'<circle class="node" cx="{x}" cy="{y}" r="{node_radius}"/>')

        if nnc_mode:
            nodes_for_nnc_visualization.append(node.value)
            xml.append(f'<text class="value" x="{x}" y="{y + 5}">key={node.value.key}</text>')
        else:
            xml.append(f'<text class="value" x="{x}" y="{y + 5}">{node.value}</text>')

    current_nodes = [root.left, root.right]
    has_more_nodes = True
    y = 1

    add_node(0, 0, root)

    while has_more_nodes:

        has_more_nodes = False
        next_nodes: List[Node] = []

        for x, node in enumerate(current_nodes):
            if node is None:
                next_nodes.append(None)
                next_nodes.append(None)
            else:
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                add_edge(x // 2, y - 1, x, y)
                add_node(x, y, node)

                next_nodes.append(node.left)
                next_nodes.append(node.right)

        current_nodes = next_nodes
        y += 1

    svg_width = scale * (2 ** tree_height)
    svg_height = scale * (2 + tree_height)
    if nnc_mode:

        line_height = 20
        box_spacing = 10
        box_margin = 5
        character_width = 10

        max_key_count = max(map(lambda obj: len(obj.dictionary), nodes_for_nnc_visualization))
        box_height = (max_key_count + 3) * line_height + box_margin

        def max_length_item_of_node_dict(node: AVLWrappedDictionary):
            # Check if dict is empty so max doesn't throw exception
            if len(node.dictionary) > 0:
                item_lengths = map(lambda pair: len(str(pair)), node.dictionary.items())
                return max(item_lengths)
            return 0

        max_value_length = max(map(max_length_item_of_node_dict, nodes_for_nnc_visualization))
        box_width = max(max_value_length * character_width, 110)

        boxes_per_row = svg_width // box_width
        rows_needed = math.ceil(len(nodes_for_nnc_visualization) / boxes_per_row)

        nodes_for_nnc_visualization.sort(key=lambda node: node.key)
        for index, node in enumerate(nodes_for_nnc_visualization):
            curr_row = index // boxes_per_row
            curr_column = index % boxes_per_row

            box_x = curr_column * (box_width + box_spacing)
            box_y = curr_row * (box_height + box_spacing) + svg_height
            box = f'<rect x="{box_x}" y="{box_y}" width="{box_width}" ' \
                  f'height="{box_height}" fill="white" />'
            xml.append(box)

            value_template = '<tspan x="{value_x}" dy="1.2em">{key}: {value}</tspan>'
            text_x = box_x + 10

            def item_pair_to_svg(pair):
                return value_template.format(key=pair[0], value=pair[1], value_x=text_x + 10)

            values = map(item_pair_to_svg, node.dictionary.items())
            text = _NNC_DICT_BOX_TEXT_TEMPLATE.format(
                y=box_y,
                label=f"key = {node.key}",
                label_x=text_x,
                bracket_x=text_x,
                values='\n'.join(values)
            )
            xml.append(text)

        svg_width = boxes_per_row * (box_width + box_spacing * 2)
        svg_height += rows_needed * (box_height + box_spacing * 2)

    return _SVG_XML_TEMPLATE.format(
        width=svg_width,
        height=svg_height,
        body="\n".join(xml),
    )


if __name__ == "__main__":
    plot_time_comparison()
