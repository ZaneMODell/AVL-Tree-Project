"""
Project 5
CSE 331 S21 (Onsay)
Andrew McDonald & Bank Premsri
Inspired by Brandon Field and Anna DeBiasi's implementation
tests.py
"""

import unittest
import random
import types
from solution import Node, AVLTree, NearestNeighborClassifier, BinarySearchTree
from xml.dom import minidom

NAMES = {0: 'Aaron', 1: 'Abigail', 2: 'Adam', 3: 'Alan', 4: 'Albert', 5: 'Alexander', 6: 'Alexis', 7: 'Alice',
         8: 'Amanda', 9: 'Amber', 10: 'Amy', 11: 'Andrea', 12: 'Andrew', 13: 'Angela', 14: 'Ann', 15: 'Anna',
         16: 'Anthony', 17: 'Arthur', 18: 'Ashley', 19: 'Austin', 20: 'Barbara', 21: 'Benjamin', 22: 'Betty',
         23: 'Beverly', 24: 'Billy', 25: 'Bobby', 26: 'Brandon', 27: 'Brenda', 28: 'Brian', 29: 'Brittany', 30: 'Bruce',
         31: 'Bryan', 32: 'Carl', 33: 'Carol', 34: 'Carolyn', 35: 'Catherine', 36: 'Charles', 37: 'Charlotte',
         38: 'Cheryl', 39: 'Christian', 40: 'Christina', 41: 'Christine', 42: 'Christopher', 43: 'Cynthia',
         44: 'Daniel', 45: 'Danielle', 46: 'David', 47: 'Deborah', 48: 'Debra', 49: 'Denise', 50: 'Dennis', 51: 'Diana',
         52: 'Diane', 53: 'Donald', 54: 'Donna', 55: 'Doris', 56: 'Dorothy', 57: 'Douglas', 58: 'Dylan', 59: 'Edward',
         60: 'Elijah', 61: 'Elizabeth', 62: 'Emily', 63: 'Emma', 64: 'Eric', 65: 'Ethan', 66: 'Eugene', 67: 'Evelyn',
         68: 'Frances', 69: 'Frank', 70: 'Gabriel', 71: 'Gary', 72: 'George', 73: 'Gerald', 74: 'Gloria', 75: 'Grace',
         76: 'Gregory', 77: 'Hannah', 78: 'Harold', 79: 'Heather', 80: 'Helen', 81: 'Henry', 82: 'Isabella', 83: 'Jack',
         84: 'Jacob', 85: 'Jacqueline', 86: 'James', 87: 'Janet', 88: 'Janice', 89: 'Jason', 90: 'Jean', 91: 'Jeffrey',
         92: 'Jennifer', 93: 'Jeremy', 94: 'Jerry', 95: 'Jesse', 96: 'Jessica', 97: 'Joan', 98: 'Joe', 99: 'John',
         100: 'Johnny', 101: 'Jonathan', 102: 'Jordan', 103: 'Jose', 104: 'Joseph', 105: 'Joshua', 106: 'Joyce',
         107: 'Juan', 108: 'Judith', 109: 'Judy', 110: 'Julia', 111: 'Julie', 112: 'Justin', 113: 'Karen',
         114: 'Katherine', 115: 'Kathleen', 116: 'Kathryn', 117: 'Kayla', 118: 'Keith', 119: 'Kelly', 120: 'Kenneth',
         121: 'Kevin', 122: 'Kimberly', 123: 'Kyle', 124: 'Larry', 125: 'Laura', 126: 'Lauren', 127: 'Lawrence',
         128: 'Linda', 129: 'Lisa', 130: 'Logan', 131: 'Louis', 132: 'Madison', 133: 'Margaret', 134: 'Maria',
         135: 'Marie', 136: 'Marilyn', 137: 'Mark', 138: 'Martha', 139: 'Mary', 140: 'Matthew', 141: 'Megan',
         142: 'Melissa', 143: 'Michael', 144: 'Michelle', 145: 'Nancy', 146: 'Natalie', 147: 'Nathan', 148: 'Nicholas',
         149: 'Nicole', 150: 'Noah', 151: 'Olivia', 152: 'Pamela', 153: 'Patricia', 154: 'Patrick', 155: 'Paul',
         156: 'Peter', 157: 'Philip', 158: 'Rachel', 159: 'Ralph', 160: 'Randy', 161: 'Raymond', 162: 'Rebecca',
         163: 'Richard', 164: 'Robert', 165: 'Roger', 166: 'Ronald', 167: 'Rose', 168: 'Roy', 169: 'Russell',
         170: 'Ruth', 171: 'Ryan', 172: 'Samantha', 173: 'Samuel', 174: 'Sandra', 175: 'Sara', 176: 'Sarah',
         177: 'Scott', 178: 'Sean', 179: 'Sharon', 180: 'Shirley', 181: 'Sophia', 182: 'Stephanie', 183: 'Stephen',
         184: 'Steven', 185: 'Susan', 186: 'Teresa', 187: 'Terry', 188: 'Theresa', 189: 'Thomas', 190: 'Timothy',
         191: 'Tyler', 192: 'Victoria', 193: 'Vincent', 194: 'Virginia', 195: 'Walter', 196: 'Wayne', 197: 'William',
         198: 'Willie', 199: 'Zachary'}


class BSTTests(unittest.TestCase):

    def test_insert_bst(self):
        """
        (1) Test inserting to empty tree
        final structure:
            1
        """
        bst = BinarySearchTree()
        bst.insert(bst.origin, 1)
        self.assertEqual(1, bst.origin.value)
        self.assertEqual(0, bst.origin.height)
        self.assertEqual(None, bst.origin.left)
        self.assertEqual(None, bst.origin.right)

        """
        (2) Test inserting to cause imbalance tree on left
        final structure:
               10
              /
             5
            /
           1
          /
        -1
        """
        bst = BinarySearchTree()
        for value in [10, 5, 1, -1]:
            bst.insert(bst.origin, value)
        self.assertEqual(10, bst.origin.value)
        self.assertEqual(3, bst.origin.height)
        self.assertEqual(5, bst.origin.left.value)
        self.assertEqual(2, bst.origin.left.height)
        self.assertEqual(1, bst.origin.left.left.value)
        self.assertEqual(1, bst.origin.left.left.height)
        self.assertEqual(-1, bst.origin.left.left.left.value)
        self.assertEqual(0, bst.origin.left.left.left.height)
        self.assertEqual(None, bst.origin.right)
        """
        (3) Test inserting to cause imbalance tree on left
        final structure:
             10
            /  \
           1    12
                 \
                  13
                   \
                   14
                    \
                    15
        """
        bst = BinarySearchTree()
        for value in [10, 12, 13, 14, 15, 1]:
            bst.insert(bst.origin, value)
        self.assertEqual(10, bst.origin.value)
        self.assertEqual(4, bst.origin.height)
        self.assertEqual(1, bst.origin.left.value)
        self.assertEqual(0, bst.origin.left.height)

        self.assertEqual(12, bst.origin.right.value)
        self.assertEqual(3, bst.origin.right.height)
        self.assertEqual(13, bst.origin.right.right.value)
        self.assertEqual(2, bst.origin.right.right.height)
        self.assertEqual(14, bst.origin.right.right.right.value)
        self.assertEqual(1, bst.origin.right.right.right.height)
        self.assertEqual(15, bst.origin.right.right.right.right.value)
        self.assertEqual(0, bst.origin.right.right.right.right.height)

        """
        (4) Test inserting to complex tree (no rotating)
        final structure:
                        10
                    /        \
                  7           19
                /             / \
               4            13   35
              /  \           \   /   
             1    6          17 25
        """
        bst = BinarySearchTree()
        for value in [10, 7, 4, 19, 35, 25, 13, 17, 1, 6]:
            bst.insert(bst.origin, value)
        # Height 3
        self.assertEqual(10, bst.origin.value)
        self.assertEqual(3, bst.origin.height)

        # Height 2
        self.assertEqual(7, bst.origin.left.value)
        self.assertEqual(2, bst.origin.left.height)
        self.assertEqual(19, bst.origin.right.value)
        self.assertEqual(2, bst.origin.right.height)

        # Height 1
        self.assertEqual(4, bst.origin.left.left.value)
        self.assertEqual(1, bst.origin.left.left.height)
        self.assertEqual(13, bst.origin.right.left.value)
        self.assertEqual(1, bst.origin.right.left.height)
        self.assertEqual(35, bst.origin.right.right.value)
        self.assertEqual(1, bst.origin.right.right.height)

        # Height 0
        self.assertEqual(1, bst.origin.left.left.left.value)
        self.assertEqual(0, bst.origin.left.left.left.height)
        self.assertEqual(6, bst.origin.left.left.right.value)
        self.assertEqual(0, bst.origin.left.left.right.height)
        self.assertEqual(17, bst.origin.right.left.right.value)
        self.assertEqual(0, bst.origin.right.left.right.height)
        self.assertEqual(25, bst.origin.right.right.left.value)
        self.assertEqual(0, bst.origin.right.right.left.height)

    def test_remove_bst(self):
        # visualize this testcase with https://www.cs.usfca.edu/~galles/visualization/AVLtree.html
        # ensure empty tree is properly handled
        bst = BinarySearchTree()
        self.assertIsNone(bst.remove(bst.origin, 0))

        """
        (1) test removal all left side (not trigger rotation)
        initial structure:
            2
           / \
          1   3
         /     \
        0       4
        final structure (removing 1, 0):
            2
             \
              3
               \
                4
        """
        bst = BinarySearchTree()
        for i in [2, 1, 3, 4, 0]:
            bst.insert(bst.origin, i)
        bst.remove(bst.origin, 1)  # one child removal
        bst.remove(bst.origin, 0)  # zero child removal, will need rebalancing
        self.assertEqual(3, bst.size)
        self.assertEqual(2, bst.origin.value)
        self.assertEqual(2, bst.origin.height)
        self.assertEqual(3, bst.origin.right.value)
        self.assertEqual(1, bst.origin.right.height)
        self.assertEqual(4, bst.origin.right.right.value)
        self.assertEqual(0, bst.origin.right.right.height)

        """
        (2) test removal all right side (not trigger rotation)
        initial structure:
            3
           / \
          2   4
         /     \
        1       5
        final structure (removing 4, 5):
            3
           /
          2   
         /     
        1       
        """
        bst = BinarySearchTree()
        for i in [3, 2, 4, 1]:
            bst.insert(bst.origin, i)
        bst.remove(bst.origin, 4)  # one child removal
        bst.remove(bst.origin, 5)  # zero child removal, will need rebalancing
        self.assertEqual(3, bst.size)
        self.assertEqual(3, bst.origin.value)
        self.assertEqual(2, bst.origin.height)
        self.assertEqual(2, bst.origin.left.value)
        self.assertEqual(1, bst.origin.left.height)
        self.assertEqual(1, bst.origin.left.left.value)
        self.assertEqual(0, bst.origin.left.left.height)

        """
        (3) test simple 2-child removal
        initial structure:
          2
         / \
        1   3
        final structure (removing 2):
         1 
          \
           3
        """
        bst = BinarySearchTree()
        for i in [2, 1, 3]:
            bst.insert(bst.origin, i)
        bst.remove(bst.origin, 2)  # two child removal
        self.assertEqual(2, bst.size)
        self.assertEqual(1, bst.origin.value)
        self.assertEqual(1, bst.origin.height)
        self.assertEqual(3, bst.origin.right.value)
        self.assertEqual(0, bst.origin.right.height)

        """
        (5) test compounded 2-child removal
        initial structure:
              4
           /     \
          2       6
         / \     / \
        1   3   5   7
        intermediate structure (removing 2, 6):
            4
           / \
          1   5
           \   \
            3   7
        final structure (removing 4)
            3
           / \
          1   5
               \
                7        
        """
        bst = BinarySearchTree()
        for i in [4, 2, 6, 1, 3, 5, 7]:
            bst.insert(bst.origin, i)
        bst.remove(bst.origin, 2)  # two child removal
        bst.remove(bst.origin, 6)  # two child removal
        bst.remove(bst.origin, 4)  # two child removal
        self.assertEqual(4, bst.size)
        self.assertEqual(3, bst.origin.value)
        self.assertEqual(2, bst.origin.height)
        self.assertEqual(1, bst.origin.left.value)
        self.assertEqual(0, bst.origin.left.height)
        self.assertEqual(5, bst.origin.right.value)
        self.assertEqual(1, bst.origin.right.height)
        self.assertEqual(7, bst.origin.right.right.value)
        self.assertEqual(0, bst.origin.right.right.height)

    def test_search(self):

        # ensure empty tree is properly handled
        bst = BinarySearchTree()
        self.assertIsNone(bst.search(bst.origin, 0))

        """
        (1) search small basic tree
        tree structure
          1
         / \
        0   3
           / \
          2   4
        """
        bst = BinarySearchTree()
        numbers = [1, 0, 3, 2, 4]
        for num in numbers:
            bst.insert(bst.origin, num)
        # search existing numbers
        for num in numbers:
            node = bst.search(bst.origin, num)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        # search non-existing numbers and ensure parent of where value would go is returned
        pairs = [(-1, 0), (0.5, 0), (5, 4), (2.5, 2),
                 (3.5, 4), (-1e5, 0), (1e5, 4)]
        for target, closest in pairs:
            node = bst.search(bst.origin, target)
            self.assertIsInstance(node, Node)
            self.assertEqual(closest, node.value)

        """(2) search large random tree"""
        random.seed(331)
        bst = BinarySearchTree()
        numbers = {random.randint(-1000, 1000) for _ in range(1000)}
        for num in numbers:
            bst.insert(bst.origin, num)
        for num in numbers:
            # search existing number
            node = bst.search(bst.origin, num)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)

            # if this node is a leaf, search non-existing numbers around it
            # to ensure it is returned as the parent of where new insertions would go
            if node.left is None and node.right is None:
                node = bst.search(bst.origin, num + 0.1)
                self.assertIsInstance(node, Node)
                self.assertEqual(num, node.value)
                node = bst.search(bst.origin, num - 0.1)
                self.assertIsInstance(node, Node)
                self.assertEqual(num, node.value)


class AVLTreeTests(unittest.TestCase):

    def test_rotate(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.right_rotate(avl.origin))
        self.assertIsNone(avl.left_rotate(avl.origin))

        """
        (1) test basic right
        initial structure:
            3
           /
          2
         /
        1
        final structure:
          2
         / \
        1   3
        """
        avl.origin = Node(3)
        avl.origin.left = Node(2, parent=avl.origin)
        avl.origin.left.left = Node(1, parent=avl.origin.left)
        avl.size = 3

        node = avl.right_rotate(avl.origin)

        self.assertIsInstance(node, Node)

        assert avl.origin.value == 2 and not avl.origin.parent  # root value and parent
        # root left value and parent
        assert avl.origin.left.value == 1 and avl.origin.left.parent == avl.origin
        # shouldn't have children
        assert not (avl.origin.left.left or avl.origin.left.right)
        # root right value and parent
        assert avl.origin.right.value == 3 and avl.origin.right.parent == avl.origin
        # shouldn't have children
        assert not (avl.origin.right.right or avl.origin.right.left)

        """
        (2) test basic left
        initial structure:
        1
         \
          2
           \
            3
        final structure:
          2
         / \
        1   3
        """
        avl = AVLTree()
        avl.origin = Node(1)
        avl.origin.right = Node(2, parent=avl.origin)
        avl.origin.right.right = Node(3, parent=avl.origin.right)
        avl.size = 3

        node = avl.left_rotate(avl.origin)
        self.assertIsInstance(node, Node)

        assert avl.origin.value == 2 and not avl.origin.parent  # root value and parent
        # root left value and parent
        assert avl.origin.left.value == 1 and avl.origin.left.parent == avl.origin
        # shouldn't have any children
        assert not (avl.origin.left.left or avl.origin.left.right)
        # root right value and parent
        assert avl.origin.right.value == 3 and avl.origin.right.parent == avl.origin
        # shouldn't have any children
        assert not (avl.origin.right.right or avl.origin.right.left)

        """
        (3) test intermediate right, rotating at origin
        initial structure:
              7
             / \
            3   10
           / \
          2   4
         /
        1 
        final structure:
            3
           / \
          2   7
         /   / \
        1   4   10
        """
        avl = AVLTree()
        avl.origin = Node(7)
        avl.origin.left = Node(3, parent=avl.origin)
        avl.origin.left.left = Node(2, parent=avl.origin.left)
        avl.origin.left.left.left = Node(1, parent=avl.origin.left.left)
        avl.origin.left.right = Node(4, parent=avl.origin.left)
        avl.origin.right = Node(10, parent=avl.origin)

        node = avl.right_rotate(avl.origin)
        self.assertIsInstance(node, Node)

        # note: node variable names correspond to node values as shown in image above
        node3 = avl.origin
        node2 = avl.origin.left
        node1 = avl.origin.left.left
        node7 = avl.origin.right
        node4 = avl.origin.right.left
        node10 = avl.origin.right.right

        assert node3.value == 3 and not node3.parent
        assert node2.value == 2 and node2.parent == node3 and not node2.right
        assert node1.value == 1 and node1.parent == node2 and not (
                node1.left or node1.right)
        assert node7.value == 7 and node7.parent == node3 and node7.left == node4 and node7.right == node10
        assert node4.value == 4 and node4.parent == node7 and not (
                node4.left or node4.right)
        assert node10.value == 10 and node10.parent == node7 and not (
                node10.left or node10.right)

        """
        (4) test intermediate left, rotating at origin
        initial structure:
          7
         /  \
        3   10
           /   \
          9    11
                 \
                  12
        final structure:
        	10
           /  \
          7   11
         / \    \
        3   9    12
        """
        avl = AVLTree()
        avl.origin = Node(7)
        avl.origin.left = Node(3, parent=avl.origin)
        avl.origin.right = Node(10, parent=avl.origin)
        avl.origin.right.left = Node(9, parent=avl.origin.right)
        avl.origin.right.right = Node(11, parent=avl.origin.right)
        avl.origin.right.right.right = Node(12, parent=avl.origin.right.right)

        node = avl.left_rotate(avl.origin)
        self.assertIsInstance(node, Node)

        # note: node variable names correspond to node values as shown in image above
        node10 = avl.origin
        node7 = avl.origin.left
        node3 = avl.origin.left.left
        node9 = avl.origin.left.right
        node11 = avl.origin.right
        node12 = avl.origin.right.right

        assert node10.value == 10 and not node10.parent
        assert node7.value == 7 and node7.parent == node10
        assert node3.value == 3 and node3.parent == node7 and not (
                node3.left or node3.right)
        assert node9.value == 9 and node9.parent == node7 and not (
                node9.left or node9.right)
        assert node11.value == 11 and node11.parent == node10 and not node11.left
        assert node12.value == 12 and node12.parent == node11 and not (
                node12.left or node12.right)

        """
        (5) test advanced right, rotating not at origin
        initial structure:
        		10
        	   /  \
        	  5	   11
        	 / \     \
        	3	7    12
           / \
          2   4
         /
        1
        final structure:
              10
             /  \
            3    11
           / \     \
          2   5     12
         /   / \
        1   4   7
        """
        avl = AVLTree()
        avl.origin = Node(10)
        avl.origin.right = Node(11, parent=avl.origin)
        avl.origin.right.right = Node(12, parent=avl.origin.right)
        avl.origin.left = Node(5, parent=avl.origin)
        avl.origin.left.right = Node(7, parent=avl.origin.left)
        avl.origin.left.left = Node(3, parent=avl.origin.left)
        avl.origin.left.left.right = Node(4, parent=avl.origin.left.left)
        avl.origin.left.left.left = Node(2, parent=avl.origin.left.left)
        avl.origin.left.left.left.left = Node(
            1, parent=avl.origin.left.left.left)

        node = avl.right_rotate(avl.origin.left)
        self.assertIsInstance(node, Node)

        # note: node variable names correspond to node values as shown in image above
        node10 = avl.origin
        node11 = avl.origin.right
        node12 = avl.origin.right.right
        node3 = avl.origin.left
        node2 = avl.origin.left.left
        node1 = avl.origin.left.left.left
        node5 = avl.origin.left.right
        node4 = avl.origin.left.right.left
        node7 = avl.origin.left.right.right

        assert node10.value == 10 and not node10.parent
        assert node3.value == 3 and node3.parent == node10
        assert node2.value == 2 and node2.parent == node3 and not node2.right
        assert node1.value == 1 and node1.parent == node2 and not (
                node1.left or node1.right)
        assert node5.value == 5 and node5.parent == node3
        assert node4.value == 4 and node4.parent == node5 and not (
                node4.left or node4.right)
        assert node7.value == 7 and node7.parent == node5 and not (
                node7.left or node7.right)
        assert node11.value == 11 and node11.parent == node10 and not node11.left
        assert node12.value == 12 and node12.parent == node11 and not (
                node12.left or node12.right)

        """
        (6) test advanced left, rotating not at origin
        initial structure:
        	3
           / \
          2   10
         /   /  \
        1   5   12
               /  \
              11   13
                     \
                      14
        final structure:
        	3
           / \
          2   12
         /   /  \
        1   10   13
           /  \    \
          5   11   14
        """
        avl = AVLTree()
        avl.origin = Node(3)
        avl.origin.left = Node(2, parent=avl.origin)
        avl.origin.left.left = Node(1, parent=avl.origin.left)
        avl.origin.right = Node(10, parent=avl.origin)
        avl.origin.right.left = Node(5, parent=avl.origin.right)
        avl.origin.right.right = Node(12, parent=avl.origin.right)
        avl.origin.right.right.left = Node(11, parent=avl.origin.right.right)
        avl.origin.right.right.right = Node(13, parent=avl.origin.right.right)
        avl.origin.right.right.right.right = Node(
            14, parent=avl.origin.right.right.right)

        node = avl.left_rotate(avl.origin.right)
        self.assertIsInstance(node, Node)

        # note: node variable names correspond to node values as shown in image above
        node3 = avl.origin
        node2 = avl.origin.left
        node1 = avl.origin.left.left
        node12 = avl.origin.right
        node10 = avl.origin.right.left
        node5 = avl.origin.right.left.left
        node11 = avl.origin.right.left.right
        node13 = avl.origin.right.right
        node14 = avl.origin.right.right.right

        assert node3.value == 3 and not node3.parent
        assert node2.value == 2 and node2.parent == node3 and not node2.right
        assert node1.value == 1 and node1.parent == node2 and not (
                node1.left or node1.right)
        assert node12.value == 12 and node12.parent == node3
        assert node10.value == 10 and node10.parent == node12
        assert node5.value == 5 and node5.parent == node10 and not (
                node5.left or node5.right)
        assert node11.value == 11 and node11.parent == node10 and not (
                node11.left or node11.right)
        assert node13.value == 13 and node13.parent == node12 and not node13.left
        assert node14.value == 14 and node14.parent == node13 and not (
                node14.left or node14.right)

    def test_balance_factor(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertEqual(0, avl.balance_factor(avl.origin))

        """
        (1) test on balanced tree
        structure:
          2
         / \
        1   3
        """
        avl.origin = Node(2)
        avl.origin.height = 1
        avl.origin.left = Node(1, parent=avl.origin)
        avl.origin.left.height = 0
        avl.origin.right = Node(3, parent=avl.origin)
        avl.origin.right.height = 0
        avl.size = 3

        self.assertEqual(0, avl.balance_factor(avl.origin))
        self.assertEqual(0, avl.balance_factor(avl.origin.left))
        self.assertEqual(0, avl.balance_factor(avl.origin.right))

        """
        (2) test on unbalanced left
        structure:
            3
           /
          2
         /
        1
        """
        avl = AVLTree()
        avl.origin = Node(3)
        avl.origin.height = 2
        avl.origin.left = Node(2, parent=avl.origin)
        avl.origin.left.height = 1
        avl.origin.left.left = Node(1, parent=avl.origin.left)
        avl.origin.left.left.height = 0
        avl.size = 3

        self.assertEqual(2, avl.balance_factor(avl.origin))
        self.assertEqual(1, avl.balance_factor(avl.origin.left))
        self.assertEqual(0, avl.balance_factor(avl.origin.left.left))

        """
        (2) test on unbalanced right
        structure:
        1
         \
          2
           \
            3
        """
        avl = AVLTree()
        avl.origin = Node(1)
        avl.origin.height = 2
        avl.origin.right = Node(2, parent=avl.origin)
        avl.origin.right.height = 1
        avl.origin.right.right = Node(3, parent=avl.origin.right)
        avl.origin.right.right.height = 0
        avl.size = 3

        self.assertEqual(-2, avl.balance_factor(avl.origin))
        self.assertEqual(-1, avl.balance_factor(avl.origin.right))
        self.assertEqual(0, avl.balance_factor(avl.origin.right.right))

    def test_rebalance(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.rebalance(avl.origin))

        """
        (1) test balanced tree (do nothing)
        initial and final structure:
          2
         / \
        1   3
        since pointers are already tested in rotation testcase, only check values and heights
        """
        avl.origin = Node(2)
        avl.origin.height = 1
        avl.origin.left = Node(1, parent=avl.origin)
        avl.origin.left.height = 0
        avl.origin.right = Node(3, parent=avl.origin)
        avl.origin.right.height = 0
        avl.size = 3

        node = avl.rebalance(avl.origin)
        self.assertIsInstance(node, Node)

        self.assertEqual(2, avl.origin.value)
        self.assertEqual(1, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(3, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)

        """
        (2) test left-left rebalance
        initial structure:
            4
           /
          2
         / \
        1   3
        final structure:
          2
         / \
        1   4
           /
          3
        """
        avl = AVLTree()
        avl.origin = Node(4)
        avl.origin.height = 2
        avl.origin.left = Node(2, parent=avl.origin)
        avl.origin.left.height = 1
        avl.origin.left.left = Node(1, parent=avl.origin.left)
        avl.origin.left.left.height = 0
        avl.origin.left.right = Node(3, parent=avl.origin.left)
        avl.origin.left.right.height = 0
        avl.size = 4

        node = avl.rebalance(avl.origin)
        self.assertIsInstance(node, Node)

        self.assertEqual(2, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(4, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(3, avl.origin.right.left.value)
        self.assertEqual(0, avl.origin.right.left.height)

        """
        (2) test right-right rebalance
        initial structure:
        1
         \
          3
         /  \
        2    4
        final structure:
          3
         / \
        1   4
         \
          2
        """
        avl = AVLTree()
        avl.origin = Node(1)
        avl.origin.height = 2
        avl.origin.right = Node(3, parent=avl.origin)
        avl.origin.right.height = 1
        avl.origin.right.right = Node(4, parent=avl.origin.right)
        avl.origin.right.right.height = 0
        avl.origin.right.left = Node(2, parent=avl.origin.right)
        avl.origin.right.left.height = 0
        avl.size = 4

        node = avl.rebalance(avl.origin)
        self.assertIsInstance(node, Node)

        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(4, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)
        self.assertEqual(2, avl.origin.left.right.value)
        self.assertEqual(0, avl.origin.left.right.height)

        """
        (4) test left-right rebalance
        initial structure:
            5
           / \
          2   6
         / \
        1   3
             \
              4
        intermediate structure:
              5
             / \
            3   6
           / \
          2   4
         /
        1
        final structure:
            3 
           / \
          2   5
         /   / \
        1   4   6
        """
        avl = AVLTree()
        avl.origin = Node(5)
        avl.origin.height = 3
        avl.origin.left = Node(2, parent=avl.origin)
        avl.origin.left.height = 2
        avl.origin.right = Node(6, parent=avl.origin)
        avl.origin.right.height = 0
        avl.origin.left.left = Node(1, parent=avl.origin.left)
        avl.origin.left.left.height = 0
        avl.origin.left.right = Node(3, parent=avl.origin.left)
        avl.origin.left.right.height = 1
        avl.origin.left.right.right = Node(4, parent=avl.origin.left.right)
        avl.origin.left.right.right.height = 0

        node = avl.rebalance(avl.origin)
        self.assertIsInstance(node, Node)

        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(1, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(4, avl.origin.right.left.value)
        self.assertEqual(0, avl.origin.right.left.height)
        self.assertEqual(6, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

        """
        (5) test right-left rebalance
        initial structure:
          2
         / \
        1   5
           / \
          4   6
         /
        3
        intermediate structure:
          2
         / \
        1   4
           / \
          3   5
               \
                6
        final structure:
            4 
           / \
          2   5
         / \   \
        1   3   6
        """
        avl = AVLTree()
        avl.origin = Node(2)
        avl.origin.height = 3
        avl.origin.left = Node(1, parent=avl.origin)
        avl.origin.left.height = 0
        avl.origin.right = Node(5, parent=avl.origin)
        avl.origin.right.height = 2
        avl.origin.right.left = Node(4, parent=avl.origin.right)
        avl.origin.right.left.height = 1
        avl.origin.right.right = Node(6, parent=avl.origin.right)
        avl.origin.right.right.height = 0
        avl.origin.right.left.left = Node(3, parent=avl.origin.right.left)
        avl.origin.right.left.left.height = 0

        node = avl.rebalance(avl.origin)
        self.assertIsInstance(node, Node)

        self.assertEqual(4, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(1, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(3, avl.origin.left.right.value)
        self.assertEqual(0, avl.origin.left.right.height)
        self.assertEqual(6, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

    def test_insert(self):

        # visualize this testcase with https://www.cs.usfca.edu/~galles/visualization/AVLtree.html
        avl = AVLTree()
        """
        (1) test insertion causing right-right rotation
        final structure
          1
         / \
        0   3
           / \
          2   4
        """
        for i in range(5):
            node = avl.insert(avl.origin, i)
            self.assertIsInstance(node, Node)

        self.assertEqual(5, avl.size)
        self.assertEqual(1, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(0, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(3, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(2, avl.origin.right.left.value)
        self.assertEqual(0, avl.origin.right.left.height)
        self.assertEqual(4, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

        """
        (2) test insertion causing left-left rotation
        final structure
            3
           / \
          1   4
         / \
        0   2
        """
        avl = AVLTree()
        for i in range(4, -1, -1):
            node = avl.insert(avl.origin, i)
            self.assertIsInstance(node, Node)
        self.assertEqual(5, avl.size)
        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(4, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)
        self.assertEqual(0, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(2, avl.origin.left.right.value)
        self.assertEqual(0, avl.origin.left.right.height)

        """
        (3) test insertion (with duplicates) causing left-right rotation
        initial structure:
            5
           / \
          2   6
         / \
        1   3
             \
              4
        final structure:
            3 
           / \
          2   5
         /   / \
        1   4   6
        """
        avl = AVLTree()
        for i in [5, 2, 6, 1, 3] * 2 + [4]:
            node = avl.insert(avl.origin, i)
            self.assertIsInstance(node, Node)
        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(1, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(4, avl.origin.right.left.value)
        self.assertEqual(0, avl.origin.right.left.height)
        self.assertEqual(6, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

        """
        (4) test insertion (with duplicates) causing right-left rotation
        initial structure:
          2
         / \
        1   5
           / \
          4   6
         /
        3
        final structure:
            4 
           / \
          2   5
         / \   \
        1   3   6
        """
        avl = AVLTree()
        for i in [2, 1, 5, 4, 6] * 2 + [3]:
            node = avl.insert(avl.origin, i)
            self.assertIsInstance(node, Node)
        self.assertEqual(4, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(1, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(3, avl.origin.left.right.value)
        self.assertEqual(0, avl.origin.left.right.height)
        self.assertEqual(6, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

    def test_min(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.min(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        min_node = avl.min(avl.origin)
        self.assertIsInstance(min_node, Node)
        self.assertEqual(0, min_node.value)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-100, 101):
            avl.insert(avl.origin, i)
        min_node = avl.min(avl.origin)
        self.assertIsInstance(min_node, Node)
        self.assertEqual(-100, min_node.value)

        """(3) large random tree"""
        random.seed(331)
        avl = AVLTree()
        numbers = [random.randint(-1000, 1000) for _ in range(1000)]
        for num in numbers:
            avl.insert(avl.origin, num)
        min_node = avl.min(avl.origin)
        self.assertIsInstance(min_node, Node)
        self.assertEqual(min(numbers), min_node.value)

    def test_max(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.max(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        max_node = avl.max(avl.origin)
        self.assertIsInstance(max_node, Node)
        self.assertEqual(9, max_node.value)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-100, 101):
            avl.insert(avl.origin, i)
        max_node = avl.max(avl.origin)
        self.assertIsInstance(max_node, Node)
        self.assertEqual(100, max_node.value)

        """(3) large random tree"""
        random.seed(331)
        avl = AVLTree()
        numbers = [random.randint(-1000, 1000) for _ in range(1000)]
        for num in numbers:
            avl.insert(avl.origin, num)
        max_node = avl.max(avl.origin)
        self.assertIsInstance(max_node, Node)
        self.assertEqual(max(numbers), max_node.value)

    def test_search(self):

        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.search(avl.origin, 0))

        """
        (1) search small basic tree
        tree structure
          1
         / \
        0   3
           / \
          2   4
        """
        avl = AVLTree()
        numbers = [1, 0, 3, 2, 4]
        for num in numbers:
            avl.insert(avl.origin, num)
        # search existing numbers
        for num in numbers:
            node = avl.search(avl.origin, num)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        # search non-existing numbers and ensure parent of where value would go is returned
        pairs = [(-1, 0), (0.5, 0), (5, 4), (2.5, 2),
                 (3.5, 4), (-1e5, 0), (1e5, 4)]
        for target, closest in pairs:
            node = avl.search(avl.origin, target)
            self.assertIsInstance(node, Node)
            self.assertEqual(closest, node.value)

        """(2) search large random tree"""
        random.seed(331)
        avl = AVLTree()
        numbers = {random.randint(-1000, 1000) for _ in range(1000)}
        for num in numbers:
            avl.insert(avl.origin, num)
        for num in numbers:
            # search existing number
            node = avl.search(avl.origin, num)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)

            # if this node is a leaf, search non-existing numbers around it
            # to ensure it is returned as the parent of where new insertions would go
            if node.left is None and node.right is None:
                node = avl.search(avl.origin, num + 0.1)
                self.assertIsInstance(node, Node)
                self.assertEqual(num, node.value)
                node = avl.search(avl.origin, num - 0.1)
                self.assertIsInstance(node, Node)
                self.assertEqual(num, node.value)

    def test_inorder(self):

        # note: Python generators will raise a StopIteration exception when there are no items
        # left to yield, and we test for this exception to ensure each traversal yields the correct
        # number of items: https://docs.python.org/3/library/exceptions.html#StopIteration

        # ensure empty tree is properly handled and returns a StopIteration
        avl = AVLTree()
        with self.assertRaises(StopIteration):
            next(avl.inorder(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        generator = avl.inorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = list(range(10))
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-100, 101):
            avl.insert(avl.origin, i)
        generator = avl.inorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = list(range(-100, 101))
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(3) large random tree of unique numbers"""
        random.seed(331)
        avl = AVLTree()
        numbers = {random.randint(-1000, 1000) for _ in range(80)}
        for num in numbers:
            avl.insert(avl.origin, num)
        generator = avl.inorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = sorted(numbers)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(4) Testing tree is iterable. Hint: Implement the __iter__ function."""
        for expected_val, actual in zip(expected, avl):
            self.assertEqual(expected_val, actual.value)

    def test_preorder(self):

        # note: Python generators will raise a StopIteration exception when there are no items
        # left to yield, and we test for this exception to ensure each traversal yields the correct
        # number of items: https://docs.python.org/3/library/exceptions.html#StopIteration

        # ensure empty tree is properly handled and returns a StopIteration
        avl = AVLTree()
        with self.assertRaises(StopIteration):
            next(avl.preorder(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        generator = avl.preorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [3, 1, 0, 2, 7, 5, 4, 6, 8, 9]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-20, 21):
            avl.insert(avl.origin, i)
        generator = avl.preorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [-5, -13, -17, -19, -20, -18, -15, -16, -14, -9, -11,
                    -12, -10, -7, -8, -6, 11, 3, -1, -3, -4, -2, 1, 0, 2,
                    7, 5, 4, 6, 9, 8, 10, 15, 13, 12, 14, 17, 16, 19, 18,
                    20]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(3) large random tree of unique numbers"""
        random.seed(331)
        avl = AVLTree()
        numbers = {random.randint(-1000, 1000) for _ in range(80)}
        for num in numbers:
            avl.insert(avl.origin, num)
        generator = avl.preorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [527, 33, -493, -834, -933, -954, -918, -655, -720,
                    -789, -705, -650, -529, -165, -343, -422, -434,
                    -359, -312, -324, -269, -113, -142, -148, -116, -43,
                    -89, -26, 327, 220, 108, 77, 44, 101, 193, 113,
                    194, 274, 251, 224, 268, 294, 283, 316, 454, 362, 358,
                    333, 360, 431, 411, 446, 486, 485, 498, 503,
                    711, 574, 565, 529, 571, 675, 641, 687, 832, 776, 733,
                    720, 775, 784, 782, 802, 914, 860, 843, 888,
                    966, 944, 975]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_postorder(self):

        # note: Python generators will raise a StopIteration exception when there are no items
        # left to yield, and we test for this exception to ensure each traversal yields the correct
        # number of items: https://docs.python.org/3/library/exceptions.html#StopIteration

        # ensure empty tree is properly handled and returns a StopIteration
        avl = AVLTree()
        with self.assertRaises(StopIteration):
            next(avl.postorder(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        generator = avl.postorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [0, 2, 1, 4, 6, 5, 9, 8, 7, 3]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-20, 20):
            avl.insert(avl.origin, i)
        generator = avl.postorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [-20, -18, -19, -16, -14, -15, -17, -12, -10, -11, -8, -6, -7, -9,
                    -13, -4, -2, -3, 0, 2, 1, -1, 4, 6, 5, 8, 10, 9, 7, 3, 12, 14, 13,
                    16, 19, 18, 17, 15, 11, -5]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(3) large random tree of unique numbers"""
        random.seed(331)
        avl = AVLTree()
        numbers = {random.randint(-1000, 1000) for _ in range(80)}
        for num in numbers:
            avl.insert(avl.origin, num)
        generator = avl.postorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [-954, -918, -933, -789, -705, -720, -529, -650, -655, -834, -434, -359, -422, -324, -269, -312,
                    -343, -148, -116, -142, -89, -26, -43, -113, -165, -493, 44, 101, 77, 113, 194, 193, 108, 224,
                    268, 251, 283, 316, 294, 274, 220, 333, 360, 358, 411, 446, 431, 362, 485, 503, 498, 486, 454,
                    327, 33, 529, 571, 565, 641, 687, 675, 574, 720, 775, 733, 782, 802, 784, 776, 843, 888, 860,
                    944, 975, 966, 914, 832, 711, 527]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_levelorder(self):

        # note: Python generators will raise a StopIteration exception when there are no items
        # left to yield, and we test for this exception to ensure each traversal yields the correct
        # number of items: https://docs.python.org/3/library/exceptions.html#StopIteration

        # ensure empty tree is properly handled and returns a StopIteration
        avl = AVLTree()
        with self.assertRaises(StopIteration):
            next(avl.levelorder(avl.origin))

        """(1) small sequential tree"""
        for i in range(10):
            avl.insert(avl.origin, i)
        generator = avl.levelorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [3, 1, 7, 0, 2, 5, 8, 4, 6, 9]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(2) large sequential tree"""
        avl = AVLTree()
        for i in range(-20, 20):
            avl.insert(avl.origin, i)
        generator = avl.levelorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [-5, -13, 11, -17, -9, 3, 15, -19, -15, -11, -7, -1, 7, 13, 17, -20, -18,
                    -16, -14, -12, -10, -8, -6, -3, 1, 5, 9, 12, 14, 16, 18, -4, -2, 0, 2,
                    4, 6, 8, 10, 19]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        """(3) large random tree of unique numbers"""
        random.seed(331)
        avl = AVLTree()
        numbers = {random.randint(-1000, 1000) for _ in range(80)}
        for num in numbers:
            avl.insert(avl.origin, num)
        generator = avl.levelorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        expected = [527, 33, 711, -493, 327, 574, 832, -834, -165, 220, 454,
                    565, 675, 776, 914, -933, -655, -343, -113, 108, 274,
                    362, 486, 529, 571, 641, 687, 733, 784, 860, 966, -954,
                    -918, -720, -650, -422, -312, -142, -43, 77, 193, 251,
                    294, 358, 431, 485, 498, 720, 775, 782, 802, 843, 888,
                    944, 975, -789, -705, -529, -434, -359, -324, -269, -148,
                    -116, -89, -26, 44, 101, 113, 194, 224, 268, 283, 316, 333,
                    360, 411, 446, 503]
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_remove(self):

        # visualize this testcase with https://www.cs.usfca.edu/~galles/visualization/AVLtree.html
        # ensure empty tree is properly handled
        avl = AVLTree()
        self.assertIsNone(avl.remove(avl.origin, 0))

        """
        (1) test removal causing right-right rotation
        initial structure:
            2
           / \
          1   3
         /     \
        0       4
        final structure (removing 1, 0):
          3 
         / \
        2   4
        """
        avl = AVLTree()
        for i in [2, 1, 3, 4, 0]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 1)  # one child removal
        avl.remove(avl.origin, 0)  # zero child removal, will need rebalancing
        self.assertEqual(3, avl.size)
        self.assertEqual(3, avl.origin.value)
        self.assertEqual(1, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(4, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)

        """
        (2) test removal causing left-left rotation
        initial structure:
            3
           / \
          2   4
         /     \
        1       5
        final structure (removing 4, 5):
          2 
         / \
        1   3
        """
        avl = AVLTree()
        for i in [3, 2, 4, 1]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 4)  # one child removal
        avl.remove(avl.origin, 5)  # zero child removal, will need rebalancing
        self.assertEqual(3, avl.size)
        self.assertEqual(2, avl.origin.value)
        self.assertEqual(1, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(3, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)

        """
        (3) test removal causing left-right rotation
        initial structure:
              5
             / \
            2   6
           / \   \
          1   3   7
         /     \
        0       4
        final structure (removing 1, 6):
            3 
           / \
          2   5
         /   / \
        0   4   7
        """
        avl = AVLTree()
        for i in [5, 2, 6, 1, 3, 7, 0, 4]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 1)  # one child removal
        avl.remove(avl.origin, 6)  # one child removal, will need rebalancing
        self.assertEqual(6, avl.size)
        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(0, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(4, avl.origin.right.left.value)
        self.assertEqual(0, avl.origin.right.left.height)
        self.assertEqual(7, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

        """
        (4) test removal causing right-left rotation
        initial structure:
            2
           / \
          1   5
         /   / \
        0   4   6
           /     \
          3       7
        final structure (removing 6, 1):
            4 
           / \
          2   5
         / \   \
        0   3   7
        """
        avl = AVLTree()
        for i in [2, 1, 5, 0, 4, 6, 3, 7]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 6)  # one child removal
        avl.remove(avl.origin, 1)  # one child removal, will need rebalancing
        self.assertEqual(6, avl.size)
        self.assertEqual(4, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(2, avl.origin.left.value)
        self.assertEqual(1, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(7, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)
        self.assertEqual(0, avl.origin.left.left.value)
        self.assertEqual(0, avl.origin.left.left.height)
        self.assertEqual(3, avl.origin.left.right.value)
        self.assertEqual(0, avl.origin.left.right.height)

        """
        (5) test simple 2-child removal
        initial structure:
          2
         / \
        1   3
        final structure (removing 2):
         1 
          \
           3
        """
        avl = AVLTree()
        for i in [2, 1, 3]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 2)  # two child removal
        self.assertEqual(2, avl.size)
        self.assertEqual(1, avl.origin.value)
        self.assertEqual(1, avl.origin.height)
        self.assertEqual(3, avl.origin.right.value)
        self.assertEqual(0, avl.origin.right.height)

        """
        (5) test compounded 2-child removal
        initial structure:
              4
           /     \
          2       6
         / \     / \
        1   3   5   7
        intermediate structure (removing 2, 6):
            4
           / \
          1   5
           \   \
            3   7
        final structure (removing 4)
            3
           / \
          1   5
               \
                7        
        """
        avl = AVLTree()
        for i in [4, 2, 6, 1, 3, 5, 7]:
            avl.insert(avl.origin, i)
        avl.remove(avl.origin, 2)  # two child removal
        avl.remove(avl.origin, 6)  # two child removal
        avl.remove(avl.origin, 4)  # two child removal
        self.assertEqual(4, avl.size)
        self.assertEqual(3, avl.origin.value)
        self.assertEqual(2, avl.origin.height)
        self.assertEqual(1, avl.origin.left.value)
        self.assertEqual(0, avl.origin.left.height)
        self.assertEqual(5, avl.origin.right.value)
        self.assertEqual(1, avl.origin.right.height)
        self.assertEqual(7, avl.origin.right.right.value)
        self.assertEqual(0, avl.origin.right.right.height)

    def test_AVL_comprehensive(self):

        # visualize some of test in this testcase with https://www.cs.usfca.edu/~galles/visualization/AVLtree.html
        # ensure empty tree is properly handled
        """
        First part, inserting and removing without rotation

        insert without any rotation (inserting 5, 0, 10):
          5
         / \
        1   10
        """

        def check_node_properties(current: Node, value: int = 0, height: int = 0, balance: int = 0):
            if value is None:
                self.assertIsNone(current)
                return
            self.assertEqual(value, current.value)
            self.assertEqual(height, current.height)
            self.assertEqual(balance, avl.balance_factor(current))

        avl = AVLTree()
        avl.insert(avl.origin, 5)
        avl.insert(avl.origin, 1)
        avl.insert(avl.origin, 10)
        self.assertEqual(3, avl.size)
        self.assertEqual(1, avl.min(avl.origin).value)
        self.assertEqual(10, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=5, height=1, balance=0)
        check_node_properties(avl.origin.left, value=1, height=0, balance=0)
        check_node_properties(avl.origin.right, value=10, height=0, balance=0)
        """
        Current AVL tree:
          5
         / \
        1   10
        After Removing 5:
          1
           \
            10
        """
        avl.remove(avl.origin, 5)
        self.assertEqual(2, avl.size)
        self.assertEqual(1, avl.min(avl.origin).value)
        self.assertEqual(10, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=1, height=1, balance=-1)
        check_node_properties(avl.origin.left, value=None)
        check_node_properties(avl.origin.right, value=10, height=0, balance=0)
        """
        Current AVL tree:
          1
            \
            10
        After inserting 0, 20:
          1
         /  \
        0   10
              \
               20
        """
        avl.insert(avl.origin, 0)
        avl.insert(avl.origin, 20)
        self.assertEqual(4, avl.size)
        self.assertEqual(0, avl.min(avl.origin).value)
        self.assertEqual(20, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=1, height=2, balance=-1)
        check_node_properties(avl.origin.left, value=0, height=0, balance=0)
        check_node_properties(avl.origin.right, value=10, height=1, balance=-1)
        check_node_properties(avl.origin.right.right,
                              value=20, height=0, balance=0)

        """
        Current AVL tree:
          1
         /  \
        0   10
              \
               20
        After removing 20, inserting -20 and inserting 5
             1
            /  \
           0   10
          /   /
        -20  5
        """
        avl.remove(avl.origin, 20)
        avl.insert(avl.origin, -20)
        avl.insert(avl.origin, 5)
        self.assertEqual(5, avl.size)
        self.assertEqual(-20, avl.min(avl.origin).value)
        self.assertEqual(10, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=1, height=2, balance=0)
        check_node_properties(avl.origin.left, value=0, height=1, balance=1)
        check_node_properties(avl.origin.left.left,
                              value=-20, height=0, balance=0)
        check_node_properties(avl.origin.right, value=10, height=1, balance=1)
        check_node_properties(avl.origin.right.left,
                              value=5, height=0, balance=0)

        """
        Second part, inserting and removing with rotation

        inserting 5, 10:
          5
           \
            10
        """
        avl = AVLTree()
        avl.insert(avl.origin, 5)
        avl.insert(avl.origin, 10)
        self.assertEqual(2, avl.size)
        self.assertEqual(5, avl.min(avl.origin).value)
        self.assertEqual(10, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=5, height=1, balance=-1)
        check_node_properties(avl.origin.right, value=10, height=0, balance=0)
        """
        Current AVL tree:
          5
           \
            10
        After inserting 8, 9, 12
           8
         /   \
        5    10
            /  \
           9   12
        """
        avl.insert(avl.origin, 8)
        avl.insert(avl.origin, 9)
        avl.insert(avl.origin, 12)
        self.assertEqual(5, avl.size)
        self.assertEqual(5, avl.min(avl.origin).value)
        self.assertEqual(12, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=8, height=2, balance=-1)
        check_node_properties(avl.origin.right, value=10, height=1, balance=0)
        check_node_properties(avl.origin.right.left,
                              value=9, height=0, balance=0)
        check_node_properties(avl.origin.right.right,
                              value=12, height=0, balance=0)
        check_node_properties(avl.origin.left, value=5, height=0, balance=0)

        """
        Current AVL tree:
           8
         /   \
        5    10
            /  \
           9   12
        After inserting 3, 1, 2
               8
           /       \
          3        10
         /  \     /   \
        1    5   9    12
          \
           2
        """
        avl.insert(avl.origin, 3)
        avl.insert(avl.origin, 1)
        avl.insert(avl.origin, 2)
        self.assertEqual(8, avl.size)
        self.assertEqual(1, avl.min(avl.origin).value)
        self.assertEqual(12, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=8, height=3, balance=1)
        check_node_properties(avl.origin.right, value=10, height=1, balance=0)
        check_node_properties(avl.origin.right.left,
                              value=9, height=0, balance=0)
        check_node_properties(avl.origin.right.right,
                              value=12, height=0, balance=0)
        check_node_properties(avl.origin.left, value=3, height=2, balance=1)
        check_node_properties(avl.origin.left.left,
                              value=1, height=1, balance=-1)
        check_node_properties(avl.origin.left.left.right,
                              value=2, height=0, balance=0)
        check_node_properties(avl.origin.left.right,
                              value=5, height=0, balance=0)
        """
        Current AVL tree:
               8
           /       \
          3        10
         /  \     /   \
        1    5   9    12
          \
           2
        After removing 5
               8
           /       \
          2        10
         /  \     /   \
        1    3   9    12
        """
        avl.remove(avl.origin, 5)
        self.assertEqual(7, avl.size)
        self.assertEqual(1, avl.min(avl.origin).value)
        self.assertEqual(12, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=8, height=2, balance=0)
        check_node_properties(avl.origin.right, value=10, height=1, balance=0)
        check_node_properties(avl.origin.right.left,
                              value=9, height=0, balance=0)
        check_node_properties(avl.origin.right.right,
                              value=12, height=0, balance=0)
        check_node_properties(avl.origin.left, value=2, height=1, balance=0)
        check_node_properties(avl.origin.left.left,
                              value=1, height=0, balance=0)
        check_node_properties(avl.origin.left.right,
                              value=3, height=0, balance=0)
        """
        Current AVL tree:
              8
           /     \
          2      10
         /  \   /   \
        1    3 9    12
        After inserting 5, 13, 0, 7, 20
               8
            /       \
           2         10
          /  \      /   \
         1    5     9    13
        /    / \        /  \
        0   3   7     12    20
        """
        avl.insert(avl.origin, 5)
        avl.insert(avl.origin, 13)
        avl.insert(avl.origin, 0)
        avl.insert(avl.origin, 7)
        avl.insert(avl.origin, 20)
        self.assertEqual(12, avl.size)
        self.assertEqual(0, avl.min(avl.origin).value)
        self.assertEqual(20, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=8, height=3, balance=0)

        check_node_properties(avl.origin.right, value=10, height=2, balance=-1)
        check_node_properties(avl.origin.right.left,
                              value=9, height=0, balance=0)
        check_node_properties(avl.origin.right.right,
                              value=13, height=1, balance=0)
        check_node_properties(avl.origin.right.right.right,
                              value=20, height=0, balance=0)
        check_node_properties(avl.origin.right.right.left,
                              value=12, height=0, balance=0)

        check_node_properties(avl.origin.left, value=2, height=2, balance=0)
        check_node_properties(avl.origin.left.left,
                              value=1, height=1, balance=1)
        check_node_properties(avl.origin.left.left.left,
                              value=0, height=0, balance=0)
        check_node_properties(avl.origin.left.right,
                              value=5, height=1, balance=-0)
        check_node_properties(avl.origin.left.right.right,
                              value=7, height=0, balance=0)
        check_node_properties(avl.origin.left.right.left,
                              value=3, height=0, balance=0)

        """
        Current AVL tree:
               8
            /       \
           2         10
          /  \      /   \
         1    5     9    13
        /    / \        /  \
        0   3   7     12    20
        After removing 1, 9
               8
            /       \
           2         13
          /  \      /   \
         0    5   10     20
             / \     \    
             3   7    12
        """
        avl.remove(avl.origin, 1)
        avl.remove(avl.origin, 9)
        self.assertEqual(10, avl.size)
        self.assertEqual(0, avl.min(avl.origin).value)
        self.assertEqual(20, avl.max(avl.origin).value)
        # Properties of all nodes
        check_node_properties(avl.origin, value=8, height=3, balance=0)

        check_node_properties(avl.origin.right, value=13, height=2, balance=1)
        check_node_properties(avl.origin.right.left,
                              value=10, height=1, balance=-1)
        check_node_properties(avl.origin.right.left.right,
                              value=12, height=0, balance=0)
        check_node_properties(avl.origin.right.right,
                              value=20, height=0, balance=0)

        check_node_properties(avl.origin.left, value=2, height=2, balance=-1)
        check_node_properties(avl.origin.left.left,
                              value=0, height=0, balance=0)
        check_node_properties(avl.origin.left.right,
                              value=5, height=1, balance=-0)
        check_node_properties(avl.origin.left.right.right,
                              value=7, height=0, balance=0)
        check_node_properties(avl.origin.left.right.left,
                              value=3, height=0, balance=0)

        """
        Part Three
        Everything but random, checking properties of tree only
        """
        random.seed(331)
        """
        randomly insert, and remove alphabet to avl tree
        """

        def random_order_1(character=True):
            order = random.randint(0, 2)
            if not len(existing_value) or (order and (not character or avl.size < 26)):
                if character:
                    inserted = chr(ord('a') + random.randint(0, 25))
                    while inserted in existing_value:
                        inserted = chr(ord('a') + random.randint(0, 25))
                else:
                    inserted = random.randint(0, 100000)
                avl.insert(avl.origin, inserted)
                existing_value[inserted] = 1
            else:
                removed = random.choice(list(existing_value.keys()))
                avl.remove(avl.origin, removed)
                existing_value.pop(removed)

        existing_value = {}
        avl = AVLTree()
        for _ in range(30):
            random_order_1()
        self.assertEqual('a', avl.min(avl.origin).value)
        self.assertEqual('y', avl.max(avl.origin).value)
        # inorder test
        expected = ['a', 'b', 'd', 'f', 'g', 'i', 'k',
                    'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
        generator = avl.inorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        expected = ['p', 'f', 'b', 'a', 'd', 'k', 'i',
                    'g', 'o', 't', 'r', 'q', 's', 'w', 'v', 'y']
        generator = avl.preorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        expected = ['a', 'd', 'b', 'g', 'i', 'o', 'k',
                    'f', 'q', 's', 'r', 'v', 'y', 'w', 't', 'p']
        generator = avl.postorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        existing_value.clear()
        avl = AVLTree()
        for _ in range(150):
            random_order_1(character=False)
        self.assertEqual(3113, avl.min(avl.origin).value)
        self.assertEqual(99254, avl.max(avl.origin).value)
        # inorder test
        expected = [3113, 4842, 8476, 9661, 9691, 9849, 12004, 13818, 16748, 19125,
                    20633, 20815, 20930, 25633, 25790, 28476, 29509, 30303, 30522,
                    32151, 32253, 35293, 35691, 36623, 37047, 40980, 41185, 42559,
                    43298, 44521, 44698, 45027, 46070, 46204, 46876, 49122, 51761,
                    51864, 54480, 55579, 56007, 56230, 58594, 59094, 59240, 59245,
                    61029, 61837, 63796, 66866, 69402, 69498, 70575, 70733, 74185,
                    74291, 74893, 76608, 76840, 77762, 78162, 78215, 80089, 80883,
                    85118, 86927, 88662, 91673, 94661, 94848, 98575, 99254]

        generator = avl.inorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        expected = [49122, 35691, 20815, 9849, 4842, 3113, 9661, 8476, 9691, 19125,
                    13818, 12004, 16748, 20633, 30303, 25790, 20930, 25633, 29509,
                    28476, 32253, 30522, 32151, 35293, 43298, 37047, 36623, 41185,
                    40980, 42559, 46070, 44698, 44521, 45027, 46204, 46876, 69498,
                    58594, 54480, 51761, 51864, 56007, 55579, 56230, 59245, 59240,
                    59094, 61837, 61029, 66866, 63796, 69402, 80883, 76840, 74185,
                    70575, 70733, 74893, 74291, 76608, 78162, 77762, 80089, 78215,
                    91673, 86927, 85118, 88662, 94848, 94661, 99254, 98575]

        generator = avl.preorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

        expected = [3113, 8476, 9691, 9661, 4842, 12004, 16748, 13818,
                    20633, 19125, 9849, 25633, 20930, 28476, 29509,
                    25790, 32151, 30522, 35293, 32253, 30303, 20815,
                    36623, 40980, 42559, 41185, 37047, 44521, 45027,
                    44698, 46876, 46204, 46070, 43298, 35691, 51864,
                    51761, 55579, 56230, 56007, 54480, 59094, 59240,
                    61029, 63796, 69402, 66866, 61837, 59245, 58594,
                    70733, 70575, 74291, 76608, 74893, 74185, 77762,
                    78215, 80089, 78162, 76840, 85118, 88662, 86927,
                    94661, 98575, 99254, 94848, 91673, 80883, 69498, 49122]

        generator = avl.postorder(avl.origin)
        self.assertIsInstance(generator, types.GeneratorType)
        for num in expected:
            node = next(generator)
            self.assertIsInstance(node, Node)
            self.assertEqual(num, node.value)
        with self.assertRaises(StopIteration):
            next(generator)

    def test_nnc(self):
        plot = False

        """
        (1) Day/Night image classification: Suppose brightness of an image is measured
        between 0 and 1, and we are provided labeled examples of the brightness levels of 
        images that were taken during night and during day. Given a new image brightness level,
        predict whether the image was taken during night or day.
        """
        # 1a: test from specs
        data = [(0.18, "night"), (0.21, "night"), (0.29, "night"),
                (0.49, "night"), (0.51, "day"), (0.53, "day"),
                (0.97, "day"), (0.98, "day"), (0.99, "day")]
        nnc = NearestNeighborClassifier(resolution=1)
        nnc.fit(data)

        test_images = [0.1, 0.2, 0.5, 0.8, 0.9]
        expected = ["night", "night", "day", None, "day"]
        actual = [nnc.predict(x=image, delta=0.1) for image in test_images]
        self.assertEqual(expected, actual)

        # 1b: larger test
        random.seed(331)
        night_images = [(random.random() / 2, "night") for _ in range(50)]
        day_images = [(random.random() / 2 + 0.5, "day") for _ in range(50)]
        data = night_images + day_images

        nnc = NearestNeighborClassifier(resolution=1)
        nnc.fit(data)

        test_images = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        expected = ["night"] * 4 + ["day"] * 4
        actual = [nnc.predict(x=image, delta=0.1) for image in test_images]
        if plot:
            import numpy as np
            import matplotlib.pyplot as plt
            np.random.seed(331)
            x_night = np.array([x[0] for x in night_images])
            x_day = np.array([x[0] for x in day_images])
            x_test = np.array(test_images)
            plt.scatter(x=x_night, y=np.random.rand(
                len(x_night)), label="night")
            plt.scatter(x=x_day, y=np.random.rand(len(x_day)), label="day")
            plt.scatter(x=x_test, y=np.zeros(len(x_test)), c="k", label="test")
            plt.xlabel("Value")
            plt.yticks([], [])
            plt.legend()
            plt.show()

        self.assertEqual(expected, actual)

        """
        (2) Season Classification: Suppose temperature is measured between 0 and 1, and we are
        provided labeled examples of the season in which each temperature was recorded.
        Given a new temperature, predict the season we are experiencing.
        """
        random.seed(331)
        winter_temps = [(random.random() / 4, "winter") for _ in range(50)]
        spring_temps = [(random.random() / 4 + 0.25, "spring")
                        for _ in range(50)]
        summer_temps = [(random.random() / 4 + 0.75, "summer")
                        for _ in range(50)]
        fall_temps = [(random.random() / 4 + 0.5, "fall") for _ in range(50)]
        data = winter_temps + spring_temps + summer_temps + fall_temps

        nnc = NearestNeighborClassifier(resolution=1)
        nnc.fit(data)

        test_temps = [i / 20 for i in range(20)]
        expected = ["winter"] * 6 + ["spring"] * 5 + ["fall"] * 4 + ["summer"] * 5
        actual = [nnc.predict(x=temp, delta=0) for temp in test_temps]
        if plot:
            np.random.seed(331)
            x_winter, x_spring, x_summer, x_fall = \
                np.array([x[0] for x in winter_temps]), np.array([x[0] for x in spring_temps]), \
                np.array([x[0] for x in summer_temps]), np.array(
                    [x[0] for x in fall_temps])
            x_test = np.array(test_temps)
            plt.scatter(x=x_winter, y=np.random.rand(
                len(x_winter)), label="winter")
            plt.scatter(x=x_spring, y=np.random.rand(
                len(x_spring)), label="spring")
            plt.scatter(x=x_summer, y=np.random.rand(
                len(x_summer)), label="summer")
            plt.scatter(x=x_fall, y=np.random.rand(len(x_fall)), label="fall")
            plt.scatter(x=x_test, y=np.zeros(len(x_test)), c="k", label="test")
            plt.xlabel("Value")
            plt.yticks([], [])
            plt.legend()
            plt.show()

        self.assertEqual(expected, actual)

        """
        (3) Rainfall Classification: Suppose daily rainfall is measured between 0 and 1
        relative to some baseline, and we are provided labeled examples of whether each year
        experienced drought, normal, or flood conditions. Given a new rainfall measurement, predict
        whether this year will experience drought, normal, or flood conditions.
        """
        random.seed(331)
        drought_rains = [(random.random() / 2, "drought") for _ in range(1000)]
        normal_rains = [(random.random() / 5 + 0.4, "normal")
                        for _ in range(1000)]
        flood_rains = [(random.random() / 2 + 0.5, "flood")
                       for _ in range(1000)]
        data = drought_rains + normal_rains + flood_rains
        # equals = data == eval(open('data.txt').read())

        nnc = NearestNeighborClassifier(resolution=2)
        nnc.fit(data)

        test_rains = [i / 100 for i in range(100)]
        expected = ["drought"] * 40 + ["normal"] * 21 + ["flood"] * 39
        actual = [nnc.predict(x=rain, delta=0.01) for rain in test_rains]
        if plot:
            np.random.seed(331)
            x_drought, x_normal, x_flood = np.array([x[0] for x in drought_rains]), \
                                           np.array([x[0] for x in normal_rains]), np.array(
                [x[0] for x in flood_rains])
            x_test = np.array(test_rains)
            plt.scatter(x=x_normal, y=np.random.rand(
                len(x_normal)), label="normal")
            plt.scatter(x=x_drought, y=np.random.rand(
                len(x_drought)), label="drought")
            plt.scatter(x=x_flood, y=np.random.rand(
                len(x_flood)), label="flood")
            plt.scatter(x=x_test, y=np.zeros(len(x_test)), c="k", label="test")
            plt.xlabel("Value")
            plt.yticks([], [])
            plt.legend()
            plt.show()

        self.assertEqual(expected, actual)

    def test_nnc_comprehensive(self):

        plot = False
        """
        (4) Iris Species Classification: Given measurements of sepal length, sepal width,
        petal length and petal width, predict the species of iris flower.
        Data from the UCI ML repository via sklearn.datasets, with credit to R.A. Fisher.
        https://archive.ics.uci.edu/ml/datasets/iris
        """
        iris_labels = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                       'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                       'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                       'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                       'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa',
                       'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica',
                       'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica']
        sepal_length = [0.2222, 0.1667, 0.1111, 0.0833, 0.1944, 0.3056, 0.0833, 0.1944, 0.0278, 0.1667, 0.3056, 0.1389,
                        0.1389, 0.0, 0.4167, 0.3889, 0.3056, 0.2222, 0.3889, 0.2222, 0.3056, 0.2222, 0.0833, 0.2222,
                        0.1389, 0.1944, 0.1944, 0.25, 0.25, 0.1111, 0.1389, 0.3056, 0.25, 0.3333, 0.1667, 0.1944,
                        0.3333, 0.1667, 0.0278, 0.2222, 0.1944, 0.0556, 0.0278, 0.1944, 0.2222, 0.1389, 0.2222, 0.0833,
                        0.2778, 0.1944, 0.75, 0.5833, 0.7222, 0.3333, 0.6111, 0.3889, 0.5556, 0.1667, 0.6389, 0.25,
                        0.1944, 0.4444, 0.4722, 0.5, 0.3611, 0.6667, 0.3611, 0.4167, 0.5278, 0.3611, 0.4444, 0.5,
                        0.5556, 0.5, 0.5833,
                        0.6389, 0.6944, 0.6667, 0.4722, 0.3889, 0.3333, 0.3333, 0.4167, 0.4722, 0.3056, 0.4722, 0.6667,
                        0.5556, 0.3611, 0.3333, 0.3333, 0.5, 0.4167, 0.1944, 0.3611, 0.3889, 0.3889, 0.5278, 0.2222,
                        0.3889, 0.5556, 0.4167, 0.7778, 0.5556, 0.6111, 0.9167, 0.1667, 0.8333, 0.6667, 0.8056, 0.6111,
                        0.5833, 0.6944, 0.3889, 0.4167, 0.5833, 0.6111, 0.9444, 0.9444, 0.4722, 0.7222, 0.3611, 0.9444,
                        0.5556, 0.6667, 0.8056, 0.5278, 0.5, 0.5833, 0.8056, 0.8611, 1.0, 0.5833, 0.5556, 0.5, 0.9444,
                        0.5556, 0.5833, 0.4722, 0.7222, 0.6667, 0.7222, 0.4167, 0.6944, 0.6667, 0.6667, 0.5556, 0.6111,
                        0.5278, 0.4444]
        sepal_width = [0.625, 0.4167, 0.5, 0.4583, 0.6667, 0.7917, 0.5833, 0.5833, 0.375, 0.4583, 0.7083, 0.5833,
                       0.4167, 0.4167, 0.8333, 1.0, 0.7917, 0.625, 0.75, 0.75, 0.5833, 0.7083, 0.6667, 0.5417, 0.5833,
                       0.4167, 0.5833, 0.625, 0.5833, 0.5, 0.4583, 0.5833, 0.875, 0.9167, 0.4583, 0.5, 0.625, 0.6667,
                       0.4167, 0.5833, 0.625, 0.125, 0.5, 0.625, 0.75, 0.4167, 0.75, 0.5, 0.7083, 0.5417, 0.5, 0.5,
                       0.4583, 0.125, 0.3333, 0.3333, 0.5417, 0.1667, 0.375, 0.2917, 0.0, 0.4167, 0.0833, 0.375, 0.375,
                       0.4583, 0.4167, 0.2917, 0.0833, 0.2083, 0.5, 0.3333, 0.2083, 0.3333, 0.375,
                       0.4167, 0.3333, 0.4167, 0.375, 0.25, 0.1667, 0.1667, 0.2917, 0.2917, 0.4167, 0.5833, 0.4583,
                       0.125, 0.4167, 0.2083, 0.25, 0.4167, 0.25, 0.125, 0.2917, 0.4167, 0.375, 0.375, 0.2083, 0.3333,
                       0.5417, 0.2917, 0.4167, 0.375, 0.4167, 0.4167, 0.2083, 0.375, 0.2083, 0.6667, 0.5, 0.2917,
                       0.4167, 0.2083, 0.3333, 0.5, 0.4167, 0.75, 0.25, 0.0833, 0.5, 0.3333, 0.3333, 0.2917, 0.5417,
                       0.5, 0.3333, 0.4167, 0.3333, 0.4167, 0.3333, 0.75, 0.3333, 0.3333, 0.25, 0.4167, 0.5833, 0.4583,
                       0.4167, 0.4583, 0.4583, 0.4583, 0.2917, 0.5, 0.5417, 0.4167, 0.2083, 0.4167, 0.5833, 0.4167]
        petal_length = [0.0678, 0.0678, 0.0508, 0.0847, 0.0678, 0.1186, 0.0678, 0.0847, 0.0678, 0.0847, 0.0847, 0.1017,
                        0.0678, 0.0169, 0.0339, 0.0847, 0.0508, 0.0678, 0.1186, 0.0847, 0.1186, 0.0847, 0.0, 0.1186,
                        0.1525, 0.1017, 0.1017, 0.0847, 0.0678, 0.1017, 0.1017, 0.0847, 0.0847, 0.0678, 0.0847, 0.0339,
                        0.0508, 0.0678, 0.0508, 0.0847, 0.0508, 0.0508, 0.0508, 0.1017, 0.1525, 0.0678, 0.1017, 0.0678,
                        0.0847, 0.0678, 0.6271, 0.5932, 0.661, 0.5085, 0.6102, 0.5932, 0.6271, 0.3898, 0.6102, 0.4915,
                        0.4237, 0.5424, 0.5085, 0.6271, 0.4407, 0.5763, 0.5932, 0.5254, 0.5932, 0.4915, 0.6441, 0.5085,
                        0.661, 0.6271,
                        0.5593, 0.5763, 0.6441, 0.678, 0.5932, 0.4237, 0.4746, 0.4576, 0.4915, 0.6949, 0.5932, 0.5932,
                        0.6271, 0.5763, 0.5254, 0.5085, 0.5763, 0.6102, 0.5085, 0.3898, 0.5424, 0.5424, 0.5424, 0.5593,
                        0.339, 0.5254, 0.8475, 0.6949, 0.8305, 0.7797, 0.8136, 0.9492, 0.5932, 0.8983, 0.8136, 0.8644,
                        0.6949, 0.7288, 0.7627, 0.678, 0.6949, 0.7288, 0.7627, 0.9661, 1.0, 0.678, 0.7966, 0.661,
                        0.9661, 0.661, 0.7966, 0.8475, 0.6441, 0.661, 0.7797, 0.8136, 0.8644, 0.9153, 0.7797, 0.6949,
                        0.7797, 0.8644, 0.7797, 0.7627, 0.6441, 0.7458, 0.7797, 0.6949, 0.6949, 0.8305, 0.7966, 0.7119,
                        0.678, 0.7119, 0.7458, 0.6949]
        petal_width = [0.0417, 0.0417, 0.0417, 0.0417, 0.0417, 0.125, 0.0833, 0.0417, 0.0417, 0.0, 0.0417, 0.0417, 0.0,
                       0.0, 0.0417, 0.125, 0.125, 0.0833, 0.0833, 0.0833, 0.0417, 0.125, 0.0417, 0.1667, 0.0417, 0.0417,
                       0.125, 0.0417, 0.0417, 0.0417, 0.0417, 0.125, 0.0, 0.0417, 0.0417, 0.0417, 0.0417, 0.0, 0.0417,
                       0.0417, 0.0833, 0.0833, 0.0417, 0.2083, 0.125, 0.0833, 0.0417, 0.0417, 0.0417, 0.0417, 0.5417,
                       0.5833, 0.5833, 0.5, 0.5833, 0.5, 0.625, 0.375, 0.5, 0.5417, 0.375, 0.5833, 0.375, 0.5417, 0.5,
                       0.5417, 0.5833, 0.375, 0.5833, 0.4167, 0.7083, 0.5, 0.5833, 0.4583,
                       0.5, 0.5417, 0.5417, 0.6667, 0.5833, 0.375, 0.4167, 0.375, 0.4583, 0.625, 0.5833, 0.625, 0.5833,
                       0.5, 0.5, 0.5, 0.4583, 0.5417, 0.4583, 0.375, 0.5, 0.4583, 0.5, 0.5, 0.4167, 0.5, 1.0, 0.75,
                       0.8333, 0.7083, 0.875, 0.8333, 0.6667, 0.7083, 0.7083, 1.0, 0.7917, 0.75, 0.8333, 0.7917, 0.9583,
                       0.9167, 0.7083, 0.875, 0.9167, 0.5833, 0.9167, 0.7917, 0.7917, 0.7083, 0.8333, 0.7083, 0.7083,
                       0.7083, 0.8333, 0.625, 0.75, 0.7917, 0.875, 0.5833, 0.5417, 0.9167, 0.9583, 0.7083, 0.7083,
                       0.8333, 0.9583, 0.9167, 0.75, 0.9167, 1.0, 0.9167, 0.75, 0.7917, 0.9167, 0.7083]
        test_points = [i / 10 for i in range(11)]

        # exploratory visualization for the curious coder
        if plot:
            import numpy as np
            import matplotlib.pyplot as plt
            for name, feature in [("sepal length", sepal_length), ("sepal width", sepal_width),
                                  ("petal length", petal_length), ("petal width", petal_width)]:
                np.random.seed(331)
                x = np.array(feature)
                x_setosa, x_versicolour, x_virginica = x[:50], x[50:100], x[100:]
                x_test = np.array(test_points)
                plt.scatter(x=x_setosa, y=np.random.rand(
                    len(x_setosa)), label="setosa")
                plt.scatter(x=x_versicolour, y=np.random.rand(
                    len(x_versicolour)), label="versicolour")
                plt.scatter(x=x_virginica, y=np.random.rand(
                    len(x_virginica)), label="virginica")
                plt.scatter(x=x_test, y=np.zeros(
                    len(x_test)), c="k", label="test")
                plt.title(name)
                plt.xlabel("Value")
                plt.yticks([], [])
                plt.legend()
                plt.show()

        # 4a: sepal length
        data = zip(sepal_length, iris_labels)
        nnc = NearestNeighborClassifier(resolution=2)
        nnc.fit(data)
        expected = ['setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'versicolor',
                    'virginica', 'virginica', 'virginica', 'virginica', 'virginica']
        actual = [nnc.predict(x=x, delta=0.05) for x in test_points]
        self.assertEqual(expected, actual)

        # 4b: sepal width
        data = zip(sepal_width, iris_labels)
        nnc = NearestNeighborClassifier(resolution=2)
        nnc.fit(data)
        expected = ['versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor',
                    'virginica', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        actual = [nnc.predict(x=x, delta=0.05) for x in test_points]
        self.assertEqual(expected, actual)

        # 4c: petal length
        data = zip(petal_length, iris_labels)
        nnc = NearestNeighborClassifier(resolution=2)
        nnc.fit(data)
        expected = ['setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor',
                    'versicolor', 'virginica', 'virginica', 'virginica', 'virginica']
        actual = [nnc.predict(x=x, delta=0.05) for x in test_points]
        self.assertEqual(expected, actual)

        # 4d: petal width
        data = zip(petal_width, iris_labels)
        nnc = NearestNeighborClassifier(resolution=2)
        nnc.fit(data)
        expected = ['setosa', 'setosa', 'setosa', None, 'versicolor', 'versicolor',
                    'versicolor', 'virginica', 'virginica', 'virginica', 'virginica']
        actual = [nnc.predict(x=x, delta=0.05) for x in test_points]
        self.assertEqual(expected, actual)

    def test_feedback_xml_validity(self):

        path = "feedback.xml"
        xml_doc = minidom.parse(path)
        response = {}
        tags = ["netid", "feedback", "difficulty", "time", "citations", "type", "number"]

        # Assert that we can access all tags
        for tag in tags:
            raw = xml_doc.getElementsByTagName(tag)[0].firstChild.nodeValue
            lines = [s.strip() for s in raw.split("\n\r")]  # If multiple lines, strip each line
            clean = " ".join(lines).strip()  # Rejoin lines with spaces and strip leading space
            self.assertNotEqual("REPLACE", clean)  # Make sure entry was edited
            response[tag] = clean  # Save each entry

        # Assert that difficulty is a float between 0-10
        difficulty_float = float(response["difficulty"])
        self.assertGreaterEqual(difficulty_float, 0.0)
        self.assertLessEqual(difficulty_float, 10.0)

        # Assert that hours is a float between 0-100 (hopefully it didn't take 100 hours!)
        time_float = float(response["time"])
        self.assertGreaterEqual(time_float, 0.0)
        self.assertLessEqual(time_float, 100.0)

        # Assert assignment type and number was not changed
        self.assertEqual("Project", response["type"])
        self.assertEqual("7", response["number"])


if __name__ == '__main__':
    unittest.main()
