from typing import Optional, List


class Node:
    """
    Node with count for duplicated keys
    """

    def __init__(self, key: float) -> None:
        self.key = key
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.count = 1

    def add_count(self):
        """
        Add the count of the node
        """
        self.count += 1

    def sub_count(self):
        """
        Subtract the count of the node
        """
        self.count -= 1


class BST:
    """
    Binary Search Tree (duplicates allowed)
    """

    def __init__(self) -> None:
        self.table: dict[float, int] = {}
        self.root: Optional[Node] = None

    def insert(self, key: float) -> None:
        """
        Insert a key into the BST
        """
        self.table[key] = self.table.get(key, 0) + 1
        self.root = self._insert(self.root, key)

    def insert_list(self, key_list: List[float]) -> None:
        """
        Insert a list of keys into the BST
        """
        for key in key_list:
            self.insert(key)

    def _insert(self, root: Optional[Node], key: float) -> Node:
        if root is None:
            return Node(key)

        if key < root.key:
            root.left = self._insert(root.left, key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        else:
            root.add_count()
        return root

    def search(self, key: float) -> Optional[Node]:
        """
        Search for a key in the BST
        """
        if self.table.get(key, 0) == 0:
            return None
        return self._search(self.root, key)

    def _search(self, root: Optional[Node], key: float) -> Optional[Node]:
        if root is None:
            return None
        if root.key == key:
            return root

        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)

    def delete(self, key: float) -> Optional[Node]:
        """
        Delete a key from the BST
        """
        self.root = self._delete(self.root, key)

        if self.table.get(key, 0) > 0:
            self.table[key] -= 1
            if self.table[key] == 0:
                self.table.pop(key)

        return self.root

    def delete_list(self, key_list: List[float]) -> None:
        """
        Delete a list of keys from the BST
        """
        for key in key_list:
            self.delete(key)

    def _delete(self, root: Optional[Node], key: float) -> Optional[Node]:
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.count > 1:  # If the key has duplicates
                root.sub_count()
                return root

            # If the key has no duplicates
            # Left child is None -> return right child
            if root.left is None:
                return root.right
            # Right child is None -> return left child
            elif root.right is None:
                return root.left

            # Should find successor from right subtree
            temp = self._find_successor(root.right)
            root.key = temp.key
            root.count = temp.count

            # Delete the original successor node
            root.right = self._delete(root.right, root.key)
            temp.count = 1  # Reset temp node count since it's being deleted

        return root

    def _find_successor(self, node: Node) -> Node:
        current = node
        while current.left is not None:
            current = current.left
        return current

    @property
    def tolist(self) -> List[float]:
        """
        Return the inorder traversal of the BST, list[float]
        """
        return self._inorder(self.root)

    @property
    def list_inquire(self) -> List[float]:
        """
        Return the BST list, but can't ensure the order
        """
        return list(self.table.keys())

    def _inorder(self, root: Optional[Node]) -> List[float]:
        if root is None:
            return []
        return (
            self._inorder(root.left)
            + [root.key] * root.count
            + self._inorder(root.right)
        )

    def reset(self) -> None:
        """
        Reset the BST
        """
        self.root = None
