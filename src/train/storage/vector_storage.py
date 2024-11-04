from typing import Union
from src.ds.bst import BST
from src.train.storage.storage_unit import KeyValueStorage


class VectorSortedStorage(KeyValueStorage[list[float]]):
    """
    Sorted VectorStorage for storing list[float] vector kinda data.

    Note:
    - The data is stored in a sorted manner.
    - If you want to preserve the order, then use `Storage` instead.
    """

    def __init__(self, root_filename: str, label: str = "VectorStorage"):
        super().__init__(root_filename, label)
        self._db: dict[str, BST] = {}
        self.load()

    def load(self) -> None:
        """
        Load the storage unit from the file.
        """
        loaded: dict[str, list[float]] = self._io.load_json()
        for key, value in loaded.items():
            self._db[key] = BST()
            self._db[key].insert_list(value)

    @property
    def values(self) -> list[list[float]]:
        """
        Return all the values in the storage unit
        """
        return [value.tolist for value in self._db.values()]

    def save(self) -> None:
        """
        Save the storage unit to the file.
        """
        to_json = {key: value.tolist for key, value in self._db.items()}
        self._io.save_json(to_json)

    def insert_field(self, field: str, value: list[float]) -> None:
        """
        Update a field

        Args:
            field: str
                The field to update.
            value: T
                The value to update.
        """
        self._db[field] = BST()

        self._db[field].insert_list(value)

    def insert_list(self, field: str, value: list[float]) -> None:
        """
        Insert a new value at existing field

        Args:
            field: str
                The field to update.
            value: list[float]
                The value to update.
        """
        if not self.check_field_exist(field):
            self._db[field] = BST()

        self._db[field].insert_list(value)

    def insert_single(self, field: str, value: float) -> None:
        """
        Insert a new value at existing field

        Args:
            field: str
                The field to update.
            value: float
                The value to update.
        """
        if not self.check_field_exist(field):
            self._db[field] = BST()

        self._db[field].insert(value)

    def inquire(self, field: str) -> Union[list[float], None]:
        """
        Inquire field
        """
        if not self.check_field_exist(field):
            if self.log:
                print(f"[{self.label}]: Inquired failed, field {field} does not exist.")
            return None
        return self._db[field].tolist

    def delete_field(self, field: str) -> None:
        """
        Delete a field

        Args:
            field: str
                The field to delete.
        """
        if not self.check_field_exist(field):
            if self.log:
                print(f"[{self.label}]: Delete failed, field {field} does not exist.")
            return

        del self._db[field]

    def delete_list(self, field: str, value: list[float]) -> None:
        """
        Delete a list

        Args:
            field: str
                The field to delete.
            value: list[float]
                The list to delete.
        """
        if not self.check_field_exist(field):
            if self.log:
                print(
                    f"[{self.label}]: Delete list failed, field {field} does not exist."
                )
            return

        self._db[field].delete_list(value)

    def delete_single(self, field: str, value: float) -> bool:
        """
        Delete a single value

        Args:
            field: str
                The field to delete.
            value: float
                The value to delete.
        """
        if not self.check_field_exist(field):
            if self.log:
                print(f"[{self.label}]: Delete single failed, {field} does not exist.")
            return False

        deleted_node = self._db[field].delete(value)
        return deleted_node is not None
