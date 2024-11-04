import os
from typing import Union, TypeVar, Generic
from src.train.storage.file_manager import JSONFileManager

StorageValue = TypeVar("StorageValue", bound=Union[str, int, float, dict, list, None])


class KeyValueStorage(Generic[StorageValue]):
    """
    Storage class is a base class for all storage units.

    Generic:
        StorageValue: The type of the storage value.

    Attributes:
        root_path: str
            The root path of the storage unit.
    """

    def __init__(self, root_filename: str, label: str = "Storage", log: bool = False):
        is_valid_json = root_filename.endswith(".json")
        if not is_valid_json:
            raise ValueError(
                f"Invalid filename, should be `*.json`.\nCheck given filename {root_filename}"
            )

        root = os.path.join(os.getcwd(), root_filename)
        self.root_path = root
        self.label = label
        self.log = log
        self._io = JSONFileManager(root_path=self.root_path, label=label)
        self._io.load_json()
        self._db: dict[str, StorageValue] = self._io.load_json()

    def check_field_exist(self, field: str) -> bool:
        """
        Check if the field exists in the storage unit.

        Args:
            field: str
                The field to check.

        Returns:
            bool
                True if the field exists, False otherwise.
        """
        return field in self.keys

    def save(self) -> None:
        """
        Save the storage unit to the file.
        """
        self._io.save_json(self._db)

    def delete_field(self, field: str) -> None:
        """
        Delete a field from the storage unit.

        Args:
            field: str
                The field to delete.
        """
        if not self.check_field_exist(field):
            if self.log:
                print(f"[{self.label}]: Delete failed, field {field} does not exist.")
            return

        del self._db[field]

    def reset(self) -> None:
        """
        Reset the storage unit.
        """
        self._db = {}
        self._io.save_json(self._db)

    def insert_field(self, field: str, value: StorageValue) -> None:
        """
        Update(replace) a field in the storage unit.

        Args:
            field: str
                The field to update.
            value: T
                The value to update.
        """
        self._db[field] = value

    def inquire(self, field: str) -> Union[StorageValue, None]:
        """
        Inquire the storage unit.
        """
        if not self.check_field_exist(field):
            if self.log:
                print(f"[{self.label}]: Inquire failed, field {field} does not exist.")
            return None
        return self._db[field]

    @property
    def keys(self) -> list[str]:
        """
        Get the keys of the storage unit.

        Returns:
            list[str]
                The list of keys.
        """
        return list(self._db.keys())

    @property
    def values(self) -> list[StorageValue]:
        """
        Get the values of the storage unit.

        Returns:
            list[T]
                The list of values.
        """
        return list(self._db.values())

    @property
    def size(self) -> int:
        """
        Get the size of the storage unit.

        Returns:
            int
                The size of the storage unit.
        """
        return len(self._db)

    def __repr__(self) -> str:
        return f"[Storage Unit: {self.label}] (root_path={self.root_path})"
