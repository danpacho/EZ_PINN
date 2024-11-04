from abc import ABC, abstractmethod
import json
import os


class FileManager(ABC):
    @abstractmethod
    def create(self, data):
        raise NotImplementedError

    @abstractmethod
    def load_json(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    @abstractmethod
    def delete(self):
        raise NotImplementedError


class JSONFileManager(FileManager):
    def __init__(self, root_path: str, label: str = "JSON_IO"):
        self.root_path = root_path
        self.label = label

    def update_root_path(self, root_path: str):
        """
        Update the root path.

        Args:
            root_path (str): New root path
        """
        self.root_path = root_path
        print(f"[{self.label}]: Root path updated to {self.root_path}")

    def create(self, data: dict):
        """
        Create new dictionary and save it as a JSON file.

        Args:
            data (dict): Data to save
        """
        if not os.path.exists(os.path.dirname(self.root_path)):
            os.makedirs(os.path.dirname(self.root_path))
            print(f"[{self.label}]: Directory created for {self.root_path}")

        self.save_json(data)

    def update(self, data: dict):
        """
        Update the existing JSON file with new data.

        Args:
            data (dict): Data to update
        """
        existing_data = self.load_json()
        existing_data.update(data)
        self.save_json(existing_data)

    def delete(self):
        """
        Delete the JSON file.
        """
        os.remove(self.root_path)
        print(f"[{self.label}]: {self.root_path} has been deleted.")

    def load_json(self) -> dict:
        """
        Load JSON data from the root path.

        Returns:
            dict: Loaded JSON data
        """
        try:
            with open(self.root_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
                print(f"[{self.label}]: JSON data loaded from {self.root_path}")
                return loaded
        except FileNotFoundError:
            print(f"[{self.label}]: {self.root_path} not found. Creating a new one.")
            self.create({})
            return {}

    def save_json(self, data: dict):
        """
        Save the given data as a JSON file.

        Args:
            data (dict): Data to save
        """
        with open(self.root_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print(f"[{self.label}]: JSON data saved to {self.root_path}")
