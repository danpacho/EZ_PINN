import uuid
import hashlib
import base64
import json
import os
from typing import Optional


class UUIDEncoder:
    def __init__(self, storage_file: str = "uuid_store.json"):
        self.storage_file = storage_file
        self.mapping = self._load_storage()

    def _load_storage(self) -> dict:
        """Load the existing mappings from a JSON file."""
        if not os.path.exists(self.storage_file):
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)

        try:
            with open(self.storage_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file does not exist or is empty, return an empty dictionary
            return {}

    def _save_storage(self):
        """Save the current mappings to a JSON file."""
        with open(self.storage_file, "w") as file:
            json.dump(self.mapping, file)

    def _string_to_md5(self, s: str) -> str:
        """Encode the string to a fixed-length MD5 hex string."""
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _uuid_from_md5(self, md5_hex: str) -> uuid.UUID:
        """Generate a UUID from a 32-character MD5 hex string."""
        return uuid.UUID(md5_hex)

    def encode_to_uuid(self, seed_str: str) -> str:
        """
        Generate an obfuscated UUID-like string from the seed string
        and store it in the mappings.
        """
        # If already encoded, return the stored UUID
        if seed_str in self.mapping:
            return self.mapping[seed_str]

        # Convert the seed string to an MD5 hash, then to UUID
        md5_hex = self._string_to_md5(seed_str)
        uuid_obj = self._uuid_from_md5(md5_hex)

        # Convert UUID to base64 for more obfuscation
        encoded_uuid = (
            base64.urlsafe_b64encode(uuid_obj.bytes).decode("utf-8").rstrip("=")
        )

        # Store the mapping and save to the JSON file
        self.mapping[seed_str] = encoded_uuid
        self._save_storage()

        return encoded_uuid

    def decode_from_uuid(self, encoded_str: str) -> Optional[str]:
        """
        Recover the original seed string from the encoded UUID-like string.
        """
        # Reverse lookup: find the seed string for the given encoded UUID
        for seed, uuid_str in self.mapping.items():
            if uuid_str == encoded_str:
                return seed
        return None

    def reset_storage(self, should_reset: bool = False):
        """Reset the storage file to an empty dictionary."""
        print(
            "Warning: reset all stored mappings.\nPlease proceed with caution.\nIf you are sure, set 'should_reset=True'."
        )
        if should_reset:
            print("Resetting the uuid storage...")
            self.mapping = {}
            self._save_storage()
            print("Storage reset successfully.")


if __name__ == "__main__":
    encoder = UUIDEncoder()

    # Encoding and storing a UUID
    seed_str_list = ["test", "hello", "world"]
    for seed_str in seed_str_list:
        encoded_uuid = encoder.encode_to_uuid(seed_str)
        print(f"Encoded UUID for '{seed_str}': {encoded_uuid}")

    # Decoding the stored UUID
    for seed_str in seed_str_list:
        decoded_str = encoder.decode_from_uuid(encoder.mapping[seed_str])
        print(f"Decoded string for '{seed_str}': {decoded_str}")

    # Resetting the storage
    encoder.reset_storage(True)
