"""
File utilities for temporal knowledge extraction.
"""

import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import yaml
import hashlib
import re


def url_to_hash(url, max_length=64):
    """
    Generate a unique hash for a URL which doesn't contain special characters and is up to 64 chars.
    Alphanumeric only. Uses SHA256 truncated and base36 encoded for compactness.
    """
    if not isinstance(url, str):
        url = str(url)
    # Get SHA256 digest
    sha_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # Convert hex digest to integer, then to base36
    hash_int = int(sha_digest, 16)
    base36_hash = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = hash_int
    if n == 0:
        base36_hash = "0"
    else:
        while n > 0:
            n, r = divmod(n, 36)
            base36_hash = alphabet[r] + base36_hash

    # Remove any leading special characters (should not be any, for safety apply)
    clean_hash = re.sub(r"[^a-zA-Z0-9]", "", base36_hash)
    if len(clean_hash) > max_length:
        clean_hash = clean_hash[:max_length]
    return clean_hash


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing the configuration
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Save data to a YAML file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]):
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
    """
    ensure_directory(Path(file_path).parent)

    with jsonlines.open(file_path, mode="w") as writer:
        for item in data:
            writer.write(item)


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with jsonlines.open(file_path, mode="r") as reader:
        for item in reader:
            data.append(item)
    return data


def save_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    file_path: Union[str, Path],
    indent: int = 2,
):
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
    """
    ensure_directory(Path(file_path).parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(
    file_path: Union[str, Path],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    index: bool = False,
    **kwargs,
):
    """
    Save data to a CSV file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
        index: Whether to include index
        **kwargs: Additional arguments for pandas.to_csv
    """
    ensure_directory(Path(file_path).parent)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=index, **kwargs)


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame containing the data
    """
    return pd.read_csv(file_path)


def save_data(
    data: List[Dict[str, Any]], file_path: Union[str, Path], format: str = "jsonl"
):
    """
    Save data in the specified format.

    Args:
        data: Data to save
        file_path: Path to save the file
        format: Output format (jsonl, json, csv)
    """
    if format.lower() == "jsonl":
        save_jsonl(data, file_path)
    elif format.lower() == "json":
        save_json(data, file_path)
    elif format.lower() == "csv":
        save_csv(data, file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_data(
    file_path: Union[str, Path], format: str = None
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Load data from a file.

    Args:
        file_path: Path to the file
        format: File format (auto-detected if None)

    Returns:
        Loaded data
    """
    if format is None:
        format = Path(file_path).suffix.lower().lstrip(".")

    if format == "jsonl":
        return load_jsonl(file_path)
    elif format == "json":
        return load_json(file_path)
    elif format == "csv":
        return load_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to the file

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.json", "*.csv")

    Returns:
        List of file paths
    """
    directory = Path(directory)
    return list(directory.glob(pattern))
