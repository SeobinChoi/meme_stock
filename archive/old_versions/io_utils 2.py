"""
Atomic file I/O and versioning utilities.

Provides safe file operations that prevent data corruption
during writes and maintains versioning for datasets.
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from .paths import versioned_filename
from .time_utils import generate_version_timestamp, parse_version_from_filename

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Atomically write DataFrame to CSV file.
    
    Writes to a temporary file first, then moves to final location
    to prevent partial writes if interrupted.
    
    Args:
        df: DataFrame to write
        path: Target file path
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory as target
    temp_dir = path.parent
    with tempfile.NamedTemporaryFile(
        mode='w', 
        dir=temp_dir, 
        suffix='.tmp',
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        temp_path = Path(tmp_file.name)
        
        # Write DataFrame to temporary file
        df.to_csv(tmp_file, index=False, lineterminator='\n')
    
    # Atomically move temporary file to final location
    os.replace(temp_path, path)

def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Atomically write dictionary to JSON file.
    
    Args:
        obj: Dictionary to write as JSON
        path: Target file path
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory as target
    temp_dir = path.parent
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=temp_dir,
        suffix='.tmp', 
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        temp_path = Path(tmp_file.name)
        
        # Write JSON to temporary file
        json.dump(obj, tmp_file, indent=2, ensure_ascii=False)
    
    # Atomically move temporary file to final location
    os.replace(temp_path, path)

def existing_versions(base_stem: Path) -> List[Path]:
    """
    Find all existing versions of a file.
    
    Args:
        base_stem: Base file path (without version)
        
    Returns:
        List of paths for all versions, sorted by version timestamp
        
    Examples:
        >>> existing_versions(Path("data/stocks/GME_stock_data.csv"))
        [Path("data/stocks/GME_stock_data.csv"), 
         Path("data/stocks/GME_stock_data_v20250812153000.csv")]
    """
    versions = []
    parent = base_stem.parent
    stem = base_stem.stem
    suffix = base_stem.suffix
    
    if not parent.exists():
        return versions
    
    # Find base file (no version)
    if base_stem.exists():
        versions.append(base_stem)
    
    # Find versioned files
    pattern = f"{stem}_v*.{suffix.lstrip('.')}" if suffix else f"{stem}_v*"
    
    for file_path in parent.glob(pattern):
        if file_path != base_stem:
            versions.append(file_path)
    
    # Sort by version timestamp
    def version_key(path):
        version = parse_version_from_filename(path.name)
        return version if version else "0" * 14
    
    versions.sort(key=version_key)
    return versions

def latest_version_path(base: Path) -> Optional[Path]:
    """
    Get path to the latest version of a file.
    
    Args:
        base: Base file path
        
    Returns:
        Path to latest version or None if no versions exist
    """
    versions = existing_versions(base)
    return versions[-1] if versions else None

def compute_sha256(path: Path) -> str:
    """
    Compute SHA256 checksum of file.
    
    Args:
        path: File path to hash
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()

def safe_write_versioned(df: pd.DataFrame, base_path: Path, metadata: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
    """
    Safely write DataFrame with versioning policy.
    
    Args:
        df: DataFrame to write
        base_path: Base file path (without version)
        metadata: Metadata dictionary to update
        
    Returns:
        Tuple of (actual_file_path, updated_metadata)
    """
    # Generate version timestamp
    version = generate_version_timestamp()
    
    # Determine actual file path based on versioning policy
    if base_path.exists():
        # Base file exists, create versioned file
        actual_path = versioned_filename(base_path, version)
        metadata["version"] = version
    else:
        # Base file doesn't exist, use base path
        actual_path = base_path
        metadata["version"] = version
    
    # Write file atomically
    atomic_write_csv(df, actual_path)
    
    # Compute checksum
    metadata["checksum_sha256"] = compute_sha256(actual_path)
    
    return actual_path, metadata

def write_metadata_sidecar(csv_path: Path, metadata: Dict[str, Any]) -> Path:
    """
    Write metadata as sidecar JSON file.
    
    Args:
        csv_path: Path to CSV file
        metadata: Metadata dictionary
        
    Returns:
        Path to metadata file
    """
    # Generate metadata file path
    if "_v" in csv_path.stem:
        # Versioned file - include version in metadata filename
        meta_path = csv_path.with_suffix('.meta.json')
    else:
        # Base file - use base metadata filename
        meta_path = csv_path.with_suffix('.meta.json')
    
    # Write metadata atomically
    atomic_write_json(metadata, meta_path)
    
    return meta_path

def append_to_index(metadata: Dict[str, Any], csv_path: Path) -> None:
    """
    Append dataset entry to INDEX.jsonl file.
    
    Args:
        metadata: Metadata dictionary
        csv_path: Path to CSV file
    """
    from .paths import get_data_index_path
    
    index_path = get_data_index_path()
    
    # Create index entry
    index_entry = {
        "path": str(csv_path),
        "symbol": metadata.get("symbol", ""),
        "asset_type": metadata.get("asset_type", ""),
        "source": metadata.get("source", ""),
        "version": metadata.get("version", ""),
        "total_records": metadata.get("total_records", 0),
        "missing_dates": metadata.get("missing_dates", 0),
        "collection_timestamp": metadata.get("collection_timestamp", ""),
        "checksum_sha256": metadata.get("checksum_sha256", "")
    }
    
    # Ensure parent directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to index file
    with open(index_path, 'a', encoding='utf-8') as f:
        json.dump(index_entry, f, ensure_ascii=False)
        f.write('\n')

def read_index() -> List[Dict[str, Any]]:
    """
    Read all entries from INDEX.jsonl file.
    
    Returns:
        List of index entries
    """
    from .paths import get_data_index_path
    
    index_path = get_data_index_path()
    
    if not index_path.exists():
        return []
    
    entries = []
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    
    return entries

def cleanup_temp_files(directory: Path) -> int:
    """
    Clean up temporary files in directory.
    
    Args:
        directory: Directory to clean
        
    Returns:
        Number of files cleaned up
    """
    if not directory.exists():
        return 0
    
    count = 0
    for temp_file in directory.glob("*.tmp"):
        try:
            temp_file.unlink()
            count += 1
        except OSError:
            continue
    
    return count

def get_file_size(path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return path.stat().st_size
    except (OSError, FileNotFoundError):
        return 0

def create_symlink_if_supported(target: Path, link: Path) -> bool:
    """
    Create symlink to latest version if OS supports it.
    
    Args:
        target: Target file path
        link: Symlink path
        
    Returns:
        True if symlink was created, False otherwise
    """
    try:
        # Remove existing symlink if it exists
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            return False  # Don't overwrite regular files
        
        # Create new symlink
        link.symlink_to(target)
        return True
        
    except (OSError, NotImplementedError):
        # Symlinks not supported on this platform
        return False