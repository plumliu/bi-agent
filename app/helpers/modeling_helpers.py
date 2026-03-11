"""
Helper functions for modeling_custom subgraph.
These functions are uploaded to sandbox and used by executor to persist files and maintain registry.
"""
import json
import os
import pandas as pd
from pathlib import Path

REGISTRY_PATH = "/home/user/registered_files.json"


def _load_registry():
    """Load or create registry file"""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"main_tables": {}, "artifacts": {}}


def _save_registry(registry):
    """Save registry to file"""
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def create_main_table(df, file_name: str, description: str, columns_desc: dict) -> str:
    """
    Create a new main table and register it.

    Args:
        df: DataFrame to save
        file_name: Must end with .feather
        description: One-line description of the table
        columns_desc: Dict mapping column names to descriptions, must cover all columns

    Returns:
        Success message
    """
    # Validate file_name
    if not file_name.endswith('.feather'):
        raise ValueError(f"file_name must end with .feather, got: {file_name}")

    file_path = f"/home/user/{file_name}"

    # Check if file already exists
    if os.path.exists(file_path):
        raise ValueError(f"File {file_name} already exists. Use append_columns_to_main_table or append_rows_to_main_table to update.")

    # Validate columns_desc covers all columns
    df_columns = set(df.columns)
    desc_columns = set(columns_desc.keys())

    if df_columns != desc_columns:
        missing = df_columns - desc_columns
        extra = desc_columns - df_columns
        msg = []
        if missing:
            msg.append(f"Missing descriptions for columns: {missing}")
        if extra:
            msg.append(f"Extra descriptions for non-existent columns: {extra}")
        raise ValueError("; ".join(msg))

    # Save DataFrame
    df.to_feather(file_path)

    # Register
    registry = _load_registry()
    registry["main_tables"][file_name] = {
        "description": description,
        "columns_desc": columns_desc,
        "type": "feather"
    }
    _save_registry(registry)

    return f"[Helper create_main_table Used ✓] Created main table: {file_name} ({len(df)} rows, {len(df.columns)} columns)"


def append_columns_to_main_table(
    file_name: str,
    df,
    join_on: str | list[str],
    columns_desc: dict,
    description: str | None = None
) -> str:
    """
    Append new columns to existing main table.

    Args:
        file_name: Target main table file name (must exist)
        df: DataFrame containing new columns (can have redundant columns)
        join_on: Key(s) to join on
        columns_desc: Dict describing ONLY new columns
        description: Optional new description for the table

    Returns:
        Success message
    """
    file_path = f"/home/user/{file_name}"

    if not os.path.exists(file_path):
        raise ValueError(f"Main table {file_name} does not exist. Use create_main_table first.")

    # Load existing table
    existing_df = pd.read_feather(file_path)

    # Check for column conflicts
    new_cols = set(columns_desc.keys())
    existing_cols = set(existing_df.columns)
    conflicts = new_cols & existing_cols
    if conflicts:
        raise ValueError(f"Columns already exist in main table: {conflicts}")

    # Prepare join columns
    join_cols = [join_on] if isinstance(join_on, str) else join_on

    # Extract only needed columns from df
    needed_cols = join_cols + list(new_cols)
    df_subset = df[needed_cols]

    # Merge
    merged_df = existing_df.merge(df_subset, on=join_on, how='left')

    # Save
    merged_df.to_feather(file_path)

    # Update registry
    registry = _load_registry()
    registry["main_tables"][file_name]["columns_desc"].update(columns_desc)
    if description:
        registry["main_tables"][file_name]["description"] = description
    _save_registry(registry)

    return f"[Helper append_columns_to_main_table Used ✓] Appended {len(new_cols)} columns to {file_name}"


def append_rows_to_main_table(
    file_name: str,
    df,
    description: str | None = None
) -> str:
    """
    Append new rows to existing main table.

    Args:
        file_name: Target main table file name (must exist)
        df: DataFrame with new rows (must have same columns as existing table)
        description: Optional new description for the table

    Returns:
        Success message
    """
    file_path = f"/home/user/{file_name}"

    if not os.path.exists(file_path):
        raise ValueError(f"Main table {file_name} does not exist. Use create_main_table first.")

    # Load existing table
    existing_df = pd.read_feather(file_path)

    # Validate columns match exactly
    existing_cols = set(existing_df.columns)
    new_cols = set(df.columns)

    if existing_cols != new_cols:
        missing = existing_cols - new_cols
        extra = new_cols - existing_cols
        msg = []
        if missing:
            msg.append(f"Missing columns: {missing}")
        if extra:
            msg.append(f"Extra columns: {extra}")
        raise ValueError("; ".join(msg))

    # Concatenate
    merged_df = pd.concat([existing_df, df], ignore_index=True)

    # Save
    merged_df.to_feather(file_path)

    # Update registry
    if description:
        registry = _load_registry()
        registry["main_tables"][file_name]["description"] = description
        _save_registry(registry)

    return f"[Helper append_rows_to_main_table Used ✓] Appended {len(df)} rows to {file_name}"


def create_artifact(
    data: dict,
    file_name: str,
    description: str,
    columns_desc: dict
) -> str:
    """
    Create a JSON artifact and register it.

    Args:
        data: Dict to save as JSON
        file_name: Must end with .json
        description: One-line description of the artifact
        columns_desc: Dict mapping top-level keys to descriptions

    Returns:
        Success message
    """
    # Validate file_name
    if not file_name.endswith('.json'):
        raise ValueError(f"file_name must end with .json, got: {file_name}")

    file_path = f"/home/user/{file_name}"

    # Check if file already exists
    if os.path.exists(file_path):
        raise ValueError(f"File {file_name} already exists. JSON artifacts cannot be modified.")

    # Validate columns_desc covers all top-level keys
    data_keys = set(data.keys())
    desc_keys = set(columns_desc.keys())

    if data_keys != desc_keys:
        missing = data_keys - desc_keys
        extra = desc_keys - data_keys
        msg = []
        if missing:
            msg.append(f"Missing descriptions for keys: {missing}")
        if extra:
            msg.append(f"Extra descriptions for non-existent keys: {extra}")
        raise ValueError("; ".join(msg))

    # Save JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Register
    registry = _load_registry()
    registry["artifacts"][file_name] = {
        "description": description,
        "columns_desc": columns_desc,
        "type": "json"
    }
    _save_registry(registry)

    return f"[Helper create_artifact Used ✓] Created artifact: {file_name}"
