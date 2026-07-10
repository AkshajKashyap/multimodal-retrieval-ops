"""Deterministic local demo data."""

from pathlib import Path

from .manifest import ManifestItem, write_manifest

DEMO_ITEMS = (
    ManifestItem("demo-001", "images/red_bicycle.jpg", "A red bicycle by a wall.", "train", "demo"),
    ManifestItem("demo-002", "images/blue_boat.jpg", "A blue boat on calm water.", "train", "demo"),
    ManifestItem("demo-003", "images/green_tree.jpg", "A green tree in a field.", "validation", "demo"),
    ManifestItem("demo-004", "images/yellow_bus.jpg", "A yellow bus on a road.", "test", "demo"),
)


def generate_demo_manifest(path: Path) -> list[ManifestItem]:
    """Write the same small manifest on every invocation."""
    items = list(DEMO_ITEMS)
    write_manifest(items, path)
    return items
