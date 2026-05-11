import json
from pathlib import Path
from dataclasses import dataclass, field
import re
from typing import Optional

@dataclass
class HierarchyLevel:
    """Represents one level in a document parse hierarchy.

    Attributes:
        level: The numeric order of the hierarchy level.
        name: A short identifier for the hierarchy level.
        pattern_type: The kind of matching pattern used.
        pattern: The regular expression for identifying this level.
        example: Example text that should match this level.
        contains_content: True if this level may contain inline content.
        compiled: The compiled regex object for `pattern`.
    """

    level:            int
    name:             str
    pattern_type:     str
    pattern:          str
    example:          str
    contains_content: bool
    compiled:         object = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compile the regex pattern after initialization."""
        self.compiled = re.compile(self.pattern, re.MULTILINE)

@dataclass  
class ParseConfig:
    """Defines a configured parse family for structured documents."""

    family_id:          str
    family_description: str
    hierarchy:          list[HierarchyLevel]
    special_sections:   list[dict]
    noise_patterns:     list[str]
    sample_passages:    list[dict]
    validated:          bool = False
    created_from:       str  = ""   # filename of the source document

class ConfigStore:
    """JSON-backed storage for ParseConfig definitions."""

    def __init__(self, path: str = "./configs"):
        """Initialize the store and ensure the storage directory exists."""
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
    
    def save(self, config: ParseConfig):
        """Save a ParseConfig instance to disk as a JSON file."""
        p = self.path / f"{config.family_id}.json"
        p.write_text(json.dumps(config.__dict__, default=str, indent=2))
    
    def load(self, family_id: str) -> Optional[ParseConfig]:
        """Load a ParseConfig by family_id, or return None if missing."""
        p = self.path / f"{family_id}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        data["hierarchy"] = [HierarchyLevel(**h) for h in data["hierarchy"]]
        return ParseConfig(**data)
    
    def list_families(self) -> list[str]:
        """List all saved family IDs in the config store."""
        return [p.stem for p in self.path.glob("*.json")]