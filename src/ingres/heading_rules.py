from dataclasses import dataclass
import re

@dataclass
class HeadingRule:
    """
    A single pattern-to-heading-level rule.
    Applied to lines that do NOT already start with '#'.
    
    pattern   : compiled regex, matched with re.match() against the line
    level     : heading level to assign (1=# 2=## 3=###)
    strip_match: if True, remove the matched prefix from the heading text
    """
    pattern:     re.Pattern
    level:       int
    strip_match: bool = False

    @classmethod
    def from_str(cls, pattern: str, level: int, 
                 strip_match: bool = False) -> "HeadingRule":
        return cls(
            pattern=re.compile(pattern),
            level=level,
            strip_match=strip_match,
        )