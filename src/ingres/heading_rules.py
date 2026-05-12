from dataclasses import dataclass
import re

# ---------------------------------------------------------------------------
# HeadingRule
# ---------------------------------------------------------------------------

@dataclass
class HeadingRule:
    """
    Maps a regex pattern to a heading level.
    Applied to lines that do NOT already start with '#'.

    pattern    : compiled regex, matched with re.match()
    level      : 1 or 2
    strip_match: if True, remove the matched prefix from the heading text
    """
    pattern:     re.Pattern
    level:       int
    strip_match: bool = False

    @classmethod
    def from_str(
        cls,
        pattern: str,
        level: int,
        strip_match: bool = False,
    ) -> "HeadingRule":
        return cls(
            pattern=re.compile(pattern),
            level=level,
            strip_match=strip_match,
        )