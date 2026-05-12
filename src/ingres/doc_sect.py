"""Docs parser.

Contains parsers for docx files.

"""

import os
import json
import logging
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from docx.document import Document as DocxDocument
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ingres.heading_rules import HeadingRule
from ingres.markdown_parser import MyMarkdownNodeParser
from ingres.util import clean_tables, md_from_doc

log = logging.getLogger(__name__)


class DocxSectionReader(BaseReader):
    """Word document parser.

    Converts .docx → Markdown via md_from_doc(), then splits into
    section and list_item nodes using MarkdownNodeParser.

    Heading rules are loaded from disk to handle documents where
    structural headings are not correctly tagged in the Word file.
    """

    def __init__(
        self,
        heading_rules_path: Optional[Path] = None,
        family_id: Optional[str] = None,
        rules_dir: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        heading_rules_path : direct path to a specific JSON rules file
        family_id          : document family id (e.g. "EU_REGULATION")
                             looks for <rules_dir>/<family_id>.json
        rules_dir          : directory to search for family rule files
                             defaults to ./structure_cache
        """
        super(BaseReader, self).__init__()
        self._heading_rules: List[HeadingRule] = self._load_rules(
            heading_rules_path=heading_rules_path,
            family_id=family_id,
            rules_dir=rules_dir or Path("./structure_cache"),
        )

    # ── rule loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_rules(
        heading_rules_path: Optional[Path],
        family_id: Optional[str],
        rules_dir: Path,
    ) -> List[HeadingRule]:
        """
        Load heading rules from disk.

        Resolution order:
          1. explicit heading_rules_path
          2. rules_dir / family_id.json
          3. no rules (empty list — existing behaviour unchanged)
        """
        path: Optional[Path] = None

        if heading_rules_path is not None:
            path = Path(heading_rules_path)
        elif family_id is not None:
            candidate = rules_dir / f"{family_id}.json"
            if candidate.exists():
                path = candidate
            else:
                log.warning(
                    "No rules file found for family '%s' at %s",
                    family_id, candidate,
                )

        if path is None:
            return []

        if not path.exists():
            log.warning("Heading rules file not found: %s", path)
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = [
                HeadingRule.from_str(
                    pattern=r["pattern"],
                    level=r["level"],
                    strip_match=r.get("strip_match", False),
                )
                for r in data.get("heading_rules", [])
            ]
            log.info("Loaded %d heading rule(s) from %s", len(rules), path)
            return rules
        except (json.JSONDecodeError, KeyError) as exc:
            log.error("Failed to load heading rules from %s: %s", path, exc)
            return []

    # ── main entry point ──────────────────────────────────────────────

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Parse a .docx file into a list of Document objects.

        Each Document corresponds to one section or list_item node
        as produced by MarkdownNodeParser. Structural metadata
        (h1, h2, header_path, node_type) is preserved on each Document.

        Parameters
        ----------
        file       : path to the .docx file
        extra_info : additional metadata merged into every document
        """
        if not os.path.exists(file):
            log.error("File %s not found", file)
            return []

        metadata: Dict = dict(extra_info or {})

        # ── step 1: docx → markdown ───────────────────────────────────
        md_path: Optional[Path] = None
        try:
            md_path  = md_from_doc(file=file)
            md_text  = clean_tables(md_path) or ""
        finally:
            if md_path is not None:
                try:
                    os.remove(md_path)
                except OSError:
                    pass

        # ── step 2: parse into nodes ──────────────────────────────────
        parser = MyMarkdownNodeParser.from_defaults(
            heading_rules=self._heading_rules,
        )

        source_doc = Document(text=md_text, metadata=metadata)
        nodes      = parser.get_nodes_from_node(source_doc)

        # ── step 3: convert TextNodes back to Documents ───────────────
        return [
            Document(
                text=node.get_content(),
                metadata=dict(node.metadata or {}),
                id_=node.node_id,
            )
            for node in nodes
        ]

    # ── utilities (kept for potential future use) ─────────────────────

    def get_file_times(
        self, path: str
    ) -> tuple[datetime.datetime, datetime.datetime]:
        """Return (last_modified, created) datetimes for a file."""
        return (
            datetime.datetime.fromtimestamp(os.path.getmtime(path)),
            datetime.datetime.fromtimestamp(os.path.getctime(path)),
        )