"""Markdown section walker.

Loads hierarchically linked markdown files where parent documents
contain [!child.md] links to child files. Recursively follows those
links and produces one Document per section across the whole tree.
"""

import os
import logging
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from src.ingres.heading_rules import HeadingRule
from src.ingres.markdown_parser import MyMarkdownNodeParser
from src.utils.tools import create_uuid_from_string

log = logging.getLogger(__name__)


class MarkDownSectionWalker(BaseReader):
    """
    Loads markdown documents organised as a hierarchy of linked files.

    Parent files contain [!child.md] links which are followed recursively.
    Each section across the whole file tree becomes a separate Document
    with h1/h2/h3 metadata and a header_path for retrieval.

    After collecting all sections, the text is parsed by MarkdownNodeParser
    to produce consistently structured nodes.
    """

    def __init__(
        self,
        heading_rules: Optional[List[HeadingRule]] = None,
    ) -> None:
        super().__init__()
        self._heading_rules: List[HeadingRule] = heading_rules or []

    # ── main entry point ──────────────────────────────────────────────

    def load_data(
        self,
        start_file: Path,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Load markdown data starting with start_file, following [!child.md]
        links recursively. Returns one Document per section node.

        Parameters
        ----------
        start_file : path to the root markdown file
        extra_info : additional metadata merged into every document
        """
        start_file = Path(start_file)

        if not start_file.exists():
            log.error("File %s not found", start_file)
            return []

        base_metadata: Dict = dict(extra_info or {})

        # ── step 1: collect sections from file tree ───────────────────
        file_content = self._read_file(start_file)
        sections     = self._process_file(
            filename=str(start_file),
            file=file_content,
        )

        # ── step 2: build source Documents from sections ─────────────
        source_docs: List[Document] = []
        for section in sections.values():
            text = section.get("text", "")
            if isinstance(text, list):
                text = "\n".join(text)
            if not text.strip():
                continue

            metadata = {
                **base_metadata,
                "child_path":          str(section["filename"]),
                "file_name":           os.path.basename(section["filename"]),
                "file_size":           float(section.get("file_size", 0)),
                "creation_date":       section.get("creation_date", ""),
                "last_modified_date":  section.get("last_modified_date", ""),
                "level":               section.get("level", "0"),
                "h1":                  section.get("h1", ""),
                "h2":                  section.get("h2", ""),
                "h3":                  section.get("h3", ""),
                "category":            "Markdown",
            }

            doc     = Document(text=text, metadata=metadata)
            doc.id_ = create_uuid_from_string(section["id"])
            source_docs.append(doc)

        # ── step 3: parse each source doc into structured nodes ───────
        parser = MyMarkdownNodeParser.from_defaults(
            heading_rules=self._heading_rules,
        )

        result: List[Document] = []
        for source_doc in source_docs:
            nodes = parser.get_nodes_from_node(source_doc)
            for node in nodes:
                result.append(
                    Document(
                        text=node.get_content(),
                        metadata=dict(node.metadata or {}),
                        id_=node.node_id,
                    )
                )

        return result

    # ── file reading ──────────────────────────────────────────────────

    def _read_file(self, filename: Path) -> str:
        with open(filename, encoding="utf-8") as f:
            return f.read()

    def _get_file_times(
        self, path: str
    ) -> tuple[datetime.datetime, datetime.datetime]:
        return (
            datetime.datetime.fromtimestamp(os.path.getmtime(path)),
            datetime.datetime.fromtimestamp(os.path.getctime(path)),
        )

    # ── recursive section collector ───────────────────────────────────

    def _process_file(
        self,
        filename: str,
        file: str,
        parent_id: str = "",
        parent_section: str = "",
        parent_subsection: str = "",
        parent_subsubsection: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse a single markdown file into a dict of sections.
        Recursively follows [!child.md] links.

        Returns
        -------
        Dict keyed by section id, values are section dicts with
        keys: text, filename, creation_date, last_modified_date,
              file_size, level, h1, h2, h3, id
        """
        dt_m, dt_c = self._get_file_times(filename)
        file_stats  = os.stat(filename)
        file_path   = os.path.dirname(filename)

        sections: Dict[str, Dict[str, Any]] = {}
        current_section: Dict[str, Any] = {
            "text":               [],
            "filename":           filename,
            "creation_date":      dt_c.strftime("%d-%m-%Y"),
            "last_modified_date": dt_m.strftime("%d-%m-%Y"),
            "file_size":          str(file_stats.st_size),
            "h1": parent_section,
            "h2": parent_subsection,
            "h3": parent_subsubsection,
            "level": "0",
        }

        # determine starting id
        if parent_id:
            current_section["id"] = parent_id + "_sub"
        elif parent_subsubsection:
            current_section["id"] = "h3_" + parent_subsubsection
            current_section["level"] = "3"
        elif parent_subsection:
            current_section["id"] = "h2_" + parent_subsection
            current_section["level"] = "2"
        elif parent_section:
            current_section["id"] = "h1_" + parent_section
            current_section["level"] = "1"
        else:
            current_section["id"] = "root_" + os.path.basename(filename)

        current_id = current_section["id"]
        sections[current_id] = current_section

        for line in file.split("\n"):
            header_level = len(line) - len(line.lstrip("#"))
            header_level = min(header_level, 3)  # cap at h3

            if header_level > 0:
                # flush into a new section
                lvl                    = f"h{header_level}"
                header_text            = line.lstrip("#").strip()
                new_section: Dict[str, Any] = {
                    "text":               [],
                    "filename":           filename,
                    "creation_date":      dt_c.strftime("%d-%m-%Y"),
                    "last_modified_date": dt_m.strftime("%d-%m-%Y"),
                    "file_size":          str(file_stats.st_size),
                    "h1":                 parent_section,
                    "h2":                 parent_subsection,
                    "h3":                 parent_subsubsection,
                    lvl:                  header_text,
                    "id":                 f"{lvl}_{header_text}",
                    "level":              str(header_level),
                }

                if header_level == 1:
                    parent_section    = header_text
                    new_section["h2"] = ""
                    new_section["h3"] = ""
                    parent_subsection    = ""
                    parent_subsubsection = ""
                elif header_level == 2:
                    parent_subsection    = header_text
                    new_section["h3"]    = ""
                    parent_subsubsection = ""
                elif header_level == 3:
                    parent_subsubsection = header_text

                current_id      = new_section["id"]
                current_section = new_section
                sections[current_id] = current_section

            elif line.startswith("[!"):
                # link to a child file — follow recursively
                fn = line.strip().strip("[]!").strip()
                if not os.path.isabs(fn):
                    fn = os.path.join(file_path, fn)
                child_path = os.path.abspath(fn)

                if os.path.exists(child_path):
                    child_content  = self._read_file(Path(child_path))
                    child_sections = self._process_file(
                        filename=child_path,
                        file=child_content,
                        parent_id=str(current_id),
                        parent_section=current_section.get("h1", ""),
                        parent_subsection=current_section.get("h2", ""),
                        parent_subsubsection=current_section.get("h3", ""),
                    )
                    sections.update(child_sections)
                else:
                    log.error("Linked file not found: %s", child_path)

            else:
                # body text — append to current section
                text = line.strip("-").strip()
                if text:
                    sections[current_id]["text"].append(text)

        return sections