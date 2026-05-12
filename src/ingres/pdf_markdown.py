"""Docs parser.

Contains parsers for docx, pdf files.

"""

import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt

from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from ingres.util import md_from_doc, clean_tables
import pymupdf as fitz
from ingres.heading_rules import HeadingRule
from ingres.markdown_parser import MyMarkdownNodeParser

from ingres.ai_markdown_structure import (
    MarkdownStructureProfileGenerator,
    MarkdownStructureRuleApplier,
    STATIC_GENERIC_PROFILE,
    STATIC_EU_REGULATION_PROFILE,
)
from models.model_factory import Modeltypes

logger = logging.getLogger(__name__)
coms = None
try:
    import comtypes.client as coms
except ImportError:
    logger.error("loading comtypes only works on Windows")

RETRY_TIMES = 3


class PDFMarkdownReader(BaseReader):
    """PDF parser.

    Converts PDF to markdown (via Word on Windows, or raw text on Linux),
    then splits into section and list_item nodes using MarkdownNodeParser.

    Heading rules are loaded from disk to handle documents where structural
    headings are not marked with # (e.g. bare "Article 17" lines in GDPR).
    """

    def __init__(
        self,
        return_full_document: Optional[bool] = True,
        heading_rules_path: Optional[Path] = None,
        family_id: Optional[str] = None,
        rules_dir: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        return_full_document  : unused legacy flag, kept for compatibility
        heading_rules_path    : direct path to a specific JSON rules file
        family_id             : document family id (e.g. "EU_REGULATION")
                                looks for <rules_dir>/<family_id>.json
        rules_dir             : directory to search for family rule files
                                defaults to ./structure_cache
        """
        self.return_full_document = return_full_document
        self._heading_rules: List[HeadingRule] = self._load_rules(
            heading_rules_path=heading_rules_path,
            family_id=family_id,
            rules_dir=rules_dir or Path("./structure_cache"),
        )
        self.project_root = Path().resolve().parent

    # ── rule loading ─────────────────────────────────────────────────

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
                logger.warning(
                    "No rules file found for family '%s' at %s",
                    family_id, candidate,
                )

        if path is None:
            return []

        if not path.exists():
            logger.warning("Heading rules file not found: %s", path)
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
            logger.info(
                "Loaded %d heading rule(s) from %s", len(rules), path
            )
            return rules
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "Failed to load heading rules from %s: %s", path, exc
            )
            return []

    # ── PDF utilities ─────────────────────────────────────────────────

    def get_tables(self, doc_path: Path):
        with fitz.open(str(doc_path.absolute())) as doc:
            for page in doc:
                yield page.find_tables()

    def extract_text_from_pdf(self, file: Path) -> str:
        text_pages = []
        with fitz.open(str(file)) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text_pages.append(page_text)
        return "\n\n".join(text_pages)

    @retry(stop=stop_after_attempt(RETRY_TIMES))
    def doc_from_pdf(self, file: Path) -> Optional[Path]:
        if coms is None:
            logger.warning(
                "comtypes is unavailable; PDF->DOCX conversion is disabled."
            )
            return None

        directory = str(file.parent)
        basename  = str(file.stem)
        out_dir   = os.path.join(directory, basename + ".docx")
        out_path  = Path(out_dir)
        if out_path.exists():
            return out_path

        word = None
        try:
            word = coms.CreateObject("Word.Application")
            word.visible = 1
            word.DisplayAlerts = 0          # suppress ALL Word dialogs
            doc = word.Documents.Open(str(file))
            doc.SaveAs2(FileName=out_dir, FileFormat=16, Encoding=65001)
            doc.Close()
            return out_path
        except Exception as exc:
            logger.error(
                "Unable to load file %s with Word: %s", str(file), exc
            )
            return None
        finally:
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass

    # ── main entry point ──────────────────────────────────────────────

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """
        Parse a PDF into a list of Document objects.

        Each Document corresponds to one section or list_item node
        as produced by MarkdownNodeParser. Structural metadata
        (h1, h2, header_path, node_type) is preserved on each Document.

        On Windows: PDF → DOCX → Markdown → nodes
        On Linux:   PDF → raw text → nodes (heading rules critical here)
        """
        metadata: Dict = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)
            
        project_root = Path().resolve().parent

        # ── step 1: get markdown text ─────────────────────────────────
        md_text: str = ""
        doc_path = self.doc_from_pdf(file=file)

        if doc_path is not None:
            md_path: Optional[Path] = None
            try:
                md_path = md_from_doc(file=doc_path)
                md_text = clean_tables(md_path) or ""
            finally:
                try:
                    os.remove(doc_path)
                except OSError:
                    pass
                if md_path is not None:
                    try:
                        os.remove(md_path)
                    except OSError:
                        pass
        else:
            # Linux fallback — heading rules are essential here since
            # raw pymupdf text has no # heading markers
            md_text = self.extract_text_from_pdf(file)
            
        # ── step 2: improve structure with AI ──────────────────────────────────
        generator = MarkdownStructureProfileGenerator(
            modeltype=Modeltypes.OPENAI,
        )

        merged_rules = STATIC_GENERIC_PROFILE["rules"] + STATIC_EU_REGULATION_PROFILE["rules"]
        profile = generator.generate_profile(md_text, merged_rules)

        generator.save_profile(
            profile,
            project_root / "structure_cache" / "generated_profiles" / "latest_profile.json",
        )

        md_text = MarkdownStructureRuleApplier(profile).apply(
            md_text,
            output_path=project_root / "structure_cache" / "normalized_markdown" / "latest_markdown.md",
        )

        # ── step 3: parse into nodes ──────────────────────────────────
        parser = MyMarkdownNodeParser.from_defaults()

        source_doc = Document(text=md_text, metadata=metadata)
        nodes      = parser.get_nodes_from_node(source_doc)

        # ── step 4: convert TextNodes back to Documents ───────────────
        # PDFMarkdownReader is a BaseReader so it must return List[Document].
        # ParentDocumentNodeParser (in the pipeline) consumes these and
        # promotes them to TextNodes with parent relationships.
        return [
            Document(
                text=node.get_content(),
                metadata=dict(node.metadata or {}),
                id_=node.node_id,
            )
            for node in nodes
        ]