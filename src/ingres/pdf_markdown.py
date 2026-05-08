"""Docs parser.

Contains parsers for docx, pdf files.

"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt

from fsspec import AbstractFileSystem

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, BaseNode
from src.ingres.util import md_from_doc, clean_tables
import pymupdf as fitz

from src.ingres.markdown_parser import MyMarkdownNodeParser, MyMarkdownElementNodeParser
logger = logging.getLogger(__name__)
coms = None
try:
    import comtypes.client as coms
except ImportError:
    logger.error("loading comtypes only works on Windows")

RETRY_TIMES = 3


class PDFMarkdownReader(BaseReader):
    """PDF parser."""

    def __init__(self, return_full_document: Optional[bool] = True) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document
        
    
    # use camelot to parse tables
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

    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    
    def doc_from_pdf(self, file: Path) -> Optional[Path]:
        if coms is None:
            logger.warning("comtypes is unavailable; PDF->DOCX conversion is disabled.")
            return None

        directory = str(file.parent)
        basename = str(file.stem)
        out_dir = os.path.join(directory, basename + ".docx")
        out_path = Path(out_dir)
        if out_path.exists():
            return out_path

        word = None
        try:
            word = coms.CreateObject("Word.Application")
            word.visible = 0
            doc = word.Documents.Open(str(file))
            doc.SaveAs2(FileName=out_dir, FileFormat=16, Encoding=65001)
            doc.Close()
            return out_path
        except Exception as exc:
            logger.error("Unable to load file %s with Word: %s", str(file), exc)
            return None
        finally:
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
        
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[BaseNode]:
        """Parse file."""

        docs = []
        #parser = LangchainNodeParser(MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on))
        parser = MyMarkdownNodeParser.from_defaults(include_metadata=True,
                                                  include_prev_next_rel=True)
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        doc_path = self.doc_from_pdf(file=file)
        md_docs = ""
        md_path = None
        if doc_path is not None:
            try:
                md_path = md_from_doc(file=doc_path)
                md_docs = clean_tables(md_path)
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
            md_docs = self.extract_text_from_pdf(file)

        d = [Document(text=md_docs, metadata=metadata)]
        docs = parser.get_nodes_from_documents(d)
        md_parser = MyMarkdownElementNodeParser.from_defaults()
        nodes: List[BaseNode] = []
        for doc in docs:
            nodes.extend(md_parser.get_nodes_from_node(doc))

        return nodes


