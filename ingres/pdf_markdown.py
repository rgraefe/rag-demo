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
from ingres.util import md_from_doc, clean_tables
import pymupdf as fitz

from ingres.markdown_parser import MyMarkdownNodeParser, MyMarkdownElementNodeParser
logger = logging.getLogger(__name__)
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
    def get_tables(self, doc:Path):
        
        doc = fitz.open(str(doc.absolute()))
        for page, i in enumerate(doc):
            tabs = page.find_tables()
    
    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    
    def doc_from_pdf(self, file: Path) -> Path:
        directory = str(file.parent)
        basename = str(file.stem)
        out_dir = os.path.join(directory,basename+".docx")
        out_path = Path(out_dir)
        if os.path.exists(out_dir):
            return out_path
        else:
            word = coms.CreateObject("Word.Application")
            word.visible = 1
            try:
                doc = word.Documents.Open(str(file))
                doc.SaveAs2(FileName=out_dir, FileFormat=16, Encoding=65001)
                doc.Close()
            except Exception:
                logger.error("Unable to load file {} with Word".format(str(file)))
                out_path = ""
            finally:     
                word.Quit()
            return out_path
        
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
        # This block returns section chunks generated from a whole PDF 
        
        # debug, leave out unreadable files:
        lines = []
        with open('C:\\Users\\rgraefe\\OneDrive - Intel Corporation\\Documents\\LLM\\unreadablePdf.csv', 'r') as f:
            # Iterate over each line in the file
            for line in f:
                # Strip the newline character and append the line to the list
                
                lines.append(line.strip())
        if str(file) in lines:
            #out_file = Path(str(file).replace(".pdf", "_new.pdf"))
            #remove_watermark(file, out_file, short_wt="Henning Schroeder henning.schroeder@intel.com", long_wt="1136492 Henning Schroeder henning.schroeder@intel.com 1136492 Henning Schroeder henning.schroeder@intel.com 1136492 Henning Schroeder henning.schroeder@intel.com 1136492 Henning Schroeder henning.schroeder@intel.com ")
            return []
        else:
            doc_path = self.doc_from_pdf(file=file)
            md_path = md_from_doc(file=doc_path)
            os.remove(doc_path)
            #md_path = Path("C:\\Users\\rgraefe\\OneDrive - Intel Corporation\\Documents\\LLM\\Aptiv\\Aptiv_OSP_Fusion_SoC_Requirements.md")
            md_docs = clean_tables(md_path)
            os.remove(md_path)
            d = [Document(text=md_docs, metadata=metadata),]
            docs = parser.get_nodes_from_documents(d)
            md_parser = MyMarkdownElementNodeParser.from_defaults()
            nodes = []
            for doc in docs:
                nodes.extend(md_parser.get_nodes_from_node(doc))

            return nodes
            pass
        return None


