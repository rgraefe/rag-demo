import os
from llama_index.core.readers.base import BaseReader
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem
from llama_index.core.schema import Document
log = logging.getLogger(__name__)
try:
    import comtypes.client as coms
except ImportError:
    log.error("loading comtypes only works on Windows")
from llama_index.readers.file import PDFReader



class VisioReader(BaseReader):
    
    def __init__(self):
        super(VisioReader, self).__init__()
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        #open and safe as pdf first, then parse with PDF reader
        visio = coms.CreateObject('Visio.Application')
        doc = visio.Documents.Open(str(file))
        directory = str(file.parent)
        basename = str(file.stem)
        out_dir = os.path.join(directory,basename+".pdf")
        out_path = Path(out_dir)
        doc.ExportAsFixedFormat( 1, out_dir, 1, 0 )
        reader = PDFReader()
        documents = reader.load_data(out_path, extra_info=extra_info)
        os.remove(out_dir)
        doc.Close()
        visio.Quit()
        return documents