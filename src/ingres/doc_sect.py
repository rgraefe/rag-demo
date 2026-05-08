import os
from llama_index.core.readers.base import BaseReader
import logging
from pathlib import Path
from llama_index.core.schema import Document, TextNode
from docx.document import Document as DocxDocument    
import re
from collections import defaultdict
import datetime
from src.utils.tools import create_uuid_from_string
from typing import List
from bs4 import BeautifulSoup
from src.ingres.markdown_parser import MyMarkdownNodeParser, MyMarkdownElementNodeParser
from src.ingres.util import clean_tables, md_from_doc

log = logging.getLogger(__name__)

class DocxSectionReader(BaseReader):
    
    def __init__(self):
        super(BaseReader, self).__init__()
    
    def load_data(
        self,
        file: Path,
        extra_info: dict = {},
    ) -> List[Document]:
        """
        Load markdown data starting with the given file and following links to section documents. 
        Each document contains the lowest level section content. E.g. if there is no sub-section between
        2 section headers than the document will be of level 1 and only the header1 metadata will be filled.
        If there are sub-sections between 2 sections then there will be as many documents as sub-sections. 
        The meta-data for these lower level documents will be: {level: "2", heading1: <parent heading>, heading2: <documents heading>}
        If text belongs to the parent section then this text will be a separate document.
        The lowest possible level is 3, in that cases heading1, heading2 and heading3 are filled in the metadata.

        Parameters
        ----------
        start_file : str
            The top level file to start with.

        Returns:
        -------
        List[Document]
            A list of loaded documents.
        """
        if os.path.exists(file):
            # documents = []
            # doc = Document(file)
            # sections, tables = self.process_file(filename=file, doc=doc)
            metadata = extra_info
            # for section in list(sections.values()):
            #     if metadata:
            #         metadata["level"] = section["level"]
            #         metadata["h1"] = section["h1"]
            #         metadata["h2"] = section["h2"]
            #         metadata["h3"] = section["h3"]
            #         metadata["category"] = "Markdown"
            #     else:
            #         metadata={
            #                   "file_path": section["filename"], 
            #                   "file_name": os.path.basename(section["filename"]),
            #                   "file_size": float(section["file_size"]),
            #                   "creation_date": section["creation_date"],
            #                   "last_modified_date": section["last_modified_date"],
            #                   "level": section["level"],
            #                   "h1": section["h1"],
            #                   "h2": section["h2"],
            #                   "h3": section["h3"],
            #                   "category": "MSWord"}
            #     txt = '\n'.join(section["text"])
            #     txt = re.sub(r'\n+', '\n', txt)
            #     document = Document(
            #         text=txt,
            #         metadata=metadata,
            #     )
            #     document.id_ = create_uuid_from_string(section["id"])
            #     documents.append(document)
            parser = MyMarkdownNodeParser.from_defaults(include_metadata=True,
                                                  include_prev_next_rel=True)
            md_path = md_from_doc(file=file)
            
            md_docs = clean_tables(md_path)
            os.remove(md_path)
            d = [Document(text=md_docs, metadata=metadata),]
            docs = parser.get_nodes_from_documents(d)
            md_parser = MyMarkdownElementNodeParser.from_defaults()
            nodes = []
            for doc in docs:
                if isinstance(doc, TextNode):
                    nodes.extend(md_parser.get_nodes_from_node(doc))
            return nodes
            #return documents
        else:
            log.error("File {} not found".format(file))
            return []
        
    def get_file_times(self, path):
        # file modification timestamp of a file
        m_time = os.path.getmtime(path)
        # convert timestamp into DateTime object
        dt_m = datetime.datetime.fromtimestamp(m_time)

        # file creation timestamp in float
        c_time = os.path.getctime(path)
        # convert creation timestamp into DateTime object
        dt_c = datetime.datetime.fromtimestamp(c_time)
        return dt_m, dt_c
    
    def process_file(
        self,
        filename: str,
        doc: DocxDocument,        
    ) -> list:
        """_summary_

        Args:
            filename (str): absolute path to file
            doc (DocxDocument): content of a Word document
            parent_section (str, optional): if the parent document contained a section title it is given here. Defaults to None.
            parent_subsection (str, optional): if the parent document contained a subsection title it is given here. Defaults to None.
            parent_subsubsection (str, optional): if the parent document contained a subsubsection title it is given here. Defaults to None.
            level (int, optional): Level corresponding to the heading level from 1-3, e.g. if subsubsection is not none 
            the level is 3. Defaults to None.

        Returns:
            List[Document]: List of Llamaindex Document objects with metadata
        """
        parent_section = None
        parent_subsection = None
        current_id = None
        dt_m, dt_c = self.get_file_times(filename)
        file_stats = os.stat(filename)
        file_path = os.path.dirname(filename)
        sections = defaultdict(dict)
        current_section = defaultdict(str)
        current_section["text"]= ""
        current_section["filename"] = filename
        current_section["creation_date"] = dt_c.strftime('%d-%m-%Y')
        current_section["last_modified_date"] = dt_m.strftime('%d-%m-%Y')
        current_section["file_size"] = str(file_stats.st_size)
        
        
        for para in doc.paragraphs:
            p_text = ""
            soup = BeautifulSoup(para._p.xml)
            for el in soup.find_all("w:p"):
                text = ' '.join([line for line in el.text.split('\n') if line.strip() != ''])
                
                if len(text) > 0 and re.search(r'^.*[\d\w\W]+.*$',text):
                    p_text += el.text.strip()
                #print(repr("Element Text: {}".format(el.text.replace("\n", ""))))
            style_name = (para.style.name or "") if para.style else ""
            match = re.match(r'^Heading ([0-9]+)', style_name)
            if match:
                p_text = re.sub(r'\n+', '', p_text)
                header_level = int(match.group(1))
        
                lvl = "h{}".format(header_level)
                current_section = defaultdict(str)
                if parent_section:
                    current_section["h1"] = parent_section
                if parent_subsection:
                    current_section["h2"] = parent_subsection
                current_section["text"]= ""
                current_section["filename"] = filename
                current_section["creation_date"] = dt_c.strftime('%d-%m-%Y')
                current_section["last_modified_date"] = dt_m.strftime('%d-%m-%Y')
                current_section["file_size"] = str(file_stats.st_size)
                current_section_header = p_text
                current_section[lvl] = current_section_header
                if header_level == 1:
                    parent_section = current_section_header
                    current_section["h2"] = ""
                    current_section["h3"] = ""
                if header_level == 2:
                    parent_subsection = current_section_header
                    current_section["h3"] = ""
                current_section["id"] = "{}_{}".format(lvl,current_section_header)
                current_section["level"] = str(header_level)
                current_section["text"] += p_text.strip()
                current_id = current_section["id"]
                sections[current_section["id"]] = current_section.copy()
             
            else:
                if current_id:
                    if len(p_text) > 0:
                        endswithnewline = p_text.endswith('\n')
                        p_text = re.sub(r'\n+', '', p_text)
                        if endswithnewline:
                            p_text = p_text + '\n'
                        p_text = re.sub(r'\s+', ' ', p_text).strip()
                        sections[current_id]["text"] += p_text
                else:
                    current_id = "Cover"
                    lvl = "h0"
                    current_section = defaultdict(str)
                    current_section["text"]= ""
                    current_section["filename"] = str(filename)
                    current_section["creation_date"] = dt_c.strftime('%d-%m-%Y')
                    current_section["last_modified_date"] = dt_m.strftime('%d-%m-%Y')
                    current_section["file_size"] = str(file_stats.st_size)
                    current_section_header = "Cover"
                    current_section[lvl] = current_section_header
                    current_section["id"] = "{}_{}".format(lvl,current_section_header)
                    current_section["level"] = "0"
                    if len(p_text) > 0:
                        p_text = re.sub(r'\n+', '\n', p_text)
                        p_text = re.sub(r'\s+', ' ', p_text).strip()
                        sections[current_id]["text"] += p_text
                    current_id = current_section["id"]
                    sections[current_section["id"]] = current_section.copy()
              
        return list(sections)
        