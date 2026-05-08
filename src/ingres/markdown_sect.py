import logging
import os
import datetime
from typing import List, Dict, Any
from collections import defaultdict
from src.utils.tools import create_uuid_from_string

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pathlib import Path
from src.ingres.markdown_parser import MyMarkdownElementNodeParser

log = logging.getLogger(__name__)

class MarkDownSectionWalker(BaseReader):
    """
    A loader for markdown documents that are organized hierarchically where some sections or subsections are stored in separate files.
    It creates a list of Llamaindex Documents adding metadata on document level and headings 1-3.

    Args:
        BaseReader (_type_): _description_
    """
    
    def load_data(
        self,
        start_file: Path,
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
        if os.path.exists(start_file):
            documents = []
            metadata = {}
            file_content = self.read_markdown_file(start_file)
            sections = self.process_file(filename=str(start_file), file=file_content)
            if extra_info:
                metadata = extra_info
            for section in list(sections.values()):
                if metadata:
                    #metadata["level"] = section["level"]
                    #metadata["h1"] = section["h1"]
                    #metadata["h2"] = section["h2"]
                    #metadata["h3"] = section["h3"]
                    #metadata["category"] = "Markdown"

                    #Update: 22.07.2024
                    metadata_new={
                              "child_path": str(section["filename"]), 
                              "file_name": os.path.basename(section["filename"]),
                              "file_size": float(section["file_size"]),
                              "creation_date": section["creation_date"],
                              "last_modified_date": section["last_modified_date"],
                              "level": section["level"],
                              "h1": section["h1"],
                              "h2": section["h2"],
                              "h3": section["h3"],
                              "category": "Markdown"}
                    metadata=dict(metadata,**metadata_new)
                    #Update End: 22.07.2024
                else:
                    metadata={
                              "child_path": str(section["filename"]), 
                              "file_name": os.path.basename(section["filename"]),
                              "file_size": float(section["file_size"]),
                              "creation_date": section["creation_date"],
                              "last_modified_date": section["last_modified_date"],
                              "level": section["level"],
                              "h1": section["h1"],
                              "h2": section["h2"],
                              "h3": section["h3"],
                              "category": "Markdown"}

                document = Document(
                    text='\n'.join(section["text"]),
                    metadata=metadata,
                )
                document.id_ = create_uuid_from_string(section["id"])
                documents.append(document)
            md_parser = MyMarkdownElementNodeParser.from_defaults()
            nodes = []
            for doc in documents:
                nodes.extend(md_parser.get_nodes_from_node(doc))
            return nodes
        else:
            log.error("File {} not found".format(start_file))
            return []
    
    def read_markdown_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
        
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
        file: str,
        parent_id: str = "",
        parent_section: str = "",
        parent_subsection: str = "",
        parent_subsubsection: str = ""
        
    ) -> Dict[str, Dict[str, Any]]:
        """_summary_

        Args:
            filename (str): absolute path to file
            file (str): content of a markdown file
            parent_section (str, optional): if the parent document contained a section title it is given here. Defaults to None.
            parent_subsection (str, optional): if the parent document contained a subsection title it is given here. Defaults to None.
            parent_subsubsection (str, optional): if the parent document contained a subsubsection title it is given here. Defaults to None.
            level (int, optional): Level corresponding to the heading level from 1-3, e.g. if subsubsection is not none 
            the level is 3. Defaults to None.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of section data where keys are section IDs and values are dictionaries containing section information
        """
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
        
        if parent_section:
            current_section["h1"] = parent_section
            current_section["id"] = "h1_" + parent_section
            current_section["level"] = "1"
        if parent_subsection:
            current_section["h2"] = parent_subsection
            current_section["id"] = "h2_" + parent_subsection
            current_section["level"] = "2"
        if parent_subsubsection:
            current_section["h3"] = parent_subsubsection
            current_section["id"] = "h3_" + parent_subsubsection
            current_section["level"] = "3"
        # if there was already a parent id add the string 'sub' to it
        # and overwrite all previously set ones
        if parent_id:
            current_section["id"] = parent_id + "_sub"
            
        if "id" in current_section.keys():
            current_id = current_section["id"]
            sections[current_section["id"]] = current_section.copy()
            
        for line in file.split('\n'):
            header_level = line.count('#',0,3)
            if header_level > 0:
                lvl = "h{}".format(header_level)
                current_section = defaultdict(str)
                if parent_section:
                    current_section["h1"] = parent_section
                if parent_subsection:
                    current_section["h2"] = parent_subsection
                if parent_subsubsection:
                    current_section["h3"] = parent_subsubsection
                current_section["text"]= ""
                current_section["filename"] = filename
                current_section["creation_date"] = dt_c.strftime('%d-%m-%Y')
                current_section["last_modified_date"] = dt_m.strftime('%d-%m-%Y')
                current_section["file_size"] = str(file_stats.st_size)
                current_section_header = line.strip('#').strip()
                current_section[lvl] = current_section_header
                if header_level == 1:
                    parent_section = current_section_header
                    current_section["h2"] = ""
                    current_section["h3"] = ""
                elif header_level == 2:
                    parent_subsection = current_section_header
                    current_section["h3"] = ""
                current_section["id"] = "{}_{}".format(lvl,current_section_header)
                current_section["level"] = str(header_level)
                current_section["text"] += line.strip('#').strip()
                current_id = current_section["id"]
                sections[current_section["id"]] = current_section.copy()
            elif line.startswith("[!"): #link to another file
                fn = line.strip().strip('[]!').strip()
                if not os.path.isabs(fn):
                    #if fn.startswith('./') or fn.startswith('../'):
                    fn = os.path.join(file_path, fn)
                file_name = os.path.abspath(fn)
                if os.path.exists(file_name):
                    file_content = self.read_markdown_file(file_name)
                    new_sections = self.process_file(
                        filename=file_name, 
                        file=file_content,
                        parent_id=str(current_id),
                        parent_section=current_section["h1"],
                        parent_subsection=current_section["h2"],
                        parent_subsubsection=current_section["h3"])
                    sections = dict(sections, **new_sections)
                else:
                    log.error("File {} not found".format(file_name))
            elif current_id:
                text = line.strip('-').strip()
                if len(text) > 0:
                    sections[current_id]["text"].append(line.strip('-').strip())
                
        return sections

        