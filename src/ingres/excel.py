from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import logging
import pandas as pd
from src.ingres.markdown_parser import MyMarkdownElementNodeParser
from src.ingres.util import clean_html_tables

log = logging.getLogger(__name__)

class ExcelReader(BaseReader):
    #TODO: add metadata from document itself
    # def load_data(self, file_path: str, extra_info: dict = None):
    #     metadata = extra_info or {}
    #     app = xw.App()
    #     app.visible = False  # Excel application not visible
    #     app.display_alerts = False   # supress alert messages
    #     app.screen_updating = False  # supress screen updates
    #     book = app.books.open(file_path)
    #     documents = []
    #     log.debug("processing file {}".format(file_path))
        
    #     for sheet in book.sheets:
    #         name = sheet.name
    #         # Get the used range of the sheet
    #         used_range = sheet.used_range

    #         # Extract plain text from each cell in the used range
    #         plain_text = ''
    #         # for row in used_range.value:
    #         #     for cell in row:
    #         for element in used_range:
    #             if element.value:
    #                 plain_text += str(element.value) + ' : '
    #                 plain_text += '\n'
    #         data = plain_text
    #         metadata["level"] = 1
    #         metadata["h1"] = name
    #         metadata["h2"] = ""
    #         metadata["h3"] = ""
    #         metadata["category"] = "excel"
    #         document = Document(text=data, metadata=metadata)
    #         documents.append(document)
    #     book.close()
    #     app.quit()
        
    #     return documents
    
    #alternative using pandas
    def load_data(self, file_path: str, extra_info: dict = {}):
        # Load the Excel file
        excel_file = file_path
        metadata = extra_info or {}

        # Read the Excel file
        xls = pd.ExcelFile(excel_file)
        documents = []
        log.debug("processing file {}".format(file_path))

        # Loop through each sheet
        for sheet_name in xls.sheet_names:
            name = sheet_name
            # Read the sheet into a DataFrame
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # plain_text = ''
            # # Loop through each row and column
            # for index, row in df.iterrows():
            #     for column in df.columns:
            #         cell_value = str(row[column])  # Convert cell value to text
            #         if cell_value != 'nan':
            #             plain_text += str(cell_value) + ' : '
            # data = plain_text
            data = df.to_html(index=False) + "  \n"
            metadata["level"] = 1
            metadata["h1"] = name
            metadata["h2"] = ""
            metadata["h3"] = ""
            metadata["category"] = "excel"
            md_data = clean_html_tables(content=data)
            document = Document(text=md_data, metadata=metadata)
            documents.append(document)
        md_parser = MyMarkdownElementNodeParser.from_defaults()
        nodes = []
        for doc in documents:
            nodes.extend(md_parser.get_nodes_from_node(doc))
        return(nodes)
