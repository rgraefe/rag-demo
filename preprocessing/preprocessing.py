import logging, sys
from llama_index.core.schema import TransformComponent
import re

log = logging.getLogger(__name__)


class TextCleaner1st(TransformComponent):
    def __call__(self, documents, **kwargs):
        #getting rid of the samples that are rather short after cleaning
        filtered=[]
        logging.info("TextCleaner")
        for i in range(len(documents)):
            doc_text=documents[i].text
            if (documents[i].dict()["metadata"]["file_name"][-5:]!=".xlsx"): 
                doc_text=re.sub(r"\u200b", "", doc_text) # cleaning "zero-width-space" chars
                doc_text = re.sub(r"(Confidential. [ ]*A[ ]*ll r)(\D+)(authoritative and controlling.)", "", doc_text) # Cleaning this part which is present for many documents as a template
                instances=len(re.findall(r"[a-zA-Z0-9_\.@]+ [a-zA-Z0-9_\.@]+\s*[A-Za-z0-9\._%-]+@\s*[A-Za-z0-9\.-]+\.com [0-9]{8}", doc_text)) #cleaning the duplicates of the name, email,id tuples
                doc_text = re.sub(r"[a-zA-Z0-9_\.@]+ [a-zA-Z0-9_\.@]+\s*[A-Za-z0-9\._%-]+@\s*[A-Za-z0-9\.-]+\.com [0-9]{8}", "", doc_text,count=instances-1)
                doc_text = re.sub(r"[\.]{4}", "", doc_text) #removing glosary part's points 
                doc_text = re.sub(r"( \.){2,}", " .", doc_text) #removing glosary part's points
                doc_text = re.sub(r"\n +\n", "\n", doc_text) # removing excessive new lines
                doc_text = re.sub(r"\n\n+", "\n", doc_text) # removing excessive new lines 
                doc_text = re.sub(r"((\s*undefined)\s(undefined\s*))+", "\n undefined \n", doc_text) #removing specific "undefined" sequences
                doc_text = re.sub(r"(?<!.)( )*(```)*xls\([ ]*[\"'].+[\"'][ ]*\)\n", "", doc_text) # cleaning lines that have only xlsx file paths
                doc_text = re.sub(r"(?<!.)\[.*[\.]md[ ]*\](\n)", "", doc_text) # todo: check the point usage here # cleaning lines that have only md file paths
                toadd= documents[i].copy()
                toadd.text=doc_text
                filtered.append(toadd)
                    
        return filtered
