import mistune
from enum import Enum
import re

class MarkdownTypes(Enum):
    SUB = 1
    FINAL = 2

class CitationFormatter:
    
    rootdir = "LLM"
    
    @staticmethod
    def escape_markdown(text):
        escape_chars = r'([_\[\]()~`>#+\-=|{}.!\\])'
        return re.sub(escape_chars, r'\\\1', text)
    
    @staticmethod
    def get_markdown(fields: dict, type:MarkdownTypes):
        question = fields["question"]
        answer = CitationFormatter.escape_markdown(fields["answer"])
        cite_details = fields.get("cite_details", None)
        ct = ""
        if cite_details:
            ct = '\n'.join([f"- {item}" for item in cite_details])
            ct = CitationFormatter.escape_markdown(ct)

        else:
            ct = "There were no citations."
        if type == MarkdownTypes.SUB:
            template = f"""
### Sub-Question:
{question}

### Sub-Answer:
{answer}

### Citations
{ct}
            """
        else:
            template = f"""
### Final-Question:
{question}

### Final-Answer:
{answer}

### Citations
{ct}
            """
        markdown = mistune.create_markdown(escape=False)
        t = markdown(template)
        return t.replace('\n', ' ')
        
    @staticmethod
    def check_and_extract_brackets(text):
        # Regular expression pattern to match brackets containing integers at the end of sentences
        pattern = re.compile(r'\[(\d+)\](?=[\.\?!]|$|(?:\[(\d+)\]))')
        
        # Check if any matches exist
        if pattern.search(text):
            # Find all matches
            matches = pattern.finditer(text)
            return [match.group(1) for match in matches]
        else:
            return None
    
    @staticmethod
    def get_value(data, primary_key, fallback_key):
        # this is a hack to get the right header-1 key
        # I had made a mistake with naming the key
        # so some use l1 and some use h1
        if data.get(primary_key):
            return data[primary_key]
        elif data.get(fallback_key):
            return data[fallback_key]
        else:
            return None
    
    @staticmethod    
    def prepare_cite_text(question:str, answer:str, citations:list, source_nodes, type:MarkdownTypes):
        
        cite_list = []
        fields = {}
        if citations:
            cite_set = sorted({int(element) for element in citations})
            for c in cite_set:
                cite_detail = ""
                node = source_nodes[c-1]
                metadata = node.metadata
                if metadata:
                    file_path = metadata["file_path"]
                    parts = file_path.split(CitationFormatter.rootdir, 1)
                    if len(parts) > 1:
                        sub_path = parts[1]
                    else:
                        sub_path = file_path
                    cite_detail += f"[{c}]  "
                    cite_detail += "\n**Directory:** " + sub_path + "  "
                    text = metadata["window"]
                    cite_detail += "\n**Text:** " + text + "  "
                    h1 = "**Heading 1:** " + CitationFormatter.get_value(metadata,"l1", "h1")  + "  "
                    cite_detail += "\n" + h1
                    h2 = metadata.get("h2", "")
                    if len(h2) > 0:
                        h2 = "**Heading 2:** " + h2  + "  "
                        cite_detail += "\n" + h2
                    h3 = metadata.get("h3", "")
                    if len(h3) > 0:
                        h3 = "**Heading 3:** " + h3  + "  "
                        cite_detail += "\n" + h3
                    cite_list.append(cite_detail)
            fields["cite_details"] = cite_list
        fields["answer"] = answer
        fields["question"] = question

        markdown = CitationFormatter.get_markdown(fields,type)
        return markdown
        
    @staticmethod
    def merge_cite_results(history):
        markdown_list = []
        for h in list(history.keys())[:-1]:
            question = history[h]["Question"]
            answer = history[h]["Answer"]
            citations = CitationFormatter.check_and_extract_brackets(answer)
            if citations:
                source_nodes = history[h]["source_nodes"]
                markdown = CitationFormatter.prepare_cite_text(question=question, answer=answer, citations=citations, source_nodes=source_nodes, type=MarkdownTypes.SUB)
                markdown_list.append(markdown)
        h = list(history.keys())[-1]
        question = history[h]["Question"]
        answer = history[h]["Answer"]
        citations = CitationFormatter.check_and_extract_brackets(answer)
        if citations:
            source_nodes = history[h]["source_nodes"]
            markdown = CitationFormatter.prepare_cite_text(question=question, answer=answer, citations=citations, source_nodes=source_nodes, type=MarkdownTypes.FINAL)
            markdown_list.append(markdown)
        else:
            markdown = CitationFormatter.prepare_cite_text(question=question, answer=answer, citations=citations, source_nodes=None, type=MarkdownTypes.FINAL)
            markdown_list.append(markdown)
        return " ".join(markdown_list)