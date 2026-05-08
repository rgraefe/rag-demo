import uuid
import hashlib
from IPython.display import Markdown, display
from llama_index.core.schema import Document, TextNode

def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))
        
def document_to_node(doc: Document):
    node = TextNode(id_= doc.id_,
                    embedding=doc.embedding,
                    extra_info=doc.extra_info,
                    excluded_llm_metadata_keys=doc.excluded_embed_metadata_keys,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                    relationships=doc.relationships,
                    text = doc.text,
                    start_char_idx=doc.start_char_idx,
                    end_char_idx=doc.end_char_idx,
                    text_template=doc.text_template,
                    metadata_template=doc.metadata_template,
                    metadata_seperator=doc.metadata_seperator)
    return node