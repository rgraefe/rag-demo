
import os
from pathlib import Path
import pypandoc
from bs4 import BeautifulSoup
import re
import pandas as pd
import pymupdf as fitz

def md_from_doc(file:Path) -> Path:
    directory = str(file.parent)
    basename = str(file.stem)
    out_dir = os.path.join(directory,basename+".md")
    out_path = Path(out_dir)
    pypandoc.convert_file(str(file), to='gfm', format='docx', outputfile=out_dir)
    return out_path

def html_to_md_table(html_content:str) -> str:
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table in the HTML
    table = soup.find('table')
    
    contains_only_th = True
    # check if table only contains header rows
    for row in table.find_all('tr'):
        for cell in row.find_all(['th', 'td']):
            if cell.name == 'td':
                contains_only_th = False
                break
        if not contains_only_th:
            break
        
    # Extract headers
    headers = [header.text.replace('\n','') for header in table.find_all('th')]
    if contains_only_th:
        patterns = {
            r'^\d{1,2}\.\d{1,2}\.\d{1,2}.*': 3,   # Pattern for a digit.digit.digit
            r'^\d{1,2}\.\d{1,2}.*': 2,      # Pattern for a digit.digit
            r'^\d{1,2}\s[^\.\/].*': 1          # Pattern for 1-2 digits without following dot or slash

            }
    
        # Check the string against each pattern
        is_number = False
        level = 0
        for pattern, value in patterns.items():
            if re.match(pattern, headers[0]):
                is_number = True
                level = value
                break
        if is_number:
            str = "#" * level + " " + " ".join(headers)
        else:
            str = ""
        return str

            
    # Initialize an empty list to store the new table rows
    new_table = []

    # Initialize a list to keep track of cells that span multiple rows
    rowspan_tracker = []

    # Process each row in the original table
    for row in table.find_all('tr'):
        new_row = []            
        if not rowspan_tracker:
            col_idx = 0
            for cell in row.find_all(['th', 'td']):
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                txt = cell.text.replace("\n\n","\n")
                txt = txt.replace("\n"," ").strip()
                
                # Add the cell content to the current row
                for _ in range(colspan):
                    new_row.append(txt)
                    rowspan_tracker.insert(col_idx, [rowspan - 1, txt])
                    col_idx += 1
                # Track rowspan for future rows
        else:
        # Add cells from the rowspan_tracker first
            cells = row.find_all(['th', 'td'])
            col_idx = 0
            while col_idx < len(rowspan_tracker):
                if rowspan_tracker[col_idx][0] > 0:
                    new_row.append(rowspan_tracker[col_idx][1])
                    rowspan_tracker[col_idx][0] -= 1
                    col_idx += 1
                else:
                    try:
                        tmp = cells.pop(0)
                        txt = tmp.text.replace("\n\n","\n")
                        txt = txt.replace("\n"," ").strip()
                    except Exception:
                        txt = ""
                    rowspan = int(tmp.get('rowspan', 1))
                    colspan = int(tmp.get('colspan', 1))
                    # Add the cell content to the current row
                    for _ in range(colspan):
                        new_row.append(txt)
                        rowspan_tracker[col_idx] = [rowspan - 1,txt]
                        col_idx += 1
        # Ensure the new row has the correct number of columns
        new_table.append(new_row)

    # Find the maximum number of columns in the new table
    max_cols = max(len(row) for row in new_table)

    # Normalize rows to have the same number of columns
    for row in new_table:
        while len(row) < max_cols:
            row.append('')
    # Create a DataFrame
    if headers: 
        df = pd.DataFrame(new_table[1:], columns=headers)
    else:
        df = pd.DataFrame(new_table)
        
    return df.to_markdown(index=False) + "\n\n"
    # # remove uneccessary spaces
    # for m in markdown.split('\n'):
        
        
def find_lowest_level_tables(table):
    if (isinstance(table, str)):
        soup = BeautifulSoup(table, 'html.parser')
    else:
        soup = table
    nested_tables = soup.find_all('table', recursive=False)  # Find only direct child tables
    if not nested_tables:  # If there are no nested tables, it's a lowest-level table
        return [table]
    
    lowest_level_tables = []
    for inner_table in nested_tables:
        lowest_level_tables.extend(find_lowest_level_tables(inner_table))  # Recursively find lowest-level tables
    return lowest_level_tables

def clean_html_tables(content: str) -> str:
    f = content
    out_str = []
    is_table = False
    nested_table = False
    table_str = []
    num_table = 0
    lines = f.split('\n')
    for line in lines:
        if "<table" in line:
            num_table += 1
            if is_table:
                nested_table = True
            else:
                is_table = True
            table_str.append(line)
        elif "</table>" in line:
            table_str.append(line)
            num_table -= 1
            if num_table == 0:
                if nested_table: # keep the original string, its too complicated for pandas
                    n_tables = find_lowest_level_tables('\n'.join(table_str))
                    md_string = ""
                    for nt in n_tables:
                        try:
                            md_string += html_to_md_table(str(nt)) + "\n"
                        except:
                            md_string = str(nt)
                else:
                    try:
                        md_string = html_to_md_table('\n'.join(table_str))
                    except ValueError as e:
                        md_string = '\n'.join(table_str)
                #md_split = md_string.split('\n')
                #for s in md_split:
                out_str.append(md_string)
                is_table = False
                table_str = []
            else:
                table_str.append(line)
        elif is_table == True:
            table_str.append(line)
        elif is_table == False:
            if '#' in line:
                pattern = r'(\S)[ ]{0,1}\n'
                line = re.sub(pattern,r'\1  \n', line)
            out_str.append(line)
    return ''.join(out_str)

def clean_tables(file:Path) -> str:
    content = []
    with open(str(file), errors='replace') as f:
        for line in f:
            content.append(line)
        out_str = clean_html_tables('\n'.join(content))
    return out_str
                

def is_watermark(text, short_wt, long_wt):
    text = text.replace('\u200b','')
    text = text.replace('\n','')
    text = text.replace('\n','')
    text = re.sub(r'\d+', '', text)
    if text in long_wt or short_wt in text:
        return True
    else:
        return False
    
def extract_elements(page):
    # Extract all elements with their positions and formatting
    elements = []
    
    # Extract text elements
    text_instances = page.get_text("dict")["blocks"]
    for block in text_instances:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    elements.append({"type": "text", "span": span})
    
    # Extract images and other non-text elements
    image_list = page.get_images(full=True)
    for img in image_list:
        xref = img[0]
        pix = fitz.Pixmap(page.parent, xref)
        bbox = page.get_image_bbox(img)
        elements.append({"type": "image", "bbox": bbox, "pix": pix})

    return elements

def get_rgb_from_color_int(color_int):
    # Convert the color integer to a hexadecimal string
    hex_color = f"{color_int:06x}"
    
    # Extract the RGB components from the hexadecimal string
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    
    return r, g, b
    
def remove_watermark(input_pdf, output_pdf, short_wt, long_wt):
    # Open the input PDF
    long_wt = re.sub(r'\d+', '', long_wt)
    short_wt = re.sub(r'\d+', '', short_wt)

    # Open the PDF file
    pdf_document = fitz.open(input_pdf)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        for xref in page.get_contents():
            stream = pdf_document.xref_stream(xref).replace(b'The string to delete', b'')
            pdf_document.update_stream(xref, stream)
        elements = extract_elements(page)
        
        # Identify and redact diagonal text
        for elem in elements:
            if elem["type"] == "text":
                if is_watermark(text=elem["span"]["text"], short_wt=short_wt,
                                                           long_wt=long_wt):
                    rect = fitz.Rect(elem["span"]["bbox"])
                    page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()

        # Reinsert non-diagonal elements
        for elem in elements:
            if elem["type"] == "image":
                page.insert_image(elem["bbox"], pixmap=elem["pix"])
            elif elem["type"] == "text" and not is_watermark(text=elem["span"]["text"], short_wt=short_wt,
                                                           long_wt=long_wt):
                span = elem["span"]
                page.insert_text(
                    (span["bbox"][0], span["bbox"][1]),  # Position
                    span["text"],  # Text
                    fontsize=span["size"],
                    #fontname=span["font"],
                    #color="black"
                )
            

    # Save the modified PDF
    pdf_document.save(output_pdf, garbage=4, deflate=True)
