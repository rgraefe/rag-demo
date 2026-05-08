"""
This script accepts a PDF document filename and converts it to a text file
in Markdown format, compatible with the GitHub standard.

It must be invoked with the filename like this:

python pymupdf_rag.py input.pdf [-pages PAGES]

The "PAGES" parameter is a string (containing no spaces) of comma-separated
page numbers to consider. Each item is either a single page number or a
number range "m-n". Use "N" to address the document's last page number.
Example: "-pages 2-15,40,43-N"

It will produce a markdown text file called "input.md".

Text will be sorted in Western reading order. Any table will be included in
the text in markdwn format as well.

Dependencies
-------------
PyMuPDF v1.24.2 or later

Copyright and License
----------------------
Copyright 2024 Artifex Software, Inc.
License GNU Affero GPL 3.0
"""

import os
import string

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

from pymupdf4llm.helpers.get_text_lines import get_raw_lines, is_white
from pymupdf4llm.helpers.multi_column import column_boxes
import re
from pathlib import Path
from tqdm import tqdm
from enum import Enum

if fitz.pymupdf_version_tuple < (1, 24, 2):
    raise NotImplementedError("PyMuPDF version 1.24.2 or later is needed.")

bullet = (
    "- ",
    "* ",
    chr(0xF0A7),
    chr(0xF0B7),
    chr(0xB7),
    chr(8226),
    chr(9679),
)
GRAPHICS_TEXT = "\n![%s](%s)\n"

class HeaderTypes(Enum):
    FONTSIZE = 1
    NUMBERS = 2

class IdentifyHeaders:
    """Compute data for identifying header text."""

    def __init__(
        self,
        doc: str,
        pages: list = None,
        body_limit: float = 10,
        header_max:float = 30,
        granularity = True
    ):
        
        self.body_limit = body_limit
        self.header_max = header_max
        font_counts, styles = self.get_font_counts(
            doc= doc,
            pages= pages,
            granularity = granularity
        )
        
        self.font_tags(styles=styles, font_counts=font_counts)
    
    def get_font_counts(
        self,
        doc: str,
        pages: list,
        granularity
    ):
        """
        “flags” is an integer, which represents font properties except for the first bit 0. They are to be interpreted like this:

        bit 0: superscripted (20) – not a font property, detected by MuPDF code.

        bit 1: italic (21)

        bit 2: serifed (22)

        bit 3: monospaced (23)

        bit 4: bold (24)
        """
        
        """Extracts fonts and their usage in PDF documents.

        :param doc: PDF document to iterate through
        :type doc: <class 'fitz.fitz.Document'>
        :param granularity: also use 'font', 'flags' and 'color' to discriminate text
        :type granularity: bool

        :rtype: [(font_size, count), (font_size, count}], dict
        :return: most used fonts sorted by count, font style information
        """
        styles = {}
        font_counts = {}
        occurrence_counts = {}
        
        if isinstance(doc, fitz.Document):
            mydoc = doc
        else:
            mydoc = fitz.open(doc)

        if pages is None:  # use all pages if omitted
            pages = range(mydoc.page_count)
            
        for page in mydoc:
            for pno in pages:
                page = mydoc.load_page(pno)
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
                for line in [  # look at all non-empty horizontal spans
                    l
                    for b in blocks
                    for l in b["lines"]
                    #for s in l["spans"]
                ]:
                    text = " ".join(s["text"] for s in line["spans"])
                    span0= line["spans"][0]
                    if granularity:
                        numbered = self.detect_pattern(text)
                        identifier = "{}_{}_{}_{}".format(round(span0['size']), span0['flags'], span0['font'], numbered)
                        styles[identifier] = {'size': int(round(span0['size'])), 'flags': span0['flags'], 'font': span0['font'],
                                            'numbered': numbered}
                    else:
                        identifier = "{0}".format(span0['size'])
                        styles[identifier] = {'size': int(round(span0['size'])), 'font': span0['font']}

                    font_counts[identifier] = font_counts.get(identifier, 0) + len(text.strip())
                    occurrence_counts[identifier] = occurrence_counts.get(identifier, 0) + 1 
                    
        if mydoc != doc:
            # if opened here, close it now
            mydoc.close()

        font_counts_s = dict(sorted(
            [(k, v) for k, v in font_counts.items()],
            key=lambda i: i[1],
            reverse=True,
        ))
        
        styles_s = dict(sorted(
            [(k, v) for k, v in styles.items()],
            key=lambda i: i[1]["size"],
            reverse=True,
        ))
        
                

        if len(font_counts) < 1:
            raise ValueError("Zero discriminating fonts found!")

        return font_counts_s, styles_s
    
    def font_tags(self, font_counts, styles):
        """Returns dictionary with font sizes as keys and tags as value.

        :param font_counts: (font_size, count) for all fonts occuring in document
        :type font_counts: list
        :param styles: all styles found in the document
        :type styles: dict

        :rtype: dict
        :return: all element tags based on font-sizes
        """
                # detecting paragraph size
        self.p_size = self.body_limit
        for key in font_counts.keys():
            if int(styles[key]["size"]) >= self.body_limit:
                self.p_size = styles[key]["size"]
                break

        #get the top n largest font sizes
        self.size_dict = [(s,styles[s]) for s in styles.keys() if int(styles[s]["size"]) > self.p_size
                          and int(styles[s]["size"]) <= self.header_max]
        levels = {}
        s_l1 = False
        s_l2 = False
        s_l3 = False
        self.sizes = []
        self.header_id = {}
        last_size = 0
        for n in self.size_dict:
            if not s_l1:
                levels["s_l1"] = n[0]
                self.sizes.append(n[1]["size"])
                s_l1 = True
                last_size = n[1]["size"]
            elif s_l1 and not s_l2:
                if n[1]["size"] == last_size:
                    continue
                levels["s_l2"] = n[0]
                self.sizes.append(n[1]["size"])
                s_l2 = True
                last_size = n[1]["size"]
            elif s_l1 and s_l2 and not s_l3:
                if n[1]["size"] == last_size:
                    continue
                levels["s_l3"] = n[0]
                self.sizes.append(n[1]["size"])
                s_l3 = True
            elif s_l1 and s_l2 and s_l3:
                break
            
        
            
        # styles should be already sorted by numbered items
        self.numbered=[(s,styles[s]) for s in styles.keys() if int(styles[s]["numbered"]) > 0]
        num_l1 = False
        num_l2 = False
        num_l3 = False
        for n in self.numbered:
            if int(n[1]["numbered"]) == 1:
                if not num_l1:
                    levels["num_l1"] = n[0]
                    num_l1 = True
            elif int(n[1]["numbered"]) == 2:
                if num_l1 and not num_l2:
                    levels["num_l2"] = n[0]
                    num_l2 = True
            elif int(n[1]["numbered"]) == 3:
                if num_l1 and num_l2 and not num_l3:
                    levels["num_l3"] = n[0]
                    num_l3 = True
                
        # just in case the header type will be based on sizes we create the size dict
        # in the other case the level string will be generated out of the span text later
        for i, size in enumerate(self.sizes):
            self.header_id[size] = "#" * (i + 1) + " "
        self.header_type = self.detect_header_type(font_counts=font_counts, levels=levels)
    
    def detect_header_type(self, font_counts: dict, levels:dict) -> HeaderTypes:
        numbers = 0
        sizes = 0
        s_l1 = levels.get("s_l1", None)
        num_l1 = levels.get("num_l1", None)
        if s_l1	and num_l1:
            if font_counts[levels["s_l1"]] > font_counts[levels["num_l1"]]:
                sizes += 1
            else:
                numbers += 1
        elif s_l1:
            sizes += 1
        elif num_l1:
            numbers += 1
        s_l2 = levels.get("s_l2", None)
        num_l2 = levels.get("num_l2", None)
        if s_l2	and num_l2:
            if font_counts[levels["s_l2"]] > font_counts[levels["num_l2"]]:
                sizes += 1
            else:
                numbers += 1
        elif s_l2:
            sizes += 1
        elif num_l2:
            numbers += 1
        s_l3 = levels.get("s_l3", None)
        num_l3 = levels.get("num_l3", None)
        if s_l3	and num_l3:
            if font_counts[levels["s_l3"]] > font_counts[levels["num_l3"]]:
                sizes += 1
            else:
                numbers += 1
        elif s_l3:
            sizes += 1
        elif num_l3:
            numbers += 1
            
        if sizes > numbers:
            return HeaderTypes.FONTSIZE
        else:
            return HeaderTypes.NUMBERS
        
            
    def detect_pattern(self, string, font=None, size= None):
        if size and size < self.p_size:
            return 0
        if font and "BOLD" not in font:
            return 0
        # Define the patterns
        patterns = {
            r'^\d{1,2}\.\d{1,2}\.\d{1,2}.+': 3,   # Pattern for a digit.digit.digit
            r'^\d{1,2}\.\d{1,2}.+': 2,      # Pattern for a digit.digit
            r'^\d{1,2}\s[^\.\/].+': 1          # Pattern for 1-2 digits without following dot or slash

            }
        
        # Check the string against each pattern
        for pattern, value in patterns.items():
            if re.match(pattern, string):
                return value
        
        # If no pattern matches, return None
        return 0

    def get_header_id(self, text, span: dict, page=None) -> str:
        """Return appropriate markdown header prefix.

        Given a text span from a "dict"/"rawdict" extraction, determine the
        markdown header prefix string of 0 to n concatenated '#' characters.
        """
        if self.header_type == HeaderTypes.NUMBERS:
            hdr_id = self.detect_pattern(text, span["font"], span["size"])
            if hdr_id:
                return "#" * hdr_id + " "
            else:
                return ""
        else:
            hdr_id = self.header_id.get(int(round(span["size"])), "")
            return hdr_id


def pdf_to_markdown(
    doc: Path,
    *,
    pages: list = None,
    hdr_info=None,
    write_images: bool = False,
    page_chunks: bool = False,
    delete_headers: bool = False,
    margins=(0, 50, 0, 50),
    dpi=150,
    page_width=612,
    page_height=None,
    table_strategy="lines_strict",
    graphics_limit=None,
) -> str:
    """Process the document and return the text of the selected pages.

    Args:
        doc: pymupdf.Document or pathlib Path.
        pages: list of page numbers to consider (0-based).
        hdr_info: callable or object having a method named 'get_hdr_info'.
        write_images: (bool) whether to save images / drawing as files.
        page_chunks: (bool) whether to segment output by page.
        margins: do not consider content overlapping margin areas.
        dpi: (int) desired resolution for generated images.
        page_width: (float) assumption if page layout is variable.
        page_height: (float) assumption if page layout is variable.
        table_strategy: choose table detection strategy
        graphics_limit: (int) ignore page with too many vector graphics.

    """

    DPI = dpi
    GRAPHICS_LIMIT = graphics_limit
    if not isinstance(doc, fitz.Document):
        doc = fitz.open(str(doc.absolute()))

    # for reflowable documents allow making 1 page for the whole document
    if doc.is_reflowable:
        if hasattr(page_height, "__float__"):
            # accept user page dimensions
            doc.layout(width=page_width, height=page_height)
        else:
            # no page height limit given: make 1 page for whole document
            doc.layout(width=page_width, height=792)
            page_count = doc.page_count
            height = 792 * page_count  # height that covers full document
            doc.layout(width=page_width, height=height)

    if pages is None:  # use all pages if no selection given
        pages = list(range(doc.page_count))

    if hasattr(margins, "__float__"):
        margins = [margins] * 4
    if len(margins) == 2:
        margins = (0, margins[0], 0, margins[1])
    if len(margins) != 4:
        raise ValueError("margins must be a float or a sequence of 2 or 4 floats")
    elif not all([hasattr(m, "__float__") for m in margins]):
        raise ValueError("margin values must be floats")

    # If "hdr_info" is not an object having method "get_header_id", scan the
    # document and use font sizes as header level indicators.
    if callable(hdr_info):
        get_header_id = hdr_info
    elif hasattr(hdr_info, "get_header_id") and callable(hdr_info.get_header_id):
        get_header_id = hdr_info.get_header_id
    else:
        hdr_info = IdentifyHeaders(doc)
        get_header_id = hdr_info.get_header_id

    def resolve_links(links, span):
        """Accept a span and return a markdown link string.

        Args:
            links: a list as returned by page.get_links()
            span: a span dictionary as returned by page.get_text("dict")

        Returns:
            None or a string representing the link in MD format.
        """
        bbox = fitz.Rect(span["bbox"])  # span bbox
        # a link should overlap at least 70% of the span
        for link in links:
            hot = link["from"]  # the hot area of the link
            middle = (hot.tl + hot.br) / 2  # middle point of hot area
            if not middle in bbox:
                continue  # does not touch the bbox
            text = f'[{span["text"].strip()}]({link["uri"]})'
            return text

    def save_image(page, rect, i):
        """Optionally render the rect part of a page."""
        filename = page.parent.name.replace("\\", "/")
        image_path = f"{filename}-{page.number}-{i}.png"
        if write_images is True:
            page.get_pixmap(clip=rect, dpi=DPI).save(image_path)
            return os.path.basename(image_path)
        return ""

    def write_text(
        page: fitz.Page,
        textpage: fitz.TextPage,
        clip: fitz.Rect,
        tabs=None,
        tab_rects: dict = None,
        img_rects: dict = None,
        links: list = None,
    ) -> string:
        """Output the text found inside the given clip.

        This is an alternative for plain text in that it outputs
        text enriched with markdown styling.
        The logic is capable of recognizing headers, body text, code blocks,
        inline code, bold, italic and bold-italic styling.
        There is also some effort for list supported (ordered / unordered) in
        that typical characters are replaced by respective markdown characters.

        'tab_rects'/'img_rects' are dictionaries of table, respectively image
        or vector graphic rectangles.
        General Markdown text generation skips these areas. Tables are written
        via their own 'to_markdown' method. Images and vector graphics are
        optionally saved as files and pointed to by respective markdown text.
        """
        if clip is None:
            clip = textpage.rect
        out_string = ""

        # This is a list of tuples (linerect, spanlist)
        nlines = get_raw_lines(textpage, clip=clip, tolerance=3)

        tab_rects0 = list(tab_rects.values())
        img_rects0 = list(img_rects.values())

        prev_lrect = None  # previous line rectangle
        prev_bno = -1  # previous block number of line
        code = False  # mode indicator: outputting code
        prev_hdr_string = None

        for lrect, spans in nlines:
            # there may tables or images inside the text block: skip them
            if intersects_rects(lrect, tab_rects0) or intersects_rects(
                lrect, img_rects0
            ):
                continue

            # Pick up tables intersecting this text block
            for i, tab_rect in sorted(
                [
                    j
                    for j in tab_rects.items()
                    if j[1].y1 <= lrect.y0 and not (j[1] & clip).is_empty
                ],
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                out_string += "\n" + tabs[i].to_markdown(clean=False) + "\n"
                del tab_rects[i]

            # Pick up images / graphics intersecting this text block
            for i, img_rect in sorted(
                [
                    j
                    for j in img_rects.items()
                    if j[1].y1 <= lrect.y0 and not (j[1] & clip).is_empty
                ],
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    out_string += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]

            text = " ".join([s["text"] for s in spans])

            # if the full line mono-spaced?
            all_mono = all([s["flags"] & 8 for s in spans])

            if all_mono:
                if not code:  # if not already in code output  mode:
                    out_string += "```\n"  # switch on "code" mode
                    code = True
                # compute approx. distance from left - assuming a width
                # of 0.5*fontsize.
                delta = int((lrect.x0 - clip.x0) / (spans[0]["size"] * 0.5))
                indent = " " * delta

                out_string += indent + text + "\n"
                continue  # done with this line

            span0 = spans[0]
            bno = span0["block"]  # block number of line
            if bno != prev_bno:
                out_string += "\n"
                prev_bno = bno

            if (  # check if we need another line break
                prev_lrect
                and lrect.y1 - prev_lrect.y1 > lrect.height * 1.5
                or span0["text"].startswith("[")
                or span0["text"].startswith(bullet)
                or span0["flags"] & 1  # superscript?
            ):
                out_string += "\n"
            prev_lrect = lrect

            # if line is a header, this will return multiple "#" characters
            #hdr_string = get_header_id(span0, page=page)
            hdr_string = get_header_id(text, span0, page=page)

            # intercept if header text has been broken in multiple lines
            if hdr_string and hdr_string == prev_hdr_string:
                out_string = out_string[:-1] + " " + text + "\n"
                continue

            prev_hdr_string = hdr_string
            if hdr_string.startswith("#"):  # if a header line skip the rest
                out_string += hdr_string + text + "\n"
                continue

            # this line is not all-mono, so switch off "code" mode
            if code:  # still in code output mode?
                out_string += "```\n"  # switch of code mode
                code = False

            for i, s in enumerate(spans):  # iterate spans of the line
                # decode font properties
                mono = s["flags"] & 8
                bold = s["flags"] & 16
                italic = s["flags"] & 2

                if mono:
                    # this is text in some monospaced font
                    out_string += f"`{s['text'].strip()}` "
                else:  # not a mono text
                    prefix = ""
                    suffix = ""
                    if hdr_string == "":
                        if bold:
                            prefix = "**"
                            suffix += "**"
                        if italic:
                            prefix += "_"
                            suffix = "_" + suffix

                    # convert intersecting link into markdown syntax
                    ltext = resolve_links(links, s)
                    if ltext:
                        text = f"{hdr_string}{prefix}{ltext}{suffix} "
                    else:
                        text = f"{hdr_string}{prefix}{s['text'].strip()}{suffix} "

                    if text.startswith(bullet):
                        text = "-  " + text[1:]
                    out_string += text
            if not code:
                out_string += "\n"
        out_string += "\n"
        if code:
            out_string += "```\n"  # switch of code mode
            code = False

        return (
            out_string.replace(" \n", "\n").replace("  ", " ").replace("\n\n\n", "\n\n")
        )

    def is_in_rects(rect, rect_list):
        """Check if rect is contained in a rect of the list."""
        for i, r in enumerate(rect_list, start=1):
            if rect in r:
                return i
        return 0

    def intersects_rects(rect, rect_list):
        """Check if middle of rect is contained in a rect of the list."""
        for i, r in enumerate(rect_list, start=1):
            if (rect.tl + rect.br) / 2 in r:  # middle point is inside r
                return i
        return 0

    def output_tables(tabs, text_rect, tab_rects):
        """Output tables above a text rectangle."""
        this_md = ""  # markdown string for table content
        if text_rect is not None:  # select tables above the text block
            for i, trect in sorted(
                [j for j in tab_rects.items() if j[1].y1 <= text_rect.y0],
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                this_md += tabs[i].to_markdown(clean=False)
                del tab_rects[i]  # do not touch this table twice

        else:  # output all remaining table
            for i, trect in sorted(
                tab_rects.items(),
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                this_md += tabs[i].to_markdown(clean=False)
                del tab_rects[i]  # do not touch this table twice
        return this_md

    def output_images(page, text_rect, img_rects):
        """Output images and graphics above text rectangle."""
        if img_rects is None:
            return ""
        this_md = ""  # markdown string
        if text_rect is not None:  # select tables above the text block
            for i, img_rect in sorted(
                [j for j in img_rects.items() if j[1].y1 <= text_rect.y0],
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    this_md += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]  # do not touch this image twice

        else:  # output all remaining table
            for i, img_rect in sorted(
                img_rects.items(),
                key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    this_md += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]  # do not touch this image twice
        return this_md

    def get_metadata(doc, pno):
        meta = doc.metadata.copy()
        meta["file_path"] = doc.name
        meta["page_count"] = doc.page_count
        meta["page"] = pno + 1
        return meta
    
    def extract_header_candidates(docs):
        header_candidates = []
        footer_candidates = []
            
        for doc in docs:
            page = doc["text"]
            header_candidates.append(page[:5])
            footer_candidates.append(page[-5:])
            
        return (header_candidates, footer_candidates)
    
    def jaccard_similarity(x,y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)
    
    def compare(a, b):
        '''Fuzzy matching of strings to compare headers/footers in neighboring pages'''
        
        count = 0
        a = re.sub('\d{1,4}', '@', a)
        b = re.sub('\d{1,4}', '@', b)
        for x, y in zip(a, b):
            if x == y:
                count += 1
        return count / max(len(a), len(b))
        #return jaccard_similarity(a, b)

    
    def remove_header(docs, header_candidates, WIN):
        # source:https://medium.com/@hussainshahbazkhawaja/paper-implementation-header-and-footer-extraction-by-page-association-3a499b2552ae
        '''Remove headers from content dictionary. Helper function for remove_header_footer() function.'''
        
        header_weights = [1.0, 0.75, 0.5, 0.5, 0.5]
        
        for i, candidate in enumerate(header_candidates):
            temp = header_candidates[max(i-WIN, 1) : min(i+WIN, len(header_candidates))]
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] =  sublist + [''] * (maxlen - len(sublist))
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn,cm) * header_weights[j]
                    score = score/len(temp)
                except:
                    score = header_weights[j]
                if score > 0.3:
                    detected.append(cn)
            del temp
            
            for d in detected:
                while d in docs[i]["text"][:5]:
                    docs[i]["text"].remove(d)
                    
        return docs
    
    def remove_footer(docs, footer_candidates, WIN):
        '''Remove footers from content dictionary. Helper function for remove_header_footer() function.'''
        
        footer_weights = [0.5, 0.5, 0.5, 0.75, 1.0]
        
        for i, candidate in enumerate(footer_candidates):
            temp = footer_candidates[max(i-WIN, 1) : min(i+WIN, len(footer_candidates))]
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] =  [''] * (maxlen - len(sublist)) + sublist
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn,cm)
                    score = score/len(cmp)
                except:
                    score = footer_weights[j]
                if score > 0.5:
                    detected.append(cn)
            del temp
            
            for d in detected:
                while d in docs[i]["text"][-5:]:
                    docs[i]["text"] = docs[i]["text"][::-1]
                    docs[i]["text"].remove(d)
                    docs[i]["text"] = docs[i]["text"][::-1]
                    
        return docs
    
    def split_lines(docs):
        for doc in docs:
            page = doc["text"]
            page = page.replace('\n\n','  \n')
            page = page.replace('\n\n','\n')
            page = [line for line in page.split('\n')]
            doc["text"] = page
        return docs
    
    def merge_lines(docs):
        for doc in docs:
            doc["text"] = "\n".join(doc["text"])
        return docs
    
    def remove_headers(docs):
        WIN = 8
        docs = split_lines(docs)
        header_candidates, footer_candidates = extract_header_candidates(docs)
        docs = remove_header(docs, header_candidates, WIN)
        docs = remove_footer(docs, footer_candidates, WIN)
        docs = merge_lines(docs)
        return docs

    def get_page_output(doc, pno, margins, textflags):
        """Process one page.

        Args:
            doc: fitz.Document
            pno: 0-based page number
            textflags: text extraction flag bits

        Returns:
            Markdown string of page content and image, table and vector
            graphics information.
        """
        page = doc[pno]
        md_string = ""
        if GRAPHICS_LIMIT is not None:
            test_paths = page.get_cdrawings()
            if (excess := len(test_paths)) > GRAPHICS_LIMIT:
                md_string = (
                    f"\n**Ignoring page {page.number} with {excess} vector graphics.**"
                )
                md_string += "\n\n-----\n\n"
                return md_string, [], [], []
        left, top, right, bottom = margins
        clip = page.rect + (left, top, -right, -bottom)
        # extract external links on page
        links = [l for l in page.get_links() if l["kind"] == fitz.LINK_URI]

        # make a TextPage for all later extractions
        textpage = page.get_textpage(flags=textflags, clip=clip)

        img_info = [img for img in page.get_image_info() if img["bbox"] in clip]
        images = img_info[:]
        tables = []
        graphics = []

        # Locate all tables on page
        tabs = page.find_tables(clip=clip, strategy=table_strategy)

        # Make a list of table boundary boxes.
        # Must include the header bbox (which may exist outside tab.bbox)
        tab_rects = {}
        for i, t in enumerate(tabs):
            tab_rects[i] = fitz.Rect(t.bbox) | fitz.Rect(t.header.bbox)
            tab_dict = {
                "bbox": tuple(tab_rects[i]),
                "rows": t.row_count,
                "columns": t.col_count,
            }
            tables.append(tab_dict)

        # list of table rectangles
        tab_rects0 = list(tab_rects.values())

        # Select paths that are not contained in any table
        page_clip = page.rect + (36, 36, -36, -36)  # ignore full page graphics
        paths = [
            p
            for p in page.get_drawings()
            if not intersects_rects(p["rect"], tab_rects0)
            and p["rect"] in page_clip
            and p["rect"].width < page_clip.width
            and p["rect"].height < page_clip.height
        ]

        # We also ignore vector graphics that only represent "background
        # sugar".
        vg_clusters = []  # worthwhile vector graphics go here

        # walk through all vector graphics not belonging to a table
        for bbox in page.cluster_drawings(drawings=paths):
            subbox = bbox + (3, 3, -3, -3)  # sub rect without any border
            box_area = abs(bbox)
            include = False
            for p in paths:
                mp = (p["rect"].tl + p["rect"].br) / 2  # center point of rect

                # fill-only paths or being part of the border will not
                # make this a worthwhile vector grahic
                if mp not in subbox or p["type"] == "f":
                    continue

                # this checks if all items are part of the bbox border
                near_border = set()
                for itm in p["items"]:  # walk through path items
                    if itm[0] == "re":  # a full-sized rectangle
                        if abs(itm[1]) / box_area < 1e-3:
                            near_border.add(True)  # is part of the border
                    elif itm[0] in ("c", "l"):  # curves and lines
                        for temp in itm[1:]:
                            # if their points are on the border
                            near_border.add(temp not in subbox)
                # if any stroked path has a point inside bbox (i.e. not on its
                # border then this vector graphic is treated as significant
                if not near_border == {True}:
                    include = True
                    break
            if include is True:  # this box is a significant vector graphic
                vg_clusters.append(bbox)

        actual_paths = [p for p in paths if is_in_rects(p["rect"], vg_clusters)]

        vg_clusters0 = [
            r
            for r in vg_clusters
            if not intersects_rects(r, tab_rects0) and r.height > 20
        ]

        if write_images is True:
            vg_clusters0 += [fitz.Rect(i["bbox"]) for i in img_info]

        vg_clusters = dict((i, r) for i, r in enumerate(vg_clusters0))

        # Determine text column bboxes on page, avoiding tables and graphics
        text_rects = column_boxes(
            page,
            paths=actual_paths,
            no_image_text=write_images,
            textpage=textpage,
            avoid=tab_rects0 + vg_clusters0,
            footer_margin=margins[3],
            header_margin=margins[1],
        )

        """Extract markdown text iterating over text rectangles.
        We also output any tables. They may live above, below or inside
        the text rectangles.
        """
        for text_rect in text_rects:
            # output tables above this block of text
            md_string += output_tables(tabs, text_rect, tab_rects)
            md_string += output_images(page, text_rect, vg_clusters)

            # output text inside this rectangle
            md_string += write_text(
                page,
                textpage,
                text_rect,
                tabs=tabs,
                tab_rects=tab_rects,
                img_rects=vg_clusters,
                links=links,
            )

        md_string = md_string.replace(" ,", ",").replace("-\n", "")
        # write any remaining tables and images
        md_string += output_tables(tabs, None, tab_rects)
        md_string += output_images(None, tab_rects, None)
        md_string += "\n-----\n\n"
        while md_string.startswith("\n"):
            md_string = md_string[1:]
        return md_string, images, tables, graphics

    if page_chunks is False and delete_headers is False:
        document_output = ""
    else:
        document_output = []

    # read the Table of Contents
    toc = doc.get_toc()
    textflags = fitz.TEXT_MEDIABOX_CLIP
    for pno in tqdm(pages, desc="processing pages"):
        page_output, images, tables, graphics = get_page_output(
            doc, pno, margins, textflags
        )
        if page_chunks is False and delete_headers is False:
            document_output += page_output
        else:
            # build subet of TOC for this page
            page_tocs = [t for t in toc if t[-1] == pno + 1]

            metadata = get_metadata(doc, pno)
            document_output.append(
                {
                    "metadata": metadata,
                    "toc_items": page_tocs,
                    "tables": tables,
                    "images": images,
                    "graphics": graphics,
                    "text": page_output,
                }
            )
    if delete_headers is True:
        document_output = remove_headers(document_output)
        if page_chunks == False:
            document_text = [x["text"] for x in document_output]
            document_output = "\n---\n".join(document_text)
        
    return document_output


