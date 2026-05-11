from llama_index.core.node_parser.file import MarkdownNodeParser
from llama_index.core.node_parser.relational import MarkdownElementNodeParser
from typing import Any, Callable, List, Optional
from io import StringIO
import csv
import pandas as pd
from heading_rules import HeadingRule
from llama_index.core.node_parser.relational.base_element import (
    BaseElementNodeParser,
    Element,
)
import re
from pymupdf.table import Table

class MyMarkdownNodeParser(MarkdownNodeParser):
    
    def _update_metadata(
        self, headers_metadata: dict, new_header: str, new_header_level: int
    ) -> dict:
        """Update the markdown headers for metadata.

        Removes all headers that are equal or less than the level
        of the newly found header
        """
        updated_headers = {}

        for i in range(1, new_header_level):
            key = f"h{i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        updated_headers[f"h{new_header_level}"] = new_header
        return updated_headers
    
class MyMarkdownElementNodeParser(MarkdownElementNodeParser):
    
    def __init__(
        self,
        *args: Any,
        heading_rules: Optional[List[HeadingRule]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.heading_rules: List[HeadingRule] = heading_rules or []
        
    def _apply_heading_rules(self, line: str) -> Optional[tuple[int, str]]:
        """
        Check line against all heading rules.
        Returns (level, cleaned_text) if a rule matches, else None.
        Only applied to lines that do not already start with '#'.
        """
        if line.startswith("#"):
            return None
        for rule in self.heading_rules:
            if rule.pattern.match(line):
                text = rule.pattern.sub("", line).strip() if rule.strip_match else line
                return rule.level, text
        return None
    
    def md_to_df(self, md_str: str) -> pd.DataFrame | None:
        """Convert Markdown to dataframe."""
        # Replace " by "" in md_str
        md_str = md_str.replace('"', '""')

        # Replace markdown pipe tables with commas
        md_str = md_str.replace("|", '","')

        # Remove the second line (table header separator)
        lines = md_str.split("\n")
        md_str = "\n".join(lines[:1] + lines[2:])

        # Remove the first and last second char of the line (the pipes, transformed to ",")
        lines = md_str.split("\n")
        md_str = "\n".join([line[2:-2] for line in lines])
        
        md_str = self.remove_empty_lines(md_str)

        # Check if the table is empty
        if len(md_str) == 0:
            return None       
        try:
            # Use pandas to read the CSV string into a DataFrame
            return pd.read_csv(StringIO(md_str))
        except Exception:
            print("unable to create table from element {}".format(md_str))
            
    
    def remove_empty_lines(self, element):
        # Use StringIO to simulate file reading
        input_csv = StringIO(element)
        output_csv = StringIO()

        reader = csv.reader(input_csv, quotechar='"')
        writer = csv.writer(output_csv, quotechar='"', quoting=csv.QUOTE_ALL)

        # Read the header and write it to the output
        header = next(reader)
        writer.writerow(header)

        # Filter and write non-empty rows
        for row in reader:
            if any(field.strip() for field in row):  # Check if any field is non-empty
                writer.writerow(row)

        # Print the resulting CSV content
        output_csv.seek(0)
        return output_csv.read()
            
    def extract_elements(
        self,
        text: str,
        node_id: Optional[str] = None,
        table_filters: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> List[Element]:
        # get node id for each node so that we can avoid using the same id for different nodes
        
        # remove page breaks
        pattern = r'\n[-]+\n'
        text = re.sub(pattern, '', text)
        """Extract elements from text."""
        lines = text.split("\n")
        currentElement = None
        numCells = 0
        elements: List[Element] = []
        # Then parse the lines
        for line in lines:
            if line.startswith("```"):
                # check if this is the end of a code block
                if currentElement is not None and currentElement.type == "code":
                    elements.append(currentElement)
                    currentElement = None
                    # if there is some text after the ``` create a text element with it
                    if len(line) > 3:
                        elements.append(
                            Element(
                                id=f"id_{len(elements)}",
                                type="text",
                                element=line.lstrip("```"),
                            )
                        )

                elif line.count("```") == 2 and line[-3] != "`":
                    # check if inline code block (aka have a second ``` in line but not at the end)
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}",
                        type="code",
                        element=line.lstrip("```"),
                    )
                elif currentElement is not None and currentElement.type == "text":
                    currentElement.element += "\n" + line
                else:
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )

            elif currentElement is not None and currentElement.type == "code":
                currentElement.element += "\n" + line

            elif line.startswith("|"):
                #make sure each table line has a closing '|'
                # If it's not an empty line and doesn't end with '|', add '|'
                stripped_line = line.strip()
                if "--" in stripped_line and line.count('|') < numCells:
                    line = stripped_line + ' |'
                if not stripped_line.endswith('|'):
                    line = stripped_line + ' |'
                if currentElement is not None and currentElement.type != "table":
                    numCells = line.count('|')
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
            elif line.startswith("#"):
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}",
                    type="title",
                    element=line.lstrip("#"),
                    title_level=len(line) - len(line.lstrip("#")),
                )
            # ── NEW: extra heading rules ──────────────────────────────
            elif (match := self._apply_heading_rules(line)) is not None:
                level, text_content = match
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}",
                    type="title",
                    element=text_content,
                    title_level=level,
                )
            # ─────────────────────────────────────────────────────────
            else:
                if currentElement is not None and currentElement.type != "text":
                    elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
        if currentElement is not None:
            elements.append(currentElement)

        for idx, element in enumerate(elements):
            if element.type == "table":
                should_keep = True
                perfect_table = True

                # verify that the table (markdown) have the same number of columns on each rows
                table_lines = element.element.split("\n")
                table_columns = [len(line.split("|")) for line in table_lines]
                if len(set(table_columns)) > 1:
                    # if the table have different number of columns on each rows, it's not a perfect table
                    # we will store the raw text for such tables instead of converting them to a dataframe
                    perfect_table = False

                # verify that the table (markdown) have at least 2 rows
                if len(table_lines) < 2:
                    should_keep = False

                # apply the table filter, now only filter empty tables
                if should_keep and perfect_table and table_filters is not None:
                    should_keep = all(tf(element) for tf in table_filters)

                # if the element is a table, convert it to a dataframe
                if should_keep:
                    if perfect_table:
                        table = self.md_to_df(element.element)

                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table",
                            element=element.element,
                            table=table,
                        )
                    else:
                        # for non-perfect tables, we will store the raw text
                        # and give it a different type to differentiate it from perfect tables
                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table_text",
                            element=element.element,
                            # table=table
                        )
                else:
                    elements[idx] = Element(
                        id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                        type="text",
                        element=element.element,
                    )
            else:
                # if the element is not a table, keep it as to text
                elements[idx] = Element(
                    id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                    type="text",
                    element=element.element,
                )

        # merge consecutive text elements together for now
        merged_elements: List[Element] = []
        for element in elements:
            if (
                len(merged_elements) > 0
                and element.type == "text"
                and merged_elements[-1].type == "text"
            ):
                merged_elements[-1].element += "\n" + element.element
            else:
                merged_elements.append(element)
        elements = merged_elements
        
        # remove garbage elements
        final_elements = []
        pattern = r'^[\s\n-]*$'
        for el in merged_elements:
        # Use re.match() to check if the entire text matches the pattern
            if not bool(re.match(pattern, el.element)):
                final_elements.append(el)
        return final_elements

        
    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        
        
        table_df = self.md_to_df(table_element.element)
        

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1