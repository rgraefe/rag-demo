import os
import fnmatch
import mimetypes
from functools import reduce
import multiprocessing
import warnings
from datetime import datetime
from functools import reduce
from itertools import repeat
from tqdm import tqdm

from llama_index.core.readers.base import BaseReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode, NodeRelationship
import sys
import fsspec
from fsspec.implementations.local import LocalFileSystem
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path, PurePosixPath
from src.utils import FileCache
from src.utils.tools import document_to_node
import logging

log = logging.getLogger(__name__)


sys.path.append('../')


def _try_loading_included_file_formats(
    family_id: Optional[str] = None,
    rules_dir: Optional[Path] = None,
) -> Dict[str, Union[Type[BaseReader], BaseReader]]:
    
    """
    This function dynamically imports file readers from llama_index and creates a mapping
    of file extensions to their corresponding reader classes. It includes support for
    standard formats (PDF, DOCX, PPTX, etc.) as well as custom readers for specialized
    formats (Excel, Visio, Markdown sections, etc.).
    
    Returns:
        Dict[str, Type[BaseReader]]: A dictionary mapping file extensions (e.g., '.pdf', '.docx')
            to their corresponding BaseReader subclass types.
            
    Raises:
        ImportError: If the required llama-index-readers-file package is not found.
    """
    try:
        from llama_index.readers.file import (
            DocxReader,
            EpubReader,
            HWPReader,
            ImageReader,
            IPYNBReader,
            MarkdownReader,
            MboxReader,
            PandasCSVReader,
            PDFReader,
            PptxReader,
            VideoAudioReader,
        )  # pants: no-infer-dep
        from src.ingres import MarkDownSectionWalker, ExcelReader, PptxSlideReader, VisioReader, DocxSectionReader, PDFMarkdownReader
    except ImportError:
        raise ImportError("`llama-index-readers-file` package not found")
    
    pdf_reader  = PDFMarkdownReader(family_id=family_id, rules_dir=rules_dir)
    #docx_reader = DocxSectionReader(family_id=family_id, rules_dir=rules_dir)

    return {
        ".hwp":   HWPReader,
        ".pdf":   pdf_reader,        # instance, not class
        ".docx":  DocxSectionReader,    
        ".pptx":  PptxSlideReader,
        ".ppt":   PptxSlideReader,
        ".pptm":  PptxSlideReader,
        ".jpg":   ImageReader,
        ".png":   ImageReader,
        ".jpeg":  ImageReader,
        ".mp3":   VideoAudioReader,
        ".mp4":   VideoAudioReader,
        ".csv":   PandasCSVReader,
        ".epub":  EpubReader,
        ".md":    MarkDownSectionWalker,
        ".mmd":   MarkDownSectionWalker,
        ".mbox":  MboxReader,
        ".ipynb": IPYNBReader,
        ".xls":   ExcelReader,
        ".xlsx":  ExcelReader,
        ".xlsm":  ExcelReader,
        ".vsdx":  VisioReader,
        ".vsd":   VisioReader,
    }


def _format_file_timestamp(timestamp: Optional[float]) -> Optional[str]:
    """Format file timestamp to a %Y-%m-%d string.

    Args:
        timestamp (Optional[float]): timestamp in float or None

    Returns:
        Optional[str]: formatted timestamp or None if invalid
    """
    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    except Exception:
        return None


def default_file_metadata_func(
    file_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
) -> Dict:
    """Extract metadata for a file from the filesystem.

    Retrieves file statistics and metadata such as file name, type, size, and timestamps
    (creation date, last modified date, last accessed date). This metadata is then
    attached to documents when they are loaded.

    Args:
        file_path (str): The path to the file as a string.
        fs (Optional[fsspec.AbstractFileSystem]): The filesystem interface to use.
            Defaults to the local filesystem if not provided.

    Returns:
        Dict: A dictionary containing non-null metadata fields including file_path,
            file_name, file_type, file_size, creation_date, last_modified_date,
            and last_accessed_date.
    """
    fs = fs or get_default_fs()
    stat_result = fs.stat(file_path)

    try:
        file_name = os.path.basename(str(stat_result["name"]))
    except Exception as e:
        file_name = os.path.basename(file_path)

    creation_date = _format_file_timestamp(stat_result.get("created"))
    last_modified_date = _format_file_timestamp(stat_result.get("mtime"))
    last_accessed_date = _format_file_timestamp(stat_result.get("atime"))
    default_meta = {
        "file_path": file_path,
        "file_name": file_name,
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": stat_result.get("size"),
        "creation_date": creation_date,
        "last_modified_date": last_modified_date,
        "last_accessed_date": last_accessed_date,
    }

    # Return not null value
    return {
        meta_key: meta_value
        for meta_key, meta_value in default_meta.items()
        if meta_value is not None
    }


def get_default_fs() -> fsspec.AbstractFileSystem:
    """Get the default local filesystem interface.
    
    Returns:
        fsspec.AbstractFileSystem: A LocalFileSystem instance for interacting with
            the local filesystem.
    """
    return LocalFileSystem()


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    """Check if the given filesystem is the default local filesystem.
    
    Args:
        fs (fsspec.AbstractFileSystem): The filesystem to check.
        
    Returns:
        bool: True if the filesystem is a LocalFileSystem instance with auto_mkdir
            disabled, False otherwise.
    """
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


# own factory, extending simple directory reader
class ReaderFactory(SimpleDirectoryReader):
    supported_suffix_fn: Callable = _try_loading_included_file_formats
    
    def __init__(
        self,
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude: Optional[List] = None,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        encoding: str = "utf-8",
        filename_as_id: bool = False,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, BaseReader]] = None,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        raise_on_error: bool = False,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        tree_root_glob: Optional[List[str]] = None,
        cached_documents: bool = False
    ) -> None:
        """Initialize the ReaderFactory with configuration parameters.
        
        This factory extends SimpleDirectoryReader with support for multiple file types
        and hierarchical file structures. It can use caching and supports parallel
        file loading across multiple worker processes.
        
        Args:
            input_dir (Optional[str]): Root directory to read files from.
            input_files (Optional[List]): Specific list of files to read instead of scanning a directory.
            exclude (Optional[List]): Patterns for files/directories to exclude from loading.
            exclude_hidden (bool): Whether to exclude hidden files. Defaults to True.
            errors (str): How to handle encoding/decoding errors. Defaults to "ignore".
            recursive (bool): Whether to recursively search subdirectories. Defaults to False.
            encoding (str): Text encoding for files. Defaults to "utf-8".
            filename_as_id (bool): Whether to use filename as the document ID. Defaults to False.
            required_exts (Optional[List[str]]): Only load files with these extensions.
            file_extractor (Optional[Dict[str, BaseReader]]): Mapping of file extensions to custom readers.
            num_files_limit (Optional[int]): Maximum number of files to load.
            file_metadata (Optional[Callable[[str], Dict]]): Function to extract metadata from file paths.
            raise_on_error (bool): Whether to raise exceptions on file read errors. Defaults to False.
            fs (Optional[fsspec.AbstractFileSystem]): Custom filesystem interface to use.
            tree_root_glob (Optional[List]): Glob patterns for hierarchical file root files (e.g., ['*.md']).
            cached_documents (bool): Whether to use cached documents. Defaults to False.
        """
        self.tree_root_glob= tree_root_glob
        self.cached_documents = cached_documents
        super(ReaderFactory, self).__init__(
            input_dir=input_dir,
            input_files=input_files,
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            errors=errors,
            recursive=recursive,
            encoding=encoding,
            filename_as_id=filename_as_id,
            required_exts=required_exts,
            file_extractor=file_extractor,
            num_files_limit=num_files_limit,
            file_metadata=file_metadata,
            raise_on_error=raise_on_error,
            fs=fs,
        )
        

        
    def depth(self, path):
        """Calculate the depth (number of path components) of a file path.
        
        Args:
            path: A file path as a string or Path object.
            
        Returns:
            int: The number of path components in the path.
        """
        path_obj = Path(path)
        return len(path_obj.parts)
    
    def get_index_files(self, input_dir, tree_root_glob) -> List[str]:
        """Find all files matching a glob pattern within a directory tree.
        
        Recursively searches the input directory for files matching the specified
        glob pattern. This is typically used to identify root index files in a
        hierarchical file structure (e.g., finding all *.md files that serve as
        section indices).
        
        Args:
            input_dir: The root directory to search.
            tree_root_glob (str): A glob pattern to match filenames against.
            
        Returns:
            List[str]: A list of full file paths matching the pattern.
        """
        index_files = []
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if fnmatch.fnmatch(filename, tree_root_glob):
                    index_files.append(os.path.join(root, filename))
        return index_files

    def get_file_refs(self, input_dir, ending) -> List[str]:
        """Find all files with a specific extension within a directory tree.
        
        Recursively searches the input directory for all files ending with the
        specified extension.
        
        Args:
            input_dir: The root directory to search.
            ending (str): The file extension to match (without leading dot).
            
        Returns:
            List[str]: A list of full file paths with the specified extension.
        """
        file_refs = []
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.' + ending):
                    file_refs.append(os.path.join(root, filename))
        return file_refs
            
    def _add_files(self, input_dir: Path | PurePosixPath) -> list[Path | PurePosixPath]:
        """Recursively discover and filter files from the input directory.
        
        This method scans the input directory and applies multiple filters:
        - Excludes files/directories matching exclude patterns
        - Removes non-root files from hierarchical file structures (e.g., sections of .md files)
        - Filters by required extensions if specified
        - Excludes hidden files if configured
        - Respects the num_files_limit if set
        
        Args:
            input_dir (Path | PurePosixPath): The root directory to scan for files.
            
        Returns:
            list[Path | PurePosixPath]: A sorted list of file paths that passed all filters.
            
        Raises:
            ValueError: If no files are found in the input directory after filtering.
        """
        all_files = set()
        rejected_files = set()
        rejected_dirs = set()
        # Default to POSIX paths for non-default file systems (e.g. S3)
        _Path = Path if is_default_fs(self.fs) else PurePosixPath
        
        if self.exclude is not None:
            for excluded_pattern in self.exclude:
                if self.recursive:
                    # Recursive glob
                    excluded_glob = _Path(input_dir) / _Path("**") / excluded_pattern
                else:
                    # Non-recursive glob
                    excluded_glob = _Path(input_dir) / excluded_pattern
                for file in self.fs.glob(str(excluded_glob)):
                    file_path = str(file)
                    if self.fs.isdir(file_path):
                        rejected_dirs.add(_Path(file_path))
                    else:
                        rejected_files.add(_Path(file_path))
        
        # remove files that belong to a hierarchy of files but are not the root file, like with .md
        if self.tree_root_glob:
            for tree_root_pattern in self.tree_root_glob:
                ending = tree_root_pattern.split('.')[-1]
                index_files = self.get_index_files(input_dir, tree_root_pattern)
                file_refs = self.get_file_refs(input_dir, ending)
                logging.debug(f"Index files for pattern {tree_root_pattern}: {index_files}")
                logging.debug(f"File refs for ending {ending}: {file_refs}")
                excluded_glob = [f for f in file_refs if f not in index_files]
                logging.debug(f"Excluded files: {excluded_glob}")
                for file in excluded_glob:
                        file_path = str(file)
                        if self.fs.isdir(file_path):
                            rejected_dirs.add(_Path(file_path))
                        else:
                            rejected_files.add(_Path(file_path))

        file_refs = []
        if self.recursive:
            file_refs = self.fs.glob(os.path.join(str(input_dir), "**","*"))
            # Sort paths by their depth
            file_refs = sorted(file_refs, key=self.depth)
        else:
            file_refs = self.fs.glob(os.path.join(str(input_dir), "*"))

        for ref in file_refs:
            # Manually check if file is hidden or directory instead of
            # in glob for backwards compatibility.
            ref = _Path(str(ref))
            is_dir = self.fs.isdir(ref)
            skip_because_hidden = self.exclude_hidden and self.is_hidden(ref)
            skip_because_bad_ext = (
                self.required_exts is not None and ref.suffix not in self.required_exts
            )
            skip_because_excluded = ref in rejected_files
            if not skip_because_excluded:
                if is_dir:
                    ref_parent_dir = ref
                else:
                    ref_parent_dir = self.fs._parent(ref)
                for rejected_dir in rejected_dirs:
                    if str(ref_parent_dir).startswith(str(rejected_dir)):
                        skip_because_excluded = True
                        log.debug(
                            "Skipping %s because it in parent dir %s which is in %s",
                            ref,
                            ref_parent_dir,
                            rejected_dir,
                        )
                        break

            if (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
            ):
                continue
            else:
                all_files.add(ref)

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            raise ValueError(f"No files found in {input_dir}.")

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        log.debug(
            f"> [ReaderFactory] Total files added: {len(new_input_files)}"
        )
        endings = set()
        for inf in new_input_files:
            endings.add(Path(inf).suffix)

        return new_input_files
    
    def extract_start_dir(self, document: Document, root_dir: Path | PurePosixPath) -> None:
        """Extract and add the top-level directory name to document metadata.
        
        For a document loaded from a nested directory structure, this method
        calculates the relative path from the root and extracts the top-level
        directory name, storing it in the document's metadata as 'main_dir'.
        This is useful for organizing documents by their primary category or folder.
        
        Args:
            document (Document): The document to update with directory metadata.
            root_dir (Path): The root directory to calculate the relative path from.
        """
        metadata = document.metadata
        file_dir = metadata["file_path"]
        relative_path = os.path.relpath(file_dir, root_dir)

        # Split the path to get the next directory
        next_directory = relative_path.split(os.sep)[0]
        metadata["main_dir"] = next_directory
        document.metadata = metadata
        
    @staticmethod
    def load_file(
        input_file: Path | PurePosixPath,
        file_metadata: Callable[[str], Dict],
        file_extractor: Dict[str, BaseReader],
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """
        Static method for loading file.

        NOTE: necessarily as a static method for parallel processing.

        Args:
            input_file (Path): _description_
            file_metadata (Callable[[str], Dict]): _description_
            file_extractor (Dict[str, BaseReader]): _description_
            filename_as_id (bool, optional): _description_. Defaults to False.
            encoding (str, optional): _description_. Defaults to "utf-8".
            errors (str, optional): _description_. Defaults to "ignore".
            fs (Optional[fsspec.AbstractFileSystem], optional): _description_. Defaults to None.

        input_file (Path): File path to read
        file_metadata ([Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
        file_extractor (Dict[str, BaseReader]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text.
        filename_as_id (bool): Whether to use the filename as the document id.
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        raise_on_error (bool): Whether to raise an error if a file cannot be read.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
            to using the local file system. Can be changed to use any remote file system

        Returns:
            List[Document]: loaded documents
        """
        # TODO: make this less redundant
        default_file_reader_cls = ReaderFactory.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())
        metadata: Optional[dict] = None
        documents: List[Document] = []

        if file_metadata is not None:
            metadata = file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()
        if file_suffix in default_file_reader_suffix or file_suffix in file_extractor:
            if file_suffix not in file_extractor:
                reader_cls = default_file_reader_cls[file_suffix]
                # ── handle both class and pre-instantiated reader ──
                if isinstance(reader_cls, type):
                    file_extractor[file_suffix] = reader_cls()
                else:
                    file_extractor[file_suffix] = reader_cls
            reader = file_extractor[file_suffix]

            # load data -- catch all errors except for ImportError
            try:
                kwargs: dict[str, Any] = {
                    "extra_info": metadata
                }
                if fs and not is_default_fs(fs):
                    kwargs["fs"] = fs
                docs = reader.load_data(input_file, **kwargs)
                            # iterate over docs if needed
                if filename_as_id:
                    for i, doc in enumerate(docs):
                        doc.id_ = f"{input_file!s}_part_{i}"

                documents.extend(docs)
                   
            except ImportError as e:
                # ensure that ImportError is raised so user knows
                # about missing dependencies
                raise ImportError(str(e))
            except Exception as e:
                if raise_on_error:
                    raise Exception("Error loading file") from e
                # otherwise, just skip the file and report the error
                print(
                    f"Failed to load file {input_file} with error: {e}. Skipping...",
                    flush=True,
                )
                return []


        else:
            # do standard read
            fs = fs or get_default_fs()
            with fs.open(input_file, errors=errors, encoding=encoding) as f:
                data = f.read()

            doc = Document(text=data, metadata=metadata or {})
            if filename_as_id:
                doc.id_ = str(input_file)

            documents.append(doc)

        return documents

    @staticmethod
    async def aload_file(
        input_file: Path | PurePosixPath,
        file_metadata: Optional[Callable[[str], dict]],
        file_extractor: dict[str, BaseReader],
        filename_as_id: bool = False,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
        fs: fsspec.AbstractFileSystem | None = None,
    ) -> list[Document]:

        default_file_reader_cls = ReaderFactory.supported_suffix_fn()
        default_file_reader_suffix = list(default_file_reader_cls.keys())

        metadata: dict | None = None
        documents: list[Document] = []

        if file_metadata is not None:
            metadata = file_metadata(str(input_file))

        file_suffix = input_file.suffix.lower()

        if file_suffix in default_file_reader_suffix or file_suffix in file_extractor:
            if file_suffix not in file_extractor:
                reader_cls = default_file_reader_cls[file_suffix]
                if isinstance(reader_cls, type):
                    file_extractor[file_suffix] = reader_cls()
                else:
                    file_extractor[file_suffix] = reader_cls
            reader = file_extractor[file_suffix]

            try:
                kwargs: dict[str, Any] = {"extra_info": metadata}

                if fs and not is_default_fs(fs):
                    kwargs["fs"] = fs

                docs = await reader.aload_data(input_file, **kwargs)

            except ImportError as e:
                raise ImportError(str(e))

            except Exception as e:
                if raise_on_error:
                    raise

                print(
                    f"Failed to load file {input_file} with error: {e}. Skipping...",
                    flush=True,
                )
                return []

            if filename_as_id:
                for i, doc in enumerate(docs):
                    doc.id_ = f"{input_file!s}_part_{i}"

            documents.extend(docs)

        else:
            active_fs = fs or get_default_fs()

            # Since encoding is passed, f.read() normally returns str in newer fsspec.
            with active_fs.open(input_file, errors=errors, encoding=encoding) as f:
                raw_data = f.read()

            if isinstance(raw_data, bytes):
                data = raw_data.decode(encoding, errors=errors)
            else:
                data = raw_data

            doc = Document(text=data, metadata=metadata or {})

            if filename_as_id:
                doc.id_ = str(input_file)

            documents.append(doc)

        return documents
    
    def metadata_to_str(self, doc: TextNode):
        metadata = doc.metadata or {}

        child_path = metadata.get("child_path")
        if isinstance(child_path, Path):
            metadata["child_path"] = str(child_path)

        rel = doc.relationships.get(NodeRelationship.SOURCE) if doc.relationships else None

        if rel is None:
            return

        # SOURCE sollte normalerweise ein einzelnes RelatedNodeInfo sein,
        # aber andere Beziehungen können Listen sein.
        if isinstance(rel, list):
            rels = rel
        else:
            rels = [rel]

        for r in rels:
            rel_metadata = getattr(r, "metadata", None)

            if not isinstance(rel_metadata, dict):
                continue

            child_path = rel_metadata.get("child_path")
            if isinstance(child_path, Path):
                rel_metadata["child_path"] = str(child_path)
            
    
        
    def load_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        documents = []
        cache_dir = os.path.join(self.input_dir,"_cache")

        files_to_process = self.input_files
        fs = fs or self.fs

        if num_workers and num_workers > 1 and not self.cached_documents:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(
                    ReaderFactory.load_file,
                    zip(
                        files_to_process,
                        repeat(self.file_metadata),
                        repeat(self.file_extractor),
                        repeat(self.filename_as_id),
                        repeat(self.encoding),
                        repeat(self.errors),
                        repeat(self.raise_on_error),
                        repeat(fs),
                    ),
                )
                documents = reduce(lambda x, y: x + y, results)
                for document in documents:
                    self.extract_start_dir(document=document, root_dir=self.input_dir)

        else:
            if show_progress:
                files_to_process = tqdm(
                    self.input_files, desc="Loading files", unit="file"
                )
            cache = FileCache(cache_dir=cache_dir, max_cache_size=2)
            cache.load_from_disk()
            documents = cache.get_documents()
            if not self.cached_documents: # read further document in addition to cache
                # need to explicitely check this because if this should be run on linux
                # some document type are not supported on linux, original parsing needs
                # to happen on windows
                for input_file in tqdm(files_to_process,"processing all files"):
                    #if ".xlsx" == input_file.suffix or ".xls" == input_file.suffix or ".xlsm" == input_file.suffix:
                    if not cache.is_file_in_cache(str(input_file)):
                        print("processing {}". format(input_file))
                        docs = self.load_file(
                                input_file=input_file,
                                file_metadata=self.file_metadata,
                                file_extractor=self.file_extractor,
                                filename_as_id=self.filename_as_id,
                                encoding=self.encoding,
                                errors=self.errors,
                                raise_on_error=self.raise_on_error,
                                fs=fs,
                            )
                        if len(docs) > 0:
                            cache.add(docs)
                            documents.extend(docs)
                # save the last few files to disk
                cache.save_to_disk()
                for document in documents:
                    self.extract_start_dir(document=document, root_dir=self.input_dir)
                    if isinstance(document, Document):
                        document = document_to_node(document)
                    self.metadata_to_str(document)
                    
        return self._exclude_metadata(documents)
    
    async def aload_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        documents = await super(ReaderFactory, self).aload_data(
            show_progress=show_progress,
            num_workers=num_workers,
            fs=fs,
        )
        
        for document in documents:
            self.extract_start_dir(document=document, root_dir=self.input_dir)
        
        return documents
        
        


