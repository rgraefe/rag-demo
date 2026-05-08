import pickle
import os
import hashlib
from llama_index.core.schema import Document
import shutil
import logging
log = logging.getLogger(__name__)

class FileCache:
    """
    Cache for reading on files into Llamaindex documents. The cached elements are of type Document. They correspond to files read in from disk. 
    This is used because the loading of many files can take hours.
    The cache prevents starting from scratch if the document loading process was interrupted. The cache is stored to disk using pickle files.
    """
    def __init__(self, cache_dir: str, max_cache_size: int):
        """Init the cache.

        Args:
            cache_dir (str): Location for storing the cache to disk
            max_cache_size (int): After how many elements is cache stored to disk.
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_count = 0
        self.cache = {}
        self.cache_data = []

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def add(self, documents: list[Document]):
        """A list of Llama documents is added to the cache after a file was loaded from disk and processed with the respective file loader
        A file on disk usually leads to multiple llamaindex Documents. Therefore disk file and its checksum will be matched to a list of Documents.
        !! make sure that the Document list comes from the same file on disk !!

        Args:
            document (Document): a llama_index.core.schema.Document 

        Returns:
            Nothing
        """
        #check that the list of Documents only contains one file path
        filepaths = {item.metadata["file_path"] for item in documents}
        if len(filepaths) > 1:
            
            raise ValueError("the list of documents contains more than one filepath")
        metadata = documents[0].metadata
        filepath = metadata["file_path"]
        checksum = self._get_file_checksum(filepath)
        if filepath in self.cache and self.cache[filepath] == checksum:
            return  # Skip if file with same name and checksum is already cached

        self.cache[filepath] = checksum
        self.cache_data.append({"filepath": filepath, "checksum": checksum, "document": documents})
        self.cache_count += 1

        if self.cache_count >= self.max_cache_size:
            self.save_to_disk()
            # self.cache_data = []
            self.cache_count = 0

    def save_to_disk(self):
        """
        Safe the current content of the cache to disk. Called internally once max_cache_size is reached, e.g. after 10 documents were added.
        The old file on disk is overwritten but one copy is kept.
        """
        #overwrite existing
        
        cache_file = os.path.join(self.cache_dir, f"cache.pkl")
        if os.path.exists(cache_file):
            cache_0_file = os.path.join(self.cache_dir, f"cache_0.pkl")
            shutil.move(cache_file, cache_0_file)
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache_data, f)

    def load_from_disk(self):
        """
        Load the complete cache from disk and rebuilds a dictionary of filepath and checksum values.
        """
        cache_file = os.path.join(self.cache_dir, f"cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.cache_data = pickle.load(f)
                self._rebuild_cache()
                
    def get_documents(self) -> list[Document]:
        """
        Gets the current cache content as a list of Document.

        Returns:
            list[Document]: a list of llama_index.core.schema.Document
        """
        documents = []
        for item in self.cache_data:
            documents.extend(item["document"])
        return documents
                
    def is_file_in_cache(self, filepath:str):
        """Check if the data for a certain filepath and a particular version is already in the cache.
           A checksum is used to compare files with similar names.

        Args:
            filepath (str): absolute filepath of file

        Returns:
            bool: True if in cache, False otherwise
        """
        filepath = str(filepath) # in case we gave a Windows Filepath instead a string
        checksum = self._get_file_checksum(filepath)
        if filepath in self.cache and self.cache[filepath] == checksum:
            return True
        else:
            return False

    def _rebuild_cache(self):
        
        self.cache = {entry["filepath"]: entry["checksum"] for entry in self.cache_data}
        log.debug("Rebuilt cache entries for {} files".format(len(self.cache)))

    def _get_file_checksum(self, filename):
        hasher = hashlib.sha256()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()