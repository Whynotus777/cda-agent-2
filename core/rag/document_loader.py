"""
Document Loader Module

Load, parse, and chunk various document types for RAG indexing.
Handles markdown, text, PDF, HTML, and code files.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of a document with metadata"""

    def __init__(
        self,
        content: str,
        metadata: Dict,
        chunk_id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }


class DocumentLoader:
    """
    Load and process documents for RAG indexing.

    Supports:
    - Markdown (.md)
    - Text (.txt)
    - PDF (.pdf)
    - HTML (.html)
    - Code files (.v, .sv, .py, .c, .cpp)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document loader.

        Args:
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Load a single file and return document chunks.

        Args:
            file_path: Path to file

        Returns:
            List of DocumentChunk objects
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        # Determine file type and load appropriately
        suffix = path.suffix.lower()

        try:
            if suffix in ['.md', '.txt']:
                return self._load_text_file(path)
            elif suffix == '.pdf':
                return self._load_pdf(path)
            elif suffix in ['.html', '.htm']:
                return self._load_html(path)
            elif suffix in ['.v', '.sv', '.vhdl', '.py', '.c', '.cpp', '.h', '.hpp']:
                return self._load_code_file(path)
            else:
                # Try as text
                return self._load_text_file(path)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def load_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[DocumentChunk]:
        """
        Load all files from a directory.

        Args:
            directory: Directory path
            extensions: List of file extensions to include (e.g., ['.md', '.txt'])
            recursive: Whether to search subdirectories

        Returns:
            List of all document chunks
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        # Default extensions
        if extensions is None:
            extensions = ['.md', '.txt', '.pdf', '.html', '.v', '.sv', '.py']

        # Find files
        all_chunks = []
        pattern = '**/*' if recursive else '*'

        for ext in extensions:
            files = list(dir_path.glob(f"{pattern}{ext}"))
            logger.info(f"Found {len(files)} {ext} files in {directory}")

            for file_path in files:
                chunks = self.load_file(str(file_path))
                all_chunks.extend(chunks)

        logger.info(f"Loaded {len(all_chunks)} total chunks from {directory}")
        return all_chunks

    def _load_text_file(self, path: Path) -> List[DocumentChunk]:
        """Load plain text or markdown file"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        metadata = {
            'source': str(path),
            'type': 'text',
            'filename': path.name
        }

        return self._chunk_text(content, metadata)

    def _load_pdf(self, path: Path) -> List[DocumentChunk]:
        """Load PDF file"""
        try:
            import PyPDF2

            text = ""
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"

            metadata = {
                'source': str(path),
                'type': 'pdf',
                'filename': path.name
            }

            return self._chunk_text(text, metadata)

        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []

    def _load_html(self, path: Path) -> List[DocumentChunk]:
        """Load HTML file"""
        try:
            from bs4 import BeautifulSoup

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()

            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            metadata = {
                'source': str(path),
                'type': 'html',
                'filename': path.name
            }

            return self._chunk_text(text, metadata)

        except ImportError:
            logger.warning("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            # Fall back to plain text
            return self._load_text_file(path)

    def _load_code_file(self, path: Path) -> List[DocumentChunk]:
        """Load code file with syntax awareness"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        metadata = {
            'source': str(path),
            'type': 'code',
            'language': path.suffix[1:],  # Remove dot
            'filename': path.name
        }

        # For code, use smaller chunks to preserve logical units
        return self._chunk_text(content, metadata, chunk_size=500)

    def _chunk_text(
        self,
        text: str,
        metadata: Dict,
        chunk_size: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Metadata for all chunks
            chunk_size: Override default chunk size

        Returns:
            List of DocumentChunk objects
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Clean text
        text = text.strip()

        if len(text) <= chunk_size:
            return [DocumentChunk(text, metadata.copy())]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(chunk_size * 0.2)
                sentence_break = text.rfind('.', search_start, end)

                if sentence_break > start:
                    end = sentence_break + 1
                else:
                    # Try newline
                    newline_break = text.rfind('\n', search_start, end)
                    if newline_break > start:
                        end = newline_break + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = len(chunks)
                chunk_metadata['start_char'] = start
                chunk_metadata['end_char'] = end

                chunks.append(DocumentChunk(chunk_text, chunk_metadata))

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        logger.debug(f"Split {metadata.get('source', 'text')} into {len(chunks)} chunks")
        return chunks

    def load_web_content(self, url: str) -> List[DocumentChunk]:
        """
        Load content from a URL.

        Args:
            url: Web URL to fetch

        Returns:
            List of document chunks
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            logger.info(f"Fetching {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            metadata = {
                'source': url,
                'type': 'web',
                'url': url
            }

            return self._chunk_text(text, metadata)

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return []
