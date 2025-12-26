"""Base search engine interface and result types"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    """Single search result"""
    doc_id: int
    url: str
    score: float
    content_snippet: str
    metadata: Optional[dict] = None

    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.3f}, url={self.url[:50]}...)"


class BaseSearchEngine(ABC):
    """
    Abstract base class for all search engines.

    All search implementations must inherit from this and implement:
    - index(): Build search index from documents
    - search(): Return ranked results for a query
    - get_name(): Return engine name for reporting
    """

    @abstractmethod
    def index(self, db_path: str):
        """
        Build search index from database.

        Args:
            db_path: Path to SQLite database with documents table
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for query and return ranked results.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return engine name for reporting.

        Returns:
            String identifier for this engine (e.g., "BM25", "Vector")
        """
        pass
