"""Ground truth dataset loader"""

import json
from pathlib import Path
from typing import Optional, Set, Dict
from dataclasses import dataclass


@dataclass
class GroundTruthQuery:
    """Single ground truth query"""
    id: int
    query: str
    description: str
    relevant_docs: Set[int]
    highly_relevant: Set[int]


class GroundTruth:
    """
    Manages ground truth dataset for evaluation.

    Ground truth defines which documents are correct for each query.
    """

    def __init__(self, json_path: str):
        """
        Load ground truth from JSON file.

        Args:
            json_path: Path to ground_truth.json
        """
        self.json_path = json_path
        self.queries: Dict[str, GroundTruthQuery] = {}
        self._load()

    def _load(self):
        """Load ground truth from JSON"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for q in data.get('queries', []):
                gt_query = GroundTruthQuery(
                    id=q['id'],
                    query=q['query'],
                    description=q.get('description', ''),
                    relevant_docs=set(q.get('relevant_docs', [])),
                    highly_relevant=set(q.get('highly_relevant', []))
                )
                self.queries[q['query']] = gt_query

        except FileNotFoundError:
            # No ground truth file, queries dict stays empty
            pass
        except json.JSONDecodeError as e:
            print(f"Warning: Error loading ground truth: {e}")

    def has_query(self, query: str) -> bool:
        """
        Check if query has ground truth.

        Args:
            query: Search query string

        Returns:
            True if ground truth exists for this query
        """
        return query in self.queries

    def get(self, query: str) -> Optional[GroundTruthQuery]:
        """
        Get ground truth for a query.

        Args:
            query: Search query string

        Returns:
            GroundTruthQuery if exists, None otherwise
        """
        return self.queries.get(query)

    def get_relevant_docs(self, query: str) -> Set[int]:
        """
        Get relevant document IDs for a query.

        Args:
            query: Search query string

        Returns:
            Set of relevant document IDs (empty if no ground truth)
        """
        gt = self.get(query)
        return gt.relevant_docs if gt else set()

    def list_queries(self) -> list:
        """
        List all queries with ground truth.

        Returns:
            List of query strings
        """
        return list(self.queries.keys())

    def count(self) -> int:
        """
        Count queries with ground truth.

        Returns:
            Number of queries
        """
        return len(self.queries)
