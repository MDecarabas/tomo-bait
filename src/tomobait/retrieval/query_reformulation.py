import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Common acronyms and expansions in tomography/beamline domain
ACRONYM_MAP = {
    "2-BM": "2-BM tomography beamline",
    "APS": "Advanced Photon Source",
    "XRF": "X-ray fluorescence",
    "CT": "computed tomography",
    "HDF5": "Hierarchical Data Format 5",
    "TIFF": "Tagged Image File Format",
    # Add more as needed
}


class QueryReformulator:
    """Reformulates queries for better retrieval"""

    def __init__(self, acronym_map: Dict[str, str] = None):
        """
        Initialize query reformulator.

        Args:
            acronym_map: Custom acronym expansion map
        """
        self.acronym_map = acronym_map or ACRONYM_MAP

    def expand_acronyms(self, query: str) -> str:
        """Expand known acronyms in query"""
        expanded = query
        for acronym, expansion in self.acronym_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r"b" + re.escape(acronym) + r"b"
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
        return expanded

    def generate_variants(self, query: str) -> List[str]:
        """
        Generate query variants for better coverage.

        Returns:
            List of query variants including original
        """
        variants = [query]

        # Add acronym-expanded version
        expanded = self.expand_acronyms(query)
        if expanded != query:
            variants.append(expanded)

        # Add question reformulations
        if query.endswith("?"):
            # Convert question to statement
            statement = query.rstrip("?")
            variants.append(statement)

        # Add domain context
        if any(
            term in query.lower()
            for term in ["beamline", "tomography", "reconstruction"]
        ):
            # Already has domain context
            pass
        else:
            # Add beamline context
            variants.append(f"{query} tomography beamline")

        return variants

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {
            "what",
            "how",
            "when",
            "where",
            "why",
            "is",
            "are",
            "the",
            "a",
            "an",
        }

        # Tokenize and filter
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords
