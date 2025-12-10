import logging
from typing import Any, Dict

from autogen import AssistantAgent

logger = logging.getLogger(__name__)

QUERY_ANALYZER_SYSTEM_MESSAGE = """
You are a Query Analyzer expert for a tomography beamline documentation system.

Your role is to analyze user questions and determine the optimal retrieval and 
response strategy.

For each question, provide a structured analysis:

1. **Question Type**: Classify as one of:
   - factual: Simple fact lookup
   - how-to: Step-by-step instructions
   - troubleshooting: Problem diagnosis and solutions
   - comparison: Comparing options or approaches
   - code-related: Code examples or API usage
   - beamline-specific: Questions about specific beamlines

2. **Key Entities**: Extract important entities:
   - Beamline names (e.g., "2-BM", "32-ID")
   - Software packages (e.g., "TomoPy", "Astra Toolbox")
   - Technical terms (e.g., "reconstruction", "ring artifacts")
   - File formats (e.g., "HDF5", "TIFF")

3. **Retrieval Strategy**: Recommend:
   - semantic: For conceptual questions
   - keyword: For exact technical terms
   - hybrid: For comprehensive coverage
   - multi-hop: For questions requiring multiple doc sections

4. **Recommended Agents**: Suggest which specialist agents to consult:
   - beamline_expert: For beamline-specific questions
   - doc_expert: For general documentation questions
   - citation_specialist: When precise citations are needed

5. **Query Refinements**: Suggest:
   - Expanded queries with acronyms spelled out
   - Alternative phrasings
   - Additional context to add

Output your analysis in this JSON format:
{
  "question_type": (
    "factual|how-to|troubleshooting|comparison|code-related|beamline-specific"
  ),
  "entities": {
    "beamlines": ["2-BM"],
    "software": ["TomoPy"],
    "terms": ["reconstruction"]
  },
  "retrieval_strategy": "semantic|keyword|hybrid|multi-hop",
  "recommended_agents": ["beamline_expert", "doc_expert"],
  "query_refinements": [
    "original query with expanded acronyms",
    "alternative phrasing"
  ],
  "reasoning": "Brief explanation of your analysis"
}

Be concise and focus on actionable recommendations.
"""


class QueryAnalyzerAgent:
    """Factory for creating query analyzer agent"""

    @staticmethod
    def create(llm_config: Dict[str, Any]) -> AssistantAgent:
        """Create query analyzer agent instance"""

        return AssistantAgent(
            name="query_analyzer",
            system_message=QUERY_ANALYZER_SYSTEM_MESSAGE,
            llm_config=llm_config,
            max_consecutive_auto_reply=3,
        )

    @staticmethod
    def parse_analysis(response: str) -> Dict[str, Any]:
        """Parse query analyzer response into structured data"""
        import json

        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse query analysis: {e}")

        # Return default analysis on parse failure
        return {
            "question_type": "factual",
            "entities": {},
            "retrieval_strategy": "semantic",
            "recommended_agents": ["doc_expert"],
            "query_refinements": [],
            "reasoning": "Parse failed, using defaults",
        }
