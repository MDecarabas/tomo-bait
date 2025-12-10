import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WorkflowRoute(str, Enum):
    """Available workflow routes"""

    STANDARD_QA = "standard_qa"
    BEAMLINE_EXPERT = "beamline_expert"
    CODE_GENERATION = "code_generation"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    ITERATIVE_REFINEMENT = "iterative_refinement"


class WorkflowRouter:
    """Routes questions to appropriate workflows based on analysis"""

    def __init__(self, orchestrator):
        """
        Initialize router.

        Args:
            orchestrator: AgentOrchestrator instance
        """
        self.orchestrator = orchestrator

        # Define routing rules
        self.routing_rules = {
            "beamline-specific": WorkflowRoute.BEAMLINE_EXPERT,
            "code-related": WorkflowRoute.CODE_GENERATION,
            "troubleshooting": WorkflowRoute.TROUBLESHOOTING,
            "comparison": WorkflowRoute.COMPARISON,
            "factual": WorkflowRoute.STANDARD_QA,
            "how-to": WorkflowRoute.STANDARD_QA,
        }

    def route(
        self, user_message: str, query_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine which workflow to use.

        Args:
            user_message: User's question
            query_analysis: Optional analysis from Query Analyzer

        Returns:
            Workflow name to execute
        """
        # If no analysis, use default
        if not query_analysis:
            logger.info("No query analysis available, using standard QA workflow")
            return WorkflowRoute.STANDARD_QA

        question_type = query_analysis.get("question_type", "factual")

        # Apply routing rules
        workflow = self.routing_rules.get(question_type, WorkflowRoute.STANDARD_QA)

        logger.info(f"Routing question type '{question_type}' to workflow '{workflow}'")

        return workflow

    async def route_and_execute(
        self, user_message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route to appropriate workflow and execute.

        Args:
            user_message: User's question
            context: Optional context

        Returns:
            Workflow execution results
        """
        # First, run query analyzer if available
        query_analysis = None
        # TODO: Execute query analyzer

        # Route to workflow
        workflow_name = self.route(user_message, query_analysis)

        # Execute workflow
        result = await self.orchestrator.execute_workflow(
            workflow_name=workflow_name,
            user_message=user_message,
            context=context or {},
        )

        # Add routing metadata
        result["workflow_used"] = workflow_name
        result["query_analysis"] = query_analysis

        return result
