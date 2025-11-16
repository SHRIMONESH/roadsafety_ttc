"""
Enhanced Prompt Builder for Road Safety LLM
============================================

Provides advanced prompt construction with:
- Structured schema definitions
- Evidence-based reasoning prompts
- JSON validation utilities
- Error handling and fallbacks

Usage:
    from llm.enhanced_prompt import build_analysis_prompt, validate_llm_response
    
    prompt = build_analysis_prompt(site_summary, top_candidates)
    result = validate_llm_response(llm_output)

Author: System
Version: 1.0
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMA DEFINITIONS
# =============================================================================

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "required": ["recommendations"],
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "intervention_id",
                    "intervention_name",
                    "rank",
                    "priority",
                    "expected_impact",
                    "final_score",
                    "reason"
                ],
                "properties": {
                    "intervention_id": {"type": "string"},
                    "intervention_name": {"type": "string"},
                    "rank": {"type": "integer", "minimum": 1},
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "expected_impact": {"type": "string"},
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                    "final_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "feasibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "kg_paths": {"type": "array"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        },
        "analysis": {"type": "string"},
        "additional_notes": {"type": "string"}
    }
}


# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def build_analysis_prompt(
    site_summary: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
    max_candidates: int = 5,
    include_schema: bool = True,
    verbose: bool = False
) -> str:
    """
    Build enhanced prompt for LLM analysis with structured guidance.
    
    Args:
        site_summary: Dictionary containing site information
        top_candidates: List of intervention candidates from RAG
        max_candidates: Maximum number of candidates to analyze
        include_schema: Include detailed JSON schema in prompt
        verbose: Include additional reasoning instructions
    
    Returns:
        Formatted prompt string
    """
    
    # Select top candidates
    candidates_to_analyze = top_candidates[:max_candidates]
    
    # Build prompt sections
    prompt_parts = []
    
    # 1. System Role
    prompt_parts.append(_build_system_role())
    
    # 2. Task Description
    prompt_parts.append(_build_task_description())
    
    # 3. Site Context
    prompt_parts.append(_build_site_context(site_summary))
    
    # 4. Candidates
    prompt_parts.append(_build_candidates_section(candidates_to_analyze))
    
    # 5. Analysis Instructions
    if verbose:
        prompt_parts.append(_build_analysis_instructions())
    
    # 6. Output Schema
    if include_schema:
        prompt_parts.append(_build_schema_section())
    else:
        prompt_parts.append(_build_basic_schema())
    
    # 7. Final Instructions
    prompt_parts.append(_build_final_instructions())
    
    # Combine all sections
    full_prompt = "\n\n".join(prompt_parts)
    
    if verbose:
        logger.info(f"Built enhanced prompt: {len(full_prompt)} characters")
    
    return full_prompt


def _build_system_role() -> str:
    """Build system role description."""
    return """You are an expert road safety analyst with deep knowledge of:
- Traffic engineering and road infrastructure
- Evidence-based intervention effectiveness
- Implementation feasibility analysis
- Cost-benefit evaluation for safety measures

Your task is to provide data-driven, actionable recommendations for road safety improvements."""


def _build_task_description() -> str:
    """Build task description."""
    return """## TASK
Analyze the provided road safety site data and intervention candidates.
Rank and recommend interventions based on:
1. Relevance to identified problems
2. Evidence-based effectiveness
3. Implementation feasibility
4. Expected impact on safety outcomes

Your analysis must be objective, evidence-based, and practical."""


def _build_site_context(site_summary: Dict[str, Any]) -> str:
    """Build site context section."""
    return f"""## SITE INFORMATION
The following describes the current road safety situation:

```json
{json.dumps(site_summary, indent=2)}
```

**Key Issues Identified:**
- Location: {site_summary.get('location', 'Not specified')}
- Problems: {', '.join(site_summary.get('problems', ['None listed']))}
- Road Type: {site_summary.get('road_type', 'Not specified')}
- Crash Types: {', '.join(site_summary.get('crash_types', ['None listed']))}
- Traffic Volume: {site_summary.get('traffic_volume', 'Not specified')}
- Severity: {site_summary.get('severity', 'Not specified')}"""


def _build_candidates_section(candidates: List[Dict[str, Any]]) -> str:
    """Build intervention candidates section."""
    
    # Simplify candidates to reduce token count
    simplified_candidates = []
    for c in candidates:
        simplified = {
            "intervention_id": c.get("intervention_id"),
            "intervention_name": c.get("intervention_name"),
            "description": c.get("intervention_description", "")[:200],  # Truncate
            "relevance_score": c.get("relevance_score", 0),
            "evidence_score": c.get("evidence_score", 0),
            "final_score": c.get("final_score", 0),
            "technical_complexity": c.get("technical_complexity", "Unknown"),
            "estimated_cost_level": c.get("estimated_cost_level", "Unknown"),
            "time_to_implement": c.get("time_to_implement", "Unknown"),
            "target_crash_types": c.get("target_crash_types", []),
            "applicable_problems": c.get("applicable_problems", [])
        }
        simplified_candidates.append(simplified)
    
    return f"""## INTERVENTION CANDIDATES
The following {len(candidates)} interventions have been pre-screened by the RAG system:

```json
{json.dumps(simplified_candidates, indent=2)}
```

These candidates have already been filtered for relevance. Your task is to:
1. Analyze each candidate's fit for this specific site
2. Rank them by expected effectiveness
3. Provide clear reasoning for your recommendations"""


def _build_analysis_instructions() -> str:
    """Build detailed analysis instructions."""
    return """## ANALYSIS GUIDELINES

When evaluating each intervention, consider:

**Relevance Assessment:**
- Does it directly address the identified problems?
- Is it appropriate for the road type and traffic conditions?
- Does it target the relevant crash types?

**Evidence & Effectiveness:**
- What is the evidence quality score?
- How strong is the knowledge graph support?
- What are the expected safety outcomes?

**Feasibility Analysis:**
- Technical complexity level
- Implementation timeframe
- Estimated cost range
- Disruption to traffic/community

**Priority Classification:**
- High: Urgent, high-impact, highly feasible
- Medium: Important but may have implementation challenges
- Low: Beneficial but lower priority or significant constraints

Provide specific, actionable reasoning for each recommendation."""


def _build_schema_section() -> str:
    """Build detailed output schema section."""
    return """## REQUIRED OUTPUT FORMAT

You MUST return a valid JSON object with this exact structure:

```json
{
  "recommendations": [
    {
      "intervention_id": "string (must match candidate ID)",
      "intervention_name": "string (must match candidate name)",
      "rank": integer (1 = highest priority, 2 = second, etc.),
      "priority": "High" | "Medium" | "Low",
      "expected_impact": "string (detailed description of expected safety improvements)",
      "supporting_evidence": ["evidence_id_1", "evidence_id_2"],
      "final_score": number (0.0 to 1.0, your confidence in this recommendation),
      "feasibility_score": number (0.0 to 1.0, implementation feasibility),
      "kg_paths": [["path_node_1", "path_node_2"]],
      "reason": "string (clear, specific explanation for why this intervention is recommended)",
      "confidence": number (0.0 to 1.0, your confidence level)
    }
  ],
  "analysis": "string (overall site analysis and summary)",
  "additional_notes": "string (important warnings, considerations, or context)"
}
```

**Field Requirements:**
- `rank`: Start at 1 (top priority), increment for each recommendation
- `priority`: Must be exactly "High", "Medium", or "Low"
- `expected_impact`: Be specific about safety outcomes (e.g., "Expected to reduce rear-end collisions by 30%")
- `reason`: Explain WHY this intervention fits this specific site
- `confidence`: Your assessment of recommendation certainty (0.0 = uncertain, 1.0 = very confident)
- `analysis`: Summarize the overall site safety situation and your recommendation strategy
- `additional_notes`: Mention any important caveats, prerequisites, or implementation considerations"""


def _build_basic_schema() -> str:
    """Build simplified schema."""
    return """## OUTPUT FORMAT

Return a JSON object with:
- `recommendations`: Array of intervention recommendations
- `analysis`: Overall site analysis summary
- `additional_notes`: Important considerations

Each recommendation must include: intervention_id, intervention_name, rank, priority, expected_impact, final_score, reason, and confidence."""


def _build_final_instructions() -> str:
    """Build final output instructions."""
    return """## CRITICAL INSTRUCTIONS

1. **Return ONLY valid JSON** - No markdown code fences, no explanations outside the JSON
2. **Use exact IDs and names** from the candidate list
3. **Be specific and actionable** in your reasoning
4. **Rank by actual effectiveness** for this site, not just by pre-computed scores
5. **Include 3-5 recommendations** (top candidates only)

Begin your response with the opening brace `{` of the JSON object."""


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_llm_response(response: str) -> Dict[str, Any]:
    """
    Validate and parse LLM response.
    
    Args:
        response: Raw text response from LLM
    
    Returns:
        Validated JSON object
    
    Raises:
        ValueError: If response is invalid
    """
    
    # Clean response
    cleaned = _clean_response(response)
    
    # Parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    # Validate structure
    _validate_structure(data)
    
    # Validate recommendations
    if "recommendations" in data:
        for i, rec in enumerate(data["recommendations"]):
            _validate_recommendation(rec, i)
    
    logger.info(f"✅ Response validated: {len(data.get('recommendations', []))} recommendations")
    
    return data


def _clean_response(response: str) -> str:
    """Clean and normalize LLM response."""
    
    # Strip whitespace
    cleaned = response.strip()
    
    # Remove markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    # Strip again
    cleaned = cleaned.strip()
    
    # Find JSON boundaries if text before/after
    if not cleaned.startswith("{"):
        # Try to find the first {
        start_idx = cleaned.find("{")
        if start_idx != -1:
            cleaned = cleaned[start_idx:]
    
    if not cleaned.endswith("}"):
        # Try to find the last }
        end_idx = cleaned.rfind("}")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]
    
    return cleaned


def _validate_structure(data: Dict[str, Any]) -> None:
    """Validate top-level structure."""
    
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
    if "recommendations" not in data:
        raise ValueError("Response missing 'recommendations' field")
    
    if not isinstance(data["recommendations"], list):
        raise ValueError("'recommendations' must be an array")
    
    if len(data["recommendations"]) == 0:
        raise ValueError("'recommendations' array is empty")


def _validate_recommendation(rec: Dict[str, Any], index: int) -> None:
    """Validate individual recommendation."""
    
    required_fields = [
        "intervention_id",
        "intervention_name",
        "rank",
        "priority",
        "expected_impact",
        "final_score",
        "reason"
    ]
    
    for field in required_fields:
        if field not in rec:
            raise ValueError(f"Recommendation {index} missing required field: {field}")
    
    # Validate types
    if not isinstance(rec["rank"], int):
        raise ValueError(f"Recommendation {index}: 'rank' must be integer")
    
    if rec["rank"] < 1:
        raise ValueError(f"Recommendation {index}: 'rank' must be >= 1")
    
    if not isinstance(rec["final_score"], (int, float)):
        raise ValueError(f"Recommendation {index}: 'final_score' must be number")
    
    if not (0 <= rec["final_score"] <= 1):
        raise ValueError(f"Recommendation {index}: 'final_score' must be 0-1")
    
    if rec["priority"] not in ["High", "Medium", "Low"]:
        raise ValueError(f"Recommendation {index}: 'priority' must be High/Medium/Low")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def truncate_for_token_limit(text: str, max_chars: int = 10000) -> str:
    """
    Truncate text to approximate token limit.
    
    Args:
        text: Input text
        max_chars: Maximum characters (rough token approximation)
    
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    # Truncate and add indicator
    return text[:max_chars] + "\n\n[... truncated for length ...]"


def extract_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract essential candidate information for compact prompts.
    
    Args:
        candidate: Full candidate dictionary
    
    Returns:
        Simplified candidate summary
    """
    return {
        "id": candidate.get("intervention_id"),
        "name": candidate.get("intervention_name"),
        "description": candidate.get("intervention_description", "")[:150],
        "score": round(candidate.get("final_score", 0), 3),
        "feasibility": candidate.get("technical_complexity", "Unknown"),
        "cost": candidate.get("estimated_cost_level", "Unknown"),
        "timeframe": candidate.get("time_to_implement", "Unknown")
    }