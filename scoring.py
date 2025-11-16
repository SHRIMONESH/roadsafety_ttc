"""
Enhanced Scoring Module - Centralized, Validated, Auditable
============================================================
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np # Added for numpy usage

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

@dataclass
class ScoringConfig:
    """Centralized scoring configuration with validated weights."""
    # Core component weights (must sum to 1.0)
    similarity_weight: float = 0.60
    feasibility_weight: float = 0.25
    evidence_weight: float = 0.15
    
    # KG bonus configuration
    kg_bonus_enabled: bool = True
    kg_bonus_max: float = 0.10
    kg_bonus_per_path: float = 0.02
    
    def __post_init__(self):
        """Validate that core weights sum to 1.0"""
        core_sum = self.similarity_weight + self.feasibility_weight + self.evidence_weight
        if abs(core_sum - 1.0) > 0.001:
            raise ValueError(
                f"Core weights must sum to 1.0, got {core_sum:.4f}. "
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary"""
        return asdict(self)

DEFAULT_CONFIG = ScoringConfig()


# =============================================================================
# INPUT VALIDATION & NORMALIZATION
# =============================================================================

def validate_and_clip(value: float, name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Validate and clip a score to the specified range."""
    original = value
    
    if value is None or not isinstance(value, (int, float)):
        logger.warning(f"[{name}] is invalid (type={type(value)}), defaulting to 0.0")
        return 0.0
    
    clipped = max(min_val, min(max_val, float(value)))
    
    if abs(clipped - original) > 0.0001:
        logger.warning(
            f"[{name}] out of range [{min_val}, {max_val}]: "
            f"original={original:.4f}, clipped={clipped:.4f}"
        )
    
    return clipped


def normalize_input(value: Any, name: str, expected_range: Tuple[float, float] = (0.0, 1.0)) -> float:
    """Normalize input to 0-1 range with intelligent detection."""
    # Handle dictionary inputs (e.g., {'overall_score': 0.8})
    if isinstance(value, dict):
        if 'overall_score' in value:
            value = value['overall_score']
        elif 'score' in value:
            value = value['score']
        else:
            logger.warning(f"[{name}] is dict without score key: {value}, defaulting to 0.0")
            return 0.0
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        logger.warning(f"[{name}] cannot be converted to float: {value}, defaulting to 0.0")
        return 0.0
    
    if value > 1.0:
        if value <= 100.0:
            normalized = value / 100.0
            return validate_and_clip(normalized, name)
        else:
            logger.warning(f"[{name}] exceeds 100: {value:.2f}, clipping to 1.0")
            return 1.0
    
    return validate_and_clip(value, name)


# =============================================================================
# CORE SCORING FUNCTIONS (compute_final_score_detailed is the core logic)
# =============================================================================

def compute_final_score(
    similarity: float,
    feasibility: float,
    evidence_strength: float,
    kg_paths: Optional[List[str]] = None,
    config: Optional[ScoringConfig] = None,
    debug: bool = False
) -> float:
    """Compute final intervention score (simple interface)."""
    result = compute_final_score_detailed(similarity, feasibility, evidence_strength, kg_paths, config)
    return result['final_score']


def compute_final_score_detailed(
    similarity: float,
    feasibility: float,
    evidence_strength: float,
    kg_paths: Optional[List[str]] = None,
    config: Optional[ScoringConfig] = None
) -> Dict[str, Any]:
    """Compute final intervention score with full breakdown."""
    config = config or DEFAULT_CONFIG
    
    # Validate and normalize inputs
    norm_similarity = normalize_input(similarity, "similarity")
    norm_feasibility = normalize_input(feasibility, "feasibility")
    norm_evidence = normalize_input(evidence_strength, "evidence_strength")
    
    if kg_paths is None or not isinstance(kg_paths, list):
        kg_paths = []
    
    kg_path_count = len(kg_paths)
    
    # Compute weighted components
    sim_contribution = config.similarity_weight * norm_similarity
    feas_contribution = config.feasibility_weight * norm_feasibility
    evi_contribution = config.evidence_weight * norm_evidence
    
    base_score = sim_contribution + feas_contribution + evi_contribution
    
    # Calculate KG bonus
    kg_bonus = 0.0
    if config.kg_bonus_enabled and kg_path_count > 0:
        kg_bonus = min(
            config.kg_bonus_max,
            config.kg_bonus_per_path * kg_path_count
        )
    
    # Final score
    final_score_raw = base_score + kg_bonus
    final_score = min(1.0, final_score_raw)
    
    # Human-readable formula (for auditing)
    formula = (
        f"{config.similarity_weight:.2f}*{norm_similarity:.3f} + "
        f"{config.feasibility_weight:.2f}*{norm_feasibility:.3f} + "
        f"{config.evidence_weight:.2f}*{norm_evidence:.3f}"
    )
    if kg_bonus > 0:
        formula += f" + {kg_bonus:.3f}(KG)"
    formula += f" = {final_score:.3f}"
    
    return {
        "final_score": final_score,
        "components": {
            "similarity": norm_similarity,
            "feasibility": norm_feasibility,
            "evidence_strength": norm_evidence,
            "kg_path_count": kg_path_count
        },
        "contributions": {
            "similarity_contribution": sim_contribution,
            "feasibility_contribution": feas_contribution,
            "evidence_contribution": evi_contribution,
            "base_score": base_score,
            "kg_bonus": kg_bonus,
            "final_score_raw": final_score_raw,
            "capped": final_score_raw > 1.0
        },
        "formula": formula,
        "metadata": {
            "config": config.to_dict()
        }
    }


def score_intervention_dict(
    intervention: Dict[str, Any], 
    site_summary: Dict[str, Any], # Added site_summary to allow feasibility engine access
    config: Optional[ScoringConfig] = None,
    in_place: bool = True
) -> Dict[str, Any]:
    """
    Score an intervention dictionary.
    
    NOTE: Feasibility is assumed to be computed externally by the feasibility_engine.
    """
    
    # Extract components with fallbacks
    similarity = intervention.get('relevance_score', 0.0) # Use relevance_score from RAGRunner
    
    # Feasibility is assumed to be pre-computed (or handled by the feasibility engine)
    feasibility = intervention.get('feasibility_score', 0.5) 
    
    # Evidence: Use evidence_score from KG enhancement, or derive from evidence_ids
    evidence = intervention.get('evidence_score') 
    if evidence is None:
        evidence_ids = intervention.get('evidence_ids', [])
        evidence = min(1.0, len(evidence_ids) * 0.1) if evidence_ids else 0.0
    
    kg_paths = intervention.get('kg_paths', [])
    
    # Compute detailed score
    result = compute_final_score_detailed(
        similarity=similarity,
        feasibility=feasibility,
        evidence_strength=evidence,
        kg_paths=kg_paths,
        config=config
    )
    
    # Update intervention
    target = intervention if in_place else intervention.copy()
    target['final_score'] = result['final_score']
    target['scoring_breakdown'] = result
    target['scoring_formula'] = result['formula']
    
    return target