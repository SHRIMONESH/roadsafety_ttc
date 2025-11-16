"""
Core RAG Pipeline for Road Safety Interventions
===========================================================
UPDATED: Groq API Integration (NOT xAI/Grok)
Version: 3.1 - Migrated from Gemini to Groq
Last Updated: 2024
"""

import os
import logging
import json
import re
import numpy as np
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core dependencies
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. KG features disabled.")

# Groq API (NOT OpenAI SDK)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq SDK not available. Install with: pip install groq")

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Relative imports
try:
    from .RAGRunner import RAGRunner
    from .scoring import score_intervention_dict 
    from .feasibility_engine import FeasibilityEngine 
except ImportError as e:
    logger.error(f"FATAL: Cannot import core components: {e}")
    raise ImportError(f"Cannot import RAG components (rag_runner/scoring/feasibility_engine).")


# =============================================================================
# EXPONENTIAL BACKOFF UTILITY
# =============================================================================

class ExponentialBackoff:
    """Handles exponential backoff for API rate limiting."""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on rate limit errors."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = any([
                    '429' in error_str,
                    'rate limit' in error_str,
                    'quota' in error_str,
                    'resource exhausted' in error_str,
                    'too many requests' in error_str
                ])
                
                if is_rate_limit:
                    if attempt < self.max_retries - 1:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        jitter = delay * 0.25 * np.random.random()
                        total_delay = delay + jitter
                        
                        logger.warning(f"⏳ Rate limit hit. Retrying in {total_delay:.1f}s")
                        logger.warning(f"   Attempt {attempt + 1}/{self.max_retries}")
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"❌ Max retries ({self.max_retries}) reached.")
                        raise Exception("API quota exceeded after maximum retries.")
                else:
                    # Non-rate-limit error, raise immediately
                    logger.error(f"❌ Non-rate-limit error: {e}")
                    raise
        
        raise Exception("Failed after maximum retry attempts")


# =============================================================================
# GROQ API SERVICE
# =============================================================================

class GroqService:
    """Handles Groq API calls using native Groq SDK."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq service.
        
        Args:
            api_key: Groq API key (or reads from GROQ_API_KEY/XAI_API_KEY env var)
            model_name: Groq model to use (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
        self.model_name = model_name
        self.backoff = ExponentialBackoff(max_retries=5, base_delay=2.0, max_delay=60.0)
        self.client: Optional[Groq] = None
        self.request_count = 0
        self.last_request_time = 0.0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        if not GROQ_AVAILABLE:
            logger.warning("⚠️ Groq SDK not available. Install with: pip install groq")
            return
        
        if self.api_key:
            try:
                # Initialize Groq client
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✅ Groq API initialized with model: {self.model_name}")
                logger.info(f"🔑 API Key: {self.api_key[:10]}...{self.api_key[-4:]}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq API: {e}")
        else:
            logger.warning("⚠️ No GROQ_API_KEY found. Groq enhancement disabled.")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce minimum time between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s before next request")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        logger.info(f"📊 Request #{self.request_count} to Groq API")
    
    def enhance_recommendations(
        self,
        site_summary: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance RAG candidates with Groq-generated insights using structured output."""
        
        if not self.client:
            logger.warning("⚠️ Groq not available. Returning candidates as-is.")
            return {
                "enhanced": False,
                "candidates": candidates,
                "error": "Groq API not configured"
            }
        
        try:
            # Enforce rate limiting before request
            self._enforce_rate_limit()
            
            # Build prompt for Groq
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_enhancement_prompt(site_summary, candidates)
            
            # Execute with exponential backoff
            def _call_groq() -> str:
                logger.info(f"🤖 Calling Groq API ({self.model_name})...")
                
                assert self.client is not None, "Client should be initialized"
                
                # Use Groq API with JSON mode
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                    response_format={"type": "json_object"}  # Request JSON response
                )
                
                return response.choices[0].message.content or ""
            
            logger.info("🔄 Executing with exponential backoff protection...")
            enhanced_text = self.backoff.execute_with_backoff(_call_groq)
            
            # Parse Groq response
            enhanced_data = self._parse_groq_response(enhanced_text, candidates)
            
            logger.info("✅ Groq enhancement complete")
            return {
                "enhanced": True,
                "candidates": enhanced_data.get("candidates", candidates),
                "summary": enhanced_data.get("summary", ""),
                "groq_raw_response": enhanced_text,
                "request_count": self.request_count
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Groq enhancement failed: {error_msg}")
            
            return {
                "enhanced": False,
                "candidates": candidates,
                "error": error_msg
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Groq."""
        return """You are an expert road safety engineer specializing in traffic safety interventions. Your role is to analyze road safety problems and recommend evidence-based interventions.

When analyzing interventions:
- Consider site-specific factors (location, traffic, road type)
- Prioritize interventions with proven effectiveness
- Account for practical implementation constraints
- Provide clear, actionable recommendations

Respond in valid JSON format with detailed analysis."""
    
    def _build_enhancement_prompt(
        self,
        site_summary: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Build enhancement prompt for Groq."""
        
        prompt = f"""Analyze these road safety intervention recommendations for a specific site.

SITE INFORMATION:
- Location: {site_summary.get('location', 'Unknown')}
- Description: {site_summary.get('description', 'N/A')}
- Problems: {', '.join(site_summary.get('problems', []))}
- Road Type: {site_summary.get('road_type', 'Unknown')}
- Traffic Volume: {site_summary.get('traffic_volume', 'Unknown')}
- Speed Limit: {site_summary.get('speed_limit', 'Unknown')}
- Crash Types: {', '.join(site_summary.get('crash_types', []))}

TOP INTERVENTION CANDIDATES:
"""
        
        for i, candidate in enumerate(candidates[:5], 1):
            prompt += f"""
{i}. {candidate.get('intervention_name', 'Unknown')}
   - Description: {candidate.get('intervention_description', 'N/A')}
   - Priority: {candidate.get('priority', 'MEDIUM')}
   - Final Score: {candidate.get('final_score', 0.0):.2f}
   - Feasibility: {candidate.get('feasibility_score', 0.0):.2f}
   - Technical Complexity: {candidate.get('technical_complexity', 'Unknown')}
   - Time to Implement: {candidate.get('time_to_implement', 'Unknown')}
   - Estimated Cost: {candidate.get('estimated_cost_level', 'Unknown')}
"""
        
        prompt += """
TASK:
For each intervention, provide:
1. Why it's recommended for this specific site (rationale)
2. Site-specific implementation considerations
3. Expected safety impact on the identified problems
4. Priority justification based on urgency and effectiveness

Respond ONLY with valid JSON in this exact structure:
{
  "summary": "Overall analysis of the site and recommendations (2-3 sentences)",
  "interventions": [
    {
      "intervention_id": "intervention ID from input",
      "rationale": "Why this intervention suits this site (2-3 sentences)",
      "implementation_notes": "Specific considerations for this location (2-3 sentences)",
      "expected_impact": "Anticipated safety outcomes (2-3 sentences)",
      "priority_justification": "Why this priority level is appropriate (1-2 sentences)"
    }
  ]
}

Ensure valid JSON - no markdown, no extra text, just the JSON object."""
        
        return prompt
    
    def _parse_groq_response(
        self,
        response_text: str,
        original_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Groq's JSON response and merge with original candidates."""
        
        try:
            # Clean response - remove markdown code blocks if present
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
            
            # Remove any leading/trailing whitespace and parse
            json_str = json_str.strip()
            parsed = json.loads(json_str)
            
            # Merge Groq insights with original candidates
            enhanced_candidates = []
            for candidate in original_candidates:
                candidate_id = candidate.get('intervention_id')
                
                # Find matching Groq insight
                groq_insight = None
                for insight in parsed.get('interventions', []):
                    if insight.get('intervention_id') == candidate_id:
                        groq_insight = insight
                        break
                
                if groq_insight:
                    candidate['groq_enhancement'] = {
                        'rationale': groq_insight.get('rationale', ''),
                        'implementation_notes': groq_insight.get('implementation_notes', ''),
                        'expected_impact': groq_insight.get('expected_impact', ''),
                        'priority_justification': groq_insight.get('priority_justification', '')
                    }
                
                enhanced_candidates.append(candidate)
            
            return {
                "summary": parsed.get('summary', ''),
                "candidates": enhanced_candidates
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse Groq JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return {
                "summary": response_text[:500],  # Fallback to raw text
                "candidates": original_candidates
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to process Groq response: {e}")
            return {
                "summary": response_text[:500],
                "candidates": original_candidates
            }


# =============================================================================
# FEASIBILITY ANALYZER & KG SERVICE
# =============================================================================

class FeasibilityAnalyzer:
    COMPLEXITY_SCORES = {
        "Very Low": 1.0, "Low": 0.85, "Medium": 0.65, "High": 0.45, "Very High": 0.25
    }
    TIMEFRAME_SCORES = {
        "Immediate (< 1 week)": 1.0, "Short (1-4 weeks)": 0.85, "Medium (1-3 months)": 0.65,
        "Long (3-6 months)": 0.45, "Very Long (> 6 months)": 0.25
    }
    COST_SCORES = {
        "Very Low": 1.0, "Low": 0.85, "Medium": 0.65, "High": 0.45, "Very High": 0.25
    }
    DISRUPTION_SCORES = {
        "None": 1.0, "Minimal": 0.90, "Low": 0.75, "Medium": 0.55,
        "High": 0.35, "Very High": 0.15
    }

    @staticmethod
    def analyze_feasibility(intervention: Dict[str, Any]) -> Dict[str, Any]:
        complexity = intervention.get("technical_complexity", "Medium")
        timeframe = intervention.get("time_to_implement", "Medium (1-3 months)")
        cost = intervention.get("estimated_cost_level", "Medium")
        disruption = intervention.get("implementation_disruption", "Medium")

        complexity_score = FeasibilityAnalyzer.COMPLEXITY_SCORES.get(complexity, 0.5)
        timeframe_score = FeasibilityAnalyzer.TIMEFRAME_SCORES.get(timeframe, 0.5)
        cost_score = FeasibilityAnalyzer.COST_SCORES.get(cost, 0.5)
        disruption_score = FeasibilityAnalyzer.DISRUPTION_SCORES.get(disruption, 0.5)

        overall_score = (complexity_score * 0.35 + cost_score * 0.30 +
                         timeframe_score * 0.20 + disruption_score * 0.15)

        if overall_score >= 0.85:
            level = "Very High"
        elif overall_score >= 0.70:
            level = "High"
        elif overall_score >= 0.55:
            level = "Medium"
        elif overall_score >= 0.40:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "overall_score": round(overall_score, 3), "feasibility_level": level,
            "components": {
                "technical_complexity": {"value": complexity, "score": complexity_score},
                "implementation_timeframe": {"value": timeframe, "score": timeframe_score},
                "estimated_cost": {"value": cost, "score": cost_score},
                "implementation_disruption": {"value": disruption, "score": disruption_score}
            },
            "weights": {
                "technical_complexity": 0.35, "estimated_cost": 0.30,
                "implementation_timeframe": 0.20, "implementation_disruption": 0.15
            }
        }


class KnowledgeGraphService:
    def __init__(self, kg_path: Optional[str] = None):
        self.kg: Optional[Any] = None
        self.kg_loaded = False
        if kg_path and NETWORKX_AVAILABLE:
            try:
                self.load_kg(kg_path)
            except Exception as e:
                logger.warning(f"Could not load KG from {kg_path}: {e}")

    def load_kg(self, kg_path: str) -> None:
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available. Cannot load KG.")
            return
        try:
            logger.info(f"Loading knowledge graph from: {kg_path}")
            self.kg = nx.read_gml(kg_path)
            self.kg_loaded = True
            logger.info(f"✅ KG loaded: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to load KG: {e}")
            self.kg_loaded = False

    def find_paths(self, problem_id: str, intervention_id: str, max_paths: int = 3, max_length: int = 4) -> List[List[str]]:
        if not self.kg_loaded or self.kg is None:
            return []
        try:
            if problem_id not in self.kg.nodes() or intervention_id not in self.kg.nodes():
                return []
            paths = []
            for path in nx.all_simple_paths(self.kg, source=problem_id, target=intervention_id, cutoff=max_length):
                paths.append(path)
                if len(paths) >= max_paths:
                    break
            return paths
        except Exception as e:
            logger.debug(f"Path finding failed: {e}")
            return []

    def get_evidence_for_path(self, path: List[str]) -> List[str]:
        if not self.kg_loaded or self.kg is None:
            return []
        evidence_ids = []
        try:
            for i in range(len(path) - 1):
                edge_data = self.kg.get_edge_data(path[i], path[i + 1])
                if edge_data and 'evidence_id' in edge_data:
                    evidence_ids.append(edge_data['evidence_id'])
            return evidence_ids
        except Exception as e:
            logger.debug(f"Evidence extraction failed: {e}")
            return []

    def score_path_quality(self, path: List[str]) -> float:
        if not self.kg_loaded or self.kg is None or len(path) < 2:
            return 0.0
        try:
            length_penalty = 1.0 / len(path)
            edge_weights = []
            for i in range(len(path) - 1):
                edge_data = self.kg.get_edge_data(path[i], path[i + 1])
                if edge_data and 'weight' in edge_data:
                    edge_weights.append(edge_data['weight'])
            avg_weight = np.mean(edge_weights) if edge_weights else 0.5
            return float(length_penalty * avg_weight)
        except Exception as e:
            logger.debug(f"Path scoring failed: {e}")
            return 0.0


# =============================================================================
# RAG-LLM PIPELINE
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator with Groq integration."""
    
    last_rag_output: List[Dict[str, Any]] = []

    def __init__(
        self,
        interventions_path: str,
        faiss_index_path: str,
        id_map_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        kg_path: Optional[str] = None,
        xai_api_key: Optional[str] = None,  # Keep for compatibility, but use for Groq
        enable_grok: bool = True,  # Keep naming for compatibility
        grok_model: str = "llama-3.3-70b-versatile"  # Groq model
    ):
        """Initialize pipeline with RAG components and optional Groq integration."""
        logger.info("=" * 70)
        logger.info("INITIALIZING PIPELINE (RAG + GROQ)")
        logger.info("=" * 70)

        # 1. Initialize RAG Runner Components
        try:
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)

            logger.info(f"Loading interventions from: {interventions_path}")
            with open(interventions_path, 'r', encoding='utf-8') as f:
                self.interventions = json.load(f)
            logger.info(f"Loaded {len(self.interventions)} interventions")

            logger.info(f"Loading FAISS index from: {faiss_index_path}")
            self.faiss_index = faiss.read_index(faiss_index_path)
            logger.info(f"FAISS index size: {self.faiss_index.ntotal}")

            self.id_map: Dict[str, str] = {}
            with open(id_map_path, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
            logger.info(f"Loaded ID Map with {len(self.id_map)} entries")
            
        except Exception as e:
            logger.error(f"❌ RAG Component Initialization Failed: {e}")
            raise

        # 2. Initialize Knowledge Graph service
        self.kg_service: Optional[KnowledgeGraphService] = None
        if kg_path is not None:
            self.kg_service = KnowledgeGraphService(kg_path)
        
        # 3. Initialize Feasibility Analyzer
        self.feasibility_analyzer = FeasibilityAnalyzer()

        # 4. Initialize Groq Service (not Grok/xAI)
        self.grok_service: Optional[GroqService] = None  # Keep variable name for compatibility
        if enable_grok:
            self.grok_service = GroqService(api_key=xai_api_key, model_name=grok_model)
        
        logger.info("=" * 70)
        logger.info("PIPELINE READY")
        logger.info("=" * 70 + "\n")

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query text."""
        return self.embedding_model.encode([query])[0]

    def retrieve_candidates(
        self,
        query_embedding: np.ndarray,
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k intervention candidates using FAISS."""
        logger.info(f"🔍 Retrieving top {k} candidates...")

        if not self.faiss_index:
            logger.error("FAISS index not loaded. Cannot retrieve candidates.")
            return []

        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)

        candidates: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue 
            
            intervention_id = self.id_map.get(str(idx)) 
            
            if intervention_id is None:
                logger.warning(f"Index {idx} not found in ID Map.")
                continue

            intervention = next(
                (item for item in self.interventions if item.get('intervention_id') == intervention_id), 
                None
            )
            
            if intervention is None:
                logger.warning(f"Intervention {intervention_id} not found.")
                continue
                
            intervention_copy = intervention.copy()
            similarity = 1.0 / (1.0 + float(dist))
            intervention_copy['relevance_score'] = float(similarity)
            candidates.append(intervention_copy)

        logger.info(f"✅ Retrieved {len(candidates)} candidates")
        return candidates

    def enhance_with_kg(
        self,
        problem_ids: List[str],
        candidates: List[Dict[str, Any]],
        max_paths_per_candidate: int = 3
    ) -> List[Dict[str, Any]]:
        """Enhance candidates with knowledge graph reasoning."""
        if not self.kg_service or not self.kg_service.kg_loaded:
            logger.info("⚠️ KG not available, skipping enhancement")
            return candidates

        logger.info(f"🔗 Enhancing {len(candidates)} candidates with KG...")

        enhanced_candidates: List[Dict[str, Any]] = []

        for candidate in candidates:
            intervention_id = candidate.get('intervention_id', 'unknown')

            all_paths = []
            all_evidence = []

            for problem_id in problem_ids:
                paths = self.kg_service.find_paths(
                    problem_id=problem_id,
                    intervention_id=intervention_id,
                    max_paths=max_paths_per_candidate
                )

                for path in paths:
                    all_paths.append(path)
                    evidence = self.kg_service.get_evidence_for_path(path)
                    all_evidence.extend(evidence)

            candidate['kg_paths'] = all_paths
            candidate['evidence_ids'] = list(set(all_evidence))

            if all_paths:
                path_scores = [
                    self.kg_service.score_path_quality(path)
                    for path in all_paths
                ]
                candidate['evidence_score'] = float(np.mean(path_scores))
            else:
                candidate['evidence_score'] = 0.0

            enhanced_candidates.append(candidate)

        logger.info("✅ KG enhancement complete")
        return enhanced_candidates

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Score and rank candidates."""
        logger.info(f"📊 Scoring {len(candidates)} candidates...")

        scored_candidates: List[Dict[str, Any]] = []

        for candidate in candidates:
            feasibility_result = self.feasibility_analyzer.analyze_feasibility(candidate)
            
            relevance = candidate.get('relevance_score', 0.5)
            evidence = candidate.get('evidence_score', 0.0)
            feasibility = feasibility_result['overall_score']

            final_score = (
                relevance * 0.40 +
                evidence * 0.30 +
                feasibility * 0.30
            )

            candidate['final_score'] = float(final_score)
            candidate['feasibility_score'] = feasibility_result
            scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        if scored_candidates:
            logger.info(f"✅ Top score: {scored_candidates[0]['final_score']:.3f}")
        
        return scored_candidates

    def prepare_candidates_for_api(
        self,
        site_summary: Dict[str, Any],
        top_candidates: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Prepare candidate data for API enhancement."""
        logger.info("🎯 Preparing candidates for API enhancement...")
        
        formatted_candidates: List[Dict[str, Any]] = []

        for i, candidate in enumerate(top_candidates[:max_recommendations], 1):
            score = candidate.get('final_score', 0.0)
            if score >= 0.80:
                priority = "HIGH"
            elif score >= 0.60:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            feas = candidate.get('feasibility_score', {})
            feas_score = feas.get('overall_score', 0.5) if isinstance(feas, dict) else 0.5
            
            formatted_candidate = {
                "intervention_id": candidate.get("intervention_id"),
                "intervention_name": candidate.get("intervention_name"),
                "intervention_description": candidate.get("intervention_description") or candidate.get("description", ""),
                "rank": i,
                "priority": priority,
                "final_score": score,
                "relevance_score": candidate.get('relevance_score', 0.0),
                "feasibility_score": feas_score,
                "feasibility_details": feas,
                "supporting_evidence": candidate.get('evidence_ids', []),
                "kg_paths": candidate.get('kg_paths', []),
                "technical_complexity": candidate.get("technical_complexity", "Unknown"),
                "time_to_implement": candidate.get("time_to_implement", "Unknown"),
                "estimated_cost_level": candidate.get("estimated_cost_level", "Unknown"),
                "implementation_disruption": candidate.get("implementation_disruption", "Unknown")
            }
            formatted_candidates.append(formatted_candidate)
        
        Pipeline.last_rag_output = formatted_candidates
        return formatted_candidates

    def generate_recommendations(
        self,
        site_summary: Dict[str, Any],
        scored_candidates: List[Dict[str, Any]],
        max_recommendations: int = 5,
        use_grok: bool = True
    ) -> Dict[str, Any]:
        """Generate final recommendations with optional Groq enhancement."""
        logger.info(f"📋 Generating {max_recommendations} recommendations...")
        
        # Format candidates
        formatted_candidates = self.prepare_candidates_for_api(
            site_summary=site_summary,
            top_candidates=scored_candidates,
            max_recommendations=max_recommendations
        )
        
        result = {
            "site_summary": site_summary,
            "candidates": formatted_candidates,
            "metadata": {
                "method": "rag-retrieval",
                "site_location": site_summary.get("location", "Unknown"),
                "groq_enhanced": False
            }
        }
        
        # Enhance with Groq if available and enabled
        if use_grok and self.grok_service:
            try:
                groq_result = self.grok_service.enhance_recommendations(
                    site_summary=site_summary,
                    candidates=formatted_candidates
                )
                
                if groq_result.get('enhanced'):
                    result['candidates'] = groq_result['candidates']
                    result['groq_summary'] = groq_result.get('summary', '')
                    result['metadata']['groq_enhanced'] = True
                    result['metadata']['groq_request_count'] = groq_result.get('request_count', 0)
                    logger.info("✅ Recommendations enhanced with Groq")
                else:
                    error = groq_result.get('error', 'Unknown error')
                    logger.warning(f"⚠️ Groq enhancement failed: {error}")
                    result['metadata']['groq_error'] = error
                    
            except Exception as e:
                logger.error(f"❌ Groq enhancement error: {e}")
                result['metadata']['groq_error'] = str(e)
        
        logger.info(f"✅ Generated {len(result['candidates'])} recommendations")
        return result

    def run_pipeline(
        self,
        site_summary: Dict[str, Any],
        k_candidates: int = 20,
        max_recommendations: int = 5,
        scoring_weights: Optional[Dict[str, float]] = None,
        use_grok: bool = True
    ) -> Dict[str, Any]:
        """Run complete pipeline for a site."""
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING PIPELINE")
        logger.info("=" * 70)

        try:
            # Stage 1: Build query
            query = self._build_query_from_summary(site_summary)
            logger.info(f"📝 Query: {query[:150]}...")
            
            # Stage 2: Embed query
            query_embedding = self.embed_query(query)

            # Stage 3: Retrieve candidates
            candidates = self.retrieve_candidates(query_embedding, k=k_candidates)

            # Stage 4: Enhance with KG (if available)
            problem_ids = site_summary.get('problem_ids', [])
            if problem_ids:
                candidates = self.enhance_with_kg(problem_ids, candidates)

            # Stage 5: Score and rank
            scored_candidates = self.score_candidates(candidates, weights=scoring_weights)

            # Stage 6: Generate final recommendations (with optional Groq)
            result = self.generate_recommendations(
                site_summary,
                scored_candidates,
                max_recommendations,
                use_grok=use_grok
            )

            # Add metadata
            result['pipeline_metadata'] = {
                'total_candidates_retrieved': len(candidates),
                'kg_enhanced': self.kg_service is not None and self.kg_service.kg_loaded,
                'groq_used': result['metadata'].get('groq_enhanced', False),
                'scoring_weights': scoring_weights,
                'query_used': query
            }

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 70 + "\n")

            return result

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise

    def _build_query_from_summary(self, site_summary: Dict[str, Any]) -> str:
        """Build search query dynamically from site summary."""
        parts: List[str] = []

        if 'description' in site_summary and site_summary['description']:
            parts.append(f"Description: {site_summary['description']}")
        
        if 'location' in site_summary:
            parts.append(f"Location: {site_summary['location']}")
        if 'problems' in site_summary:
            parts.append(f"Problems: {', '.join(site_summary['problems'])}")
        if 'road_type' in site_summary:
            parts.append(f"Road type: {site_summary['road_type']}")
        if 'crash_types' in site_summary:
            parts.append(f"Crash types: {', '.join(site_summary['crash_types'])}")
        if 'traffic_volume' in site_summary:
            parts.append(f"Traffic: {site_summary['traffic_volume']}")
        if 'speed_limit' in site_summary:
            parts.append(f"Speed limit: {site_summary['speed_limit']}")

        return " | ".join(parts)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("\n" + "=" * 70)
    print("ROAD SAFETY PIPELINE - GROQ API INTEGRATION")
    print("=" * 70 + "\n")
    
    pipeline = Pipeline(
        interventions_path="data/interventions.json",
        faiss_index_path="data/faiss_index.bin",
        id_map_path="data/id_map.json",
        kg_path="data/knowledge_graph.gml",
        xai_api_key=os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY"),
        enable_grok=True,
        grok_model="llama-3.3-70b-versatile"
    )
    
    # Example site summary
    site_summary = {
        "location": "Main Street and Oak Avenue intersection",
        "description": "High-traffic urban intersection with frequent rear-end collisions",
        "problems": ["Rear-end collisions", "Poor visibility", "High speed"],
        "problem_ids": ["P001", "P002"],
        "road_type": "Urban arterial",
        "crash_types": ["Rear-end", "Angle"],
        "traffic_volume": "High (>20,000 vehicles/day)",
        "speed_limit": "45 mph"
    }
    
    print("Site Summary:")
    print(f"  Location: {site_summary['location']}")
    print(f"  Problems: {', '.join(site_summary['problems'])}")
    print(f"  Road Type: {site_summary['road_type']}")
    print()
    
    # Run pipeline with Groq enhancement
    try:
        print("Running pipeline with Groq enhancement...")
        results = pipeline.run_pipeline(
            site_summary=site_summary,
            k_candidates=20,
            max_recommendations=5,
            use_grok=True  # Set to False to disable Groq enhancement
        )
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total Candidates Retrieved: {results['pipeline_metadata']['total_candidates_retrieved']}")
        print(f"Groq Enhanced: {results['metadata']['groq_enhanced']}")
        print(f"KG Enhanced: {results['pipeline_metadata']['kg_enhanced']}")
        print()
        
        print("Top Recommendations:")
        for i, candidate in enumerate(results['candidates'], 1):
            print(f"\n{i}. {candidate['intervention_name']}")
            print(f"   Priority: {candidate['priority']}")
            print(f"   Final Score: {candidate['final_score']:.3f}")
            print(f"   Feasibility: {candidate['feasibility_score']:.3f}")
            
            if 'groq_enhancement' in candidate:
                print(f"   Groq Rationale: {candidate['groq_enhancement']['rationale'][:100]}...")
        
        # Display Groq summary if available
        if 'groq_summary' in results:
            print("\n" + "-" * 70)
            print("GROQ ANALYSIS SUMMARY:")
            print("-" * 70)
            print(results['groq_summary'])
        
        # Save results
        output_file = Path(f"output/recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTrying with Groq disabled...")
        
        # Fallback to RAG-only mode
        results = pipeline.run_pipeline(
            site_summary=site_summary,
            k_candidates=20,
            max_recommendations=5,
            use_grok=False
        )
        
        print("\n✅ Successfully generated RAG-only recommendations")
        print(f"Total Recommendations: {len(results['candidates'])}")
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)