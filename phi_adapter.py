"""
Phi-3 Local LLM Adapter with JSON5 Structured Output
====================================================
UPDATED: Uses JSON5 parser for robust handling of malformed JSON
"""

import logging
import json
import torch
import os 
import re 
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

# JSON5 parser - more forgiving than standard JSON
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    logging.warning("⚠️  json5 not installed. Install with: pip install json5")

logger = logging.getLogger(__name__)

# =============================================================================
# JSON5 EXTRACTOR (ROBUST PARSER)
# =============================================================================

class JSON5Extractor:
    """
    Uses JSON5 parser which handles:
    - Trailing commas
    - Single quotes
    - Unquoted keys
    - Comments
    - And many other JSON variations
    """
    
    @staticmethod
    def extract(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON/JSON5 from LLM response."""
        if not raw_text or len(raw_text.strip()) < 10:
            logger.error("❌ Empty or too short response")
            return None
        
        # Step 1: Clean markdown code fences
        cleaned = raw_text.strip()
        cleaned = re.sub(r'^```(?:json5?|JSON5?)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Step 2: Find JSON object boundaries
        json_match = re.search(r'(\{[\s\S]*\})', cleaned)
        
        if not json_match:
            logger.error("❌ No JSON object found in response")
            return None
        
        json_str = json_match.group(0).strip()
        
        # Step 3: Try JSON5 first (most forgiving)
        if HAS_JSON5:
            try:
                result = json5.loads(json_str)
                if isinstance(result, dict):
                    logger.info("✅ JSON5 parsed successfully")
                    return result
                else:
                    logger.warning(f"⚠️  JSON5 returned non-dict type: {type(result)}")
            except Exception as e:
                logger.warning(f"⚠️  JSON5 parsing failed: {e}")
        
        # Step 4: Fallback to standard JSON
        try:
            result = json.loads(json_str)
            logger.info("✅ Standard JSON parsed successfully")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"❌ Standard JSON parsing failed: {e}")
        
        # Step 5: Last resort - aggressive repair
        try:
            repaired = JSON5Extractor._repair_json(json_str)
            result = json.loads(repaired)
            logger.info("✅ Repaired JSON parsed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ All JSON parsing attempts failed: {e}")
            return None
    
    @staticmethod
    def _repair_json(json_str: str) -> str:
        """Aggressive JSON repair as last resort."""
        # Replace single quotes with double quotes
        repaired = json_str.replace("'", '"')
        
        # Remove trailing commas before } and ]
        repaired = re.sub(r',\s*}', '}', repaired)
        repaired = re.sub(r',\s*]', ']', repaired)
        
        # Quote unquoted keys (basic pattern)
        repaired = re.sub(r'([\{\,]\s*)(\w+)(\s*):', r'\1"\2"\3:', repaired)
        
        # Remove comments
        repaired = re.sub(r'//.*?$', '', repaired, flags=re.MULTILINE)
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
        
        return repaired

# =============================================================================
# PHI ADAPTER CLASS
# =============================================================================

class PhiAdapter:
    """
    Adapter for TinyLlama/Phi local LLM with JSON5 structured output.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False, 
        model_path: Optional[str] = None
    ):
        """Initialize LLM adapter with quantization support."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_path = model_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Setup quantization
        quantization_config = None
        if self.device == "cuda" and BitsAndBytesConfig:
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model and tokenizer
        try:
            load_id = self.model_path if self.model_path and os.path.exists(self.model_path) else model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto",
                "quantization_config": quantization_config
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(load_id, **model_kwargs)
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            logger.info("✅ LLM adapter initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM adapter: {e}")
            raise
    
    def _format_prompt_for_chat(self, user_prompt: str) -> str:
        """Format prompt for chat-based models."""
        system_message = (
            "You are a helpful AI assistant specialized in road safety analysis. "
            "You provide structured, accurate recommendations based on evidence and data. "
            "You must respond with valid JSON ONLY. No markdown, no explanations, just JSON."
        )
        
        formatted = (
            f"<|system|>\n{system_message}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        return formatted
    
    def _call_model(self, prompt: str) -> str:
        """Call the LLM model with the prompt."""
        try:
            formatted_prompt = self._format_prompt_for_chat(prompt)
            
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_full_text=False,
                truncation=True,  # Enable truncation explicitly
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]['generated_text'].strip()
            return response
            
        except Exception as e:
            logger.error(f"❌ Model inference failed: {e}")
            raise

    def _extract_json_from_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON using JSON5 parser."""
        return JSON5Extractor.extract(raw_response)
    
    def analyze_with_kg_context(
        self,
        site_summary: Dict[str, Any],
        top_candidates: List[Dict[str, Any]],
        max_candidates: int = 5
    ) -> Dict[str, Any]:
        """Analyze site and return structured recommendations."""
        logger.info("=" * 70)
        logger.info("ANALYZING SITE WITH LLM")
        
        try:
            prompt = self._build_basic_prompt(site_summary, top_candidates, max_candidates)
            
            response = self._call_model(prompt)
            
            # Use JSON5 extraction
            result = self._extract_json_from_response(response)
            
            if result is None:
                logger.error("❌ Failed to extract valid JSON")
                return self._generate_fallback(top_candidates, "LLM returned invalid JSON format")
            
            # Validate and format structure
            valid_recs = []
            for i, rec in enumerate(result.get('recommendations', [])):
                if not isinstance(rec, dict):
                    continue
                
                # Add missing fields with defaults
                rec.setdefault('intervention_id', top_candidates[i].get('intervention_id', f'I{i:03d}'))
                rec.setdefault('intervention_name', top_candidates[i].get('intervention_name', 'Unknown'))
                rec.setdefault('rank', i + 1)
                rec.setdefault('final_score', top_candidates[i].get('final_score', 0.5))
                rec.setdefault('confidence', 0.7)
                rec.setdefault('reason', 'Recommended by LLM analysis.')
                rec.setdefault('expected_impact', 'Impact analysis pending.')
                
                valid_recs.append(rec)
            
            result['recommendations'] = valid_recs
            result['analysis'] = result.get('analysis', 'Recommendations generated by LLM analysis')
            result['additional_notes'] = result.get('additional_notes', '')
            
            # Add metadata
            result['metadata'] = self.get_model_info()
            
            logger.info("✅ ANALYSIS COMPLETE")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._generate_fallback(top_candidates, f"Analysis exception: {str(e)}")

    def _build_basic_prompt(
        self,
        site_summary: Dict[str, Any],
        top_candidates: List[Dict[str, Any]],
        max_candidates: int
    ) -> str:
        """Build prompt for the LLM."""
        candidates_to_analyze = top_candidates[:max_candidates]
        
        prompt = f"""You are a road safety expert. Analyze the following site and recommend interventions.

SITE INFORMATION:
{json.dumps(site_summary, indent=2)}

INTERVENTION CANDIDATES:
{json.dumps(candidates_to_analyze, indent=2)}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. Do NOT use markdown (no ```json or ```)
3. Start with {{ and end with }}
4. Use double quotes for strings
5. NO trailing commas
6. NO comments

REQUIRED JSON FORMAT:
{{
  "recommendations": [
    {{
      "intervention_id": "string",
      "intervention_name": "string",
      "rank": 1,
      "priority": "High",
      "expected_impact": "Detailed explanation",
      "reason": "Why this intervention is recommended",
      "final_score": 0.85,
      "feasibility_score": 0.75,
      "confidence": 0.8
    }}
  ],
  "analysis": "Overall analysis summary",
  "additional_notes": "Implementation considerations"
}}

OUTPUT (JSON ONLY):"""
        
        return prompt
    
    def _generate_fallback(
        self,
        candidates: List[Dict[str, Any]],
        error: str = "Unknown error"
    ) -> Dict[str, Any]:
        """Generate fallback response when LLM fails."""
        logger.info("🔄 Generating fallback response...")
        
        fallback_recs = []
        
        for i, candidate in enumerate(candidates[:3], 1):
            final_score = candidate.get('final_score', 0.5)
            feas_score = candidate.get('feasibility_score', {}).get('overall_score', 0.5)
            priority = "HIGH" if final_score >= 0.8 else "MEDIUM" if final_score >= 0.6 else "LOW"
            
            fallback_recs.append({
                "intervention_id": candidate.get("intervention_id", f"I{i:03d}"),
                "intervention_name": candidate.get("intervention_name", "Unknown Intervention"),
                "rank": i,
                "priority": priority,
                "expected_impact": f"Score: {final_score:.2f}, Feasibility: {feas_score:.2f}",
                "reason": f"Fallback due to LLM failure. Error: {error}",
                "final_score": final_score,
                "confidence": 0.5
            })
        
        return {
            "recommendations": fallback_recs,
            "analysis": "Fallback analysis. LLM returned invalid JSON format",
            "additional_notes": "Generated by rule-based fallback logic",
            "turn_type": "FALLBACK"
        }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get current model information."""
        return {
            "model_name": self.model_name,
            "provider": 'local-llm',
            "device": self.device,
            "max_tokens": str(self.max_new_tokens),
            "temperature": str(self.temperature)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_phi_adapter(
    model_variant: str = "tinyllama",
    use_quantization: bool = True,
    model_path: Optional[str] = None 
) -> PhiAdapter:
    """Create PhiAdapter with common configurations."""
    model_map = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mini-4k": "microsoft/Phi-3-mini-4k-instruct",
        "medium-4k": "microsoft/Phi-3-medium-4k-instruct"
    }
    
    model_name = model_map.get(model_variant, model_map["tinyllama"])
    
    return PhiAdapter(
        model_name=model_name,
        device="auto",
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        load_in_4bit=use_quantization,
        load_in_8bit=False, 
        model_path=model_path
    )


# Export key components
__all__ = ['PhiAdapter', 'create_phi_adapter']