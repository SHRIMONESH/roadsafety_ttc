"""
Phi-3 Local LLM Adapter - BULLETPROOF VERSION
==============================================
Fixes: JSON parsing, prompt optimization, error recovery
Version: 5.0 - Production Hardened
"""

import logging
import json
import torch
import os 
import re
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

logger = logging.getLogger(__name__)


# =============================================================================
# BULLETPROOF JSON EXTRACTOR
# =============================================================================

class BulletproofJSONExtractor:
    @staticmethod
    def extract(raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text or len(raw_text.strip()) < 10:
            return None
        
        # Strategy 1: Direct parse
        try:
            return json.loads(raw_text)
        except:
            pass
        
        # Strategy 2: Remove markdown
        cleaned = raw_text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except:
            pass
        
        # Strategy 3: Regex extraction
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, cleaned, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and 'recommendations' in result:
                    logger.info("✅ JSON extracted via regex")
                    return result
            except:
                continue
        
        # Strategy 4: Fix trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        try:
            return json.loads(fixed)
        except:
            pass
        
        # Strategy 5: Quote keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        try:
            return json.loads(fixed)
        except:
            pass
        
        # Strategy 6: Fix quotes
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except:
            pass
        
        logger.error("❌ All extraction strategies failed")
        return None


# =============================================================================
# PHI ADAPTER CLASS
# =============================================================================

class PhiAdapter:
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
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_path = model_path
        
        logger.info("=" * 70)
        logger.info(f"🚀 Initializing LLM: {model_name}")
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"📍 Device: {self.device}")
        
        quantization_config = None
        if self.device == "cuda" and BitsAndBytesConfig:
            if load_in_4bit:
                logger.info("💾 4-bit quantization enabled")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
        
        try:
            load_id = self.model_path if self.model_path and os.path.exists(self.model_path) else model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                load_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                quantization_config=quantization_config
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            logger.info("✅ LLM initialized")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Init failed: {e}")
            raise
    
    def _format_prompt_for_chat(self, user_prompt: str) -> str:
        system = (
            "You are a road safety expert. You ALWAYS respond with valid JSON. "
            "You NEVER use markdown code fences. You NEVER add explanations outside the JSON."
        )
        
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
    
    def _call_model(self, prompt: str) -> str:
        try:
            formatted = self._format_prompt_for_chat(prompt)
            
            logger.info("🤖 Calling LLM...")
            
            outputs = self.pipe(
                formatted,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]['generated_text'].strip()
            logger.info(f"✅ Generated {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Model call failed: {e}")
            raise
    
    def _extract_json_from_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """BULLETPROOF: Use enhanced extractor."""
        return BulletproofJSONExtractor.extract(raw_response)
    
    def _build_optimized_prompt(
        self,
        site_summary: Dict[str, Any],
        top_candidates: List[Dict[str, Any]],
        max_candidates: int
    ) -> str:
        """OPTIMIZED: Build prompt that forces valid JSON."""
        
        problem = site_summary.get('description', 'Road safety issue')
        location = site_summary.get('location', 'Unknown')
        
        candidate_lines = []
        for i, c in enumerate(top_candidates[:max_candidates], 1):
            candidate_lines.append(
                f"{i}. {c.get('intervention_name', 'Unknown')} "
                f"(Score: {c.get('final_score', 0):.2f})\n"
                f"   {c.get('intervention_description', 'No description')[:100]}..."
            )
        
        candidates_text = "\n\n".join(candidate_lines)
        
        prompt = f"""Analyze this road safety problem and recommend interventions.

PROBLEM: {problem}
LOCATION: {location}

AVAILABLE INTERVENTIONS:
{candidates_text}

CRITICAL INSTRUCTIONS:
1. Output ONLY a JSON object (no markdown, no explanation)
2. Start with {{ and end with }}
3. Use exact intervention names from the list above
4. Provide detailed expected_impact (minimum 2 sentences)
5. Provide detailed reason (minimum 2 sentences)
6. NO trailing commas

REQUIRED JSON STRUCTURE:
{{"recommendations":[{{"intervention_id":"I001","intervention_name":"Exact Name From List","rank":1,"priority":"HIGH","expected_impact":"Detailed 2+ sentence explanation of how this improves safety for the specific problem.","reason":"Detailed 2+ sentence explanation of why this is the best choice for this location.","final_score":0.85,"confidence":0.80}}]}}

OUTPUT (JSON ONLY):"""
        
        return prompt
    
    def analyze_with_kg_context(
        self,
        site_summary: Dict[str, Any],
        top_candidates: List[Dict[str, Any]],
        max_candidates: int = 5
    ) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("ANALYZING WITH LLM")
        logger.info("=" * 70)
        
        try:
            prompt = self._build_optimized_prompt(site_summary, top_candidates, max_candidates)
            
            response = self._call_model(prompt)
            
            logger.info("🔍 Extracting JSON...")
            result = self._extract_json_from_response(response)
            
            if result is None:
                logger.error("❌ JSON extraction failed")
                logger.error(f"Response sample: {response[:200]}...")
                return self._generate_fallback(top_candidates, "LLM returned invalid JSON")
            
            if 'recommendations' not in result:
                logger.warning("⚠️ Missing recommendations field")
                result['recommendations'] = []
            
            # Validate and enrich
            valid_recs = []
            for i, rec in enumerate(result['recommendations']):
                if not isinstance(rec, dict):
                    continue
                
                rec.setdefault('intervention_id', top_candidates[i].get('intervention_id', f'I{i:03d}'))
                rec.setdefault('intervention_name', top_candidates[i].get('intervention_name', 'Unknown'))
                rec.setdefault('rank', i + 1)
                rec.setdefault('priority', 'MEDIUM')
                rec.setdefault('expected_impact', 'Impact analysis in progress')
                rec.setdefault('reason', 'Recommended by analysis')
                rec.setdefault('final_score', top_candidates[i].get('final_score', 0.5))
                rec.setdefault('confidence', 0.75)
                
                valid_recs.append(rec)
            
            result['recommendations'] = valid_recs
            result.setdefault('analysis', 'LLM analysis complete')
            result['metadata'] = {
                'model': self.model_name,
                'provider': 'local-llm',
                'device': self.device,
                'candidates_analyzed': len(top_candidates),
                'recommendations_generated': len(valid_recs)
            }
            
            logger.info(f"✅ {len(valid_recs)} recommendations generated")
            logger.info("=" * 70)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._generate_fallback(top_candidates, f"Exception: {str(e)}")
    
    def _generate_fallback(
        self,
        candidates: List[Dict[str, Any]],
        error: str = "Unknown error"
    ) -> Dict[str, Any]:
        logger.info(f"🔄 Generating fallback: {error}")
        
        recs = []
        for i, c in enumerate(candidates[:5], 1):
            feas = c.get('feasibility_score', {})
            feas_score = feas.get('overall_score', 0.5) if isinstance(feas, dict) else 0.5
            
            final_score = c.get('final_score', 0.5)
            priority = "HIGH" if final_score >= 0.7 else "MEDIUM" if final_score >= 0.5 else "LOW"
            
            recs.append({
                "intervention_id": c.get("intervention_id", f"I{i:03d}"),
                "intervention_name": c.get("intervention_name", "Unknown"),
                "rank": i,
                "priority": priority,
                "expected_impact": c.get("intervention_description", f"Score: {final_score:.2f}"),
                "reason": f"Selected based on computed scores. Relevance: {c.get('relevance_score', 0):.2f}",
                "final_score": final_score,
                "confidence": min(0.8, final_score + 0.1)
            })
        
        return {
            "recommendations": recs,
            "analysis": f"Rule-based analysis. {error}",
            "additional_notes": "Generated by fallback logic",
            "metadata": {
                "model": self.model_name,
                "provider": "local-llm",
                "fallback": True,
                "error": error
            }
        }
    
    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "provider": "local-llm",
            "device": self.device,
            "max_tokens": str(self.max_new_tokens),
            "temperature": str(self.temperature)
        }


def create_phi_adapter(
    model_variant: str = "tinyllama",
    use_quantization: bool = True,
    model_path: Optional[str] = None
) -> PhiAdapter:
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