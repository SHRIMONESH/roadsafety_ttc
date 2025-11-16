from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import sys
from pathlib import Path
import json
import logging
import os
import re
import time
from datetime import datetime

# --- Setup Paths and Imports ---
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed. Run: pip install python-dotenv")

# Initialize logger
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# Import Groq SDK (NOT OpenAI SDK)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    logger.info("✅ Groq SDK imported successfully")
except ImportError:
    logger.warning("⚠️ Groq SDK not available. Install with: pip install groq")
    GROQ_AVAILABLE = False
    Groq = None  # type: ignore

# Get API key from environment - Check for both GROQ_API_KEY and XAI_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY")

if GROQ_API_KEY:
    logger.info(f"✅ API Key found: {GROQ_API_KEY[:10]}...{GROQ_API_KEY[-4:]}")
else:
    logger.warning("⚠️ No API key found in environment variables")

if GROQ_API_KEY and GROQ_AVAILABLE and Groq is not None:
    try:
        # Initialize Groq client with proper configuration
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq API client configured successfully")
    except Exception as e:
        groq_client = None
        logger.error(f"❌ Failed to initialize Groq client: {e}")
else:
    groq_client = None
    logger.warning("⚠️ Groq client not initialized - missing API key or SDK")

# Import core pipeline
try:
    from core.pipeline import Pipeline
except ImportError as e:
    raise ImportError(f"Cannot import Pipeline from core.pipeline: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Road Safety RAG Pipeline API",
    description="Backend service for generating intervention recommendations using RAG + Groq API.",
    version="3.1"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Models ---
class ChatMessage(BaseModel):
    """Chat message model with strict typing."""
    role: str
    content: Union[str, Dict[str, Any]] 
    timestamp: str

class RecommendationRequest(BaseModel):
    """Request model for recommendations endpoint."""
    user_query: str
    max_recommendations: int = 5
    chat_history: List[ChatMessage] = Field(default_factory=list)

# --- Global Pipeline Instance ---
pipeline_instance: Optional[Pipeline] = None

# --- Configuration ---
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast, powerful model
# Alternative models: "mixtral-8x7b-32768", "llama-3.1-70b-versatile"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

@app.on_event("startup")
def load_pipeline() -> None:
    """Initialize the RAG pipeline on server startup."""
    global pipeline_instance
    try:
        logger.info("=" * 80)
        logger.info("🚀 Initializing RAG Pipeline with Groq Integration...")
        
        base_dir = PROJECT_ROOT
        
        # Initialize pipeline with Groq integration
        pipeline_instance = Pipeline(
            interventions_path=str(base_dir / "data/artifacts/interventions.json"),
            faiss_index_path=str(base_dir / "data/artifacts/faiss.index"),
            id_map_path=str(base_dir / "data/artifacts/id_map.json"),
            kg_path=str(base_dir / "data/artifacts/knowledge_graph.gml"),
            xai_api_key=GROQ_API_KEY,  # Pass the key to pipeline
            enable_grok=True,
            grok_model=GROQ_MODEL
        )
        logger.info("✅ RAG Pipeline initialization complete (Groq API mode)")
        logger.info(f"📊 Model: {GROQ_MODEL}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ FATAL: Pipeline initialization failed: {e}", exc_info=True)
        pipeline_instance = None

@app.get("/")
def read_root() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "Backend Online", 
        "pipeline_ready": pipeline_instance is not None,
        "groq_available": GROQ_AVAILABLE and GROQ_API_KEY is not None,
        "model": GROQ_MODEL,
        "api_version": "3.1",
        "api_key_configured": GROQ_API_KEY is not None
    }

def build_groq_prompt(
    site_summary: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    user_query: str,
    chat_history: Optional[List[ChatMessage]] = None
) -> str:
    """Build comprehensive prompt for Groq API."""
    
    context_parts = []
    
    # System context
    context_parts.append("You are an expert road safety engineer specializing in traffic safety interventions.")
    context_parts.append("Analyze the site data and recommend evidence-based interventions.")
    context_parts.append("")
    
    # Site analysis request
    context_parts.append("SITE ANALYSIS REQUEST:")
    context_parts.append(f"Location: {site_summary.get('location', 'Unknown')}")
    context_parts.append(f"User Query: {user_query}")
    context_parts.append(f"Description: {site_summary.get('description', '')}")
    context_parts.append(f"Road Type: {site_summary.get('road_type', 'Unknown')}")
    context_parts.append(f"Traffic Volume: {site_summary.get('traffic_volume', 'Unknown')}")
    context_parts.append(f"Speed Limit: {site_summary.get('speed_limit', 'Unknown')}")
    context_parts.append(f"Problems: {', '.join(site_summary.get('problems', []))}")
    context_parts.append("")
    
    # Candidate interventions
    context_parts.append("CANDIDATE INTERVENTIONS (Retrieved by RAG system):")
    for i, candidate in enumerate(candidates, 1):
        context_parts.append(f"\n{i}. {candidate.get('intervention_name', 'Unknown')}")
        context_parts.append(f"   ID: {candidate.get('intervention_id', 'N/A')}")
        context_parts.append(f"   Description: {candidate.get('intervention_description', 'N/A')}")
        context_parts.append(f"   Relevance Score: {candidate.get('relevance_score', 0):.3f}")
        context_parts.append(f"   Feasibility Score: {candidate.get('feasibility_score', 0):.3f}")
        context_parts.append(f"   Final Score: {candidate.get('final_score', 0):.3f}")
        context_parts.append(f"   Priority: {candidate.get('priority', 'MEDIUM')}")
        context_parts.append(f"   Complexity: {candidate.get('technical_complexity', 'Unknown')}")
        context_parts.append(f"   Implementation Time: {candidate.get('time_to_implement', 'Unknown')}")
        context_parts.append(f"   Cost Level: {candidate.get('estimated_cost_level', 'Unknown')}")
    
    context_parts.append("")
    
    # Chat history for follow-up queries
    if chat_history and len(chat_history) > 0:
        context_parts.append("CONVERSATION HISTORY:")
        for msg in chat_history[-5:]:  # Last 5 messages for context
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            context_parts.append(f"{msg.role.upper()}: {content_str[:200]}")
        context_parts.append("")
    
    # Task instructions
    context_parts.append("TASK:")
    context_parts.append("For FIRST-TIME queries: Provide comprehensive analysis with all recommendations")
    context_parts.append("For FOLLOW-UP queries: Provide focused answers based on conversation history")
    context_parts.append("")
    context_parts.append("Return your response in valid JSON format with this structure:")
    context_parts.append("""{
  "analysis": "Expert analysis of the situation (2-3 sentences)",
  "recommendations": [
    {
      "intervention_id": "ID from candidate list",
      "intervention_name": "Name from candidate list",
      "rank": 1,
      "priority": "HIGH/MEDIUM/LOW",
      "expected_impact": "Detailed explanation of expected safety improvements",
      "reason": "Why this intervention is recommended for this specific site",
      "implementation_notes": "Practical guidance for implementation",
      "estimated_cost": "Cost estimate or guidance",
      "confidence": 0.85
    }
  ],
  "additional_notes": "Any additional context or recommendations"
}""")
    context_parts.append("")
    context_parts.append("IMPORTANT:")
    context_parts.append("- Use intervention_id and intervention_name exactly as given in candidates")
    context_parts.append("- Provide specific, actionable insights")
    context_parts.append("- Consider site context in your reasoning")
    context_parts.append("- Return ONLY valid JSON, no markdown formatting")
    
    return "\n".join(context_parts)

def call_groq_api_for_analysis(
    site_summary: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    user_query: str,
    chat_history: Optional[List[ChatMessage]] = None
) -> Dict[str, Any]:
    """
    Call Groq API to generate enhanced recommendations and analysis.
    Uses Groq SDK with JSON mode for structured output.
    """
    if not GROQ_AVAILABLE or not GROQ_API_KEY or groq_client is None:
        raise HTTPException(
            status_code=503,
            detail="Groq API not available. Please configure GROQ_API_KEY and install groq package."
        )
    
    logger.info(f"🤖 Calling Groq API ({GROQ_MODEL}) for recommendation analysis...")
    
    # Build prompt
    prompt = build_groq_prompt(site_summary, candidates, user_query, chat_history)
    
    # Retry logic for API calls
    for attempt in range(MAX_RETRIES):
        try:
            # Verify groq_client is available
            if groq_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="Groq client not initialized"
                )
            
            # Call Groq API
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a road safety expert. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000,
                response_format={"type": "json_object"}  # Request structured JSON
            )
            
            # Verify response is valid and extract content
            if not response or not response.choices or len(response.choices) == 0:
                raise ValueError("Invalid response from Groq API - no choices returned")
            
            # Extract response text safely
            first_choice = response.choices[0]
            if not first_choice.message or not first_choice.message.content:
                raise ValueError("Invalid response from Groq API - no content in message")
            
            response_text: str = first_choice.message.content
            
            # Parse JSON response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present (backup cleanup)
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*$', '', response_text)
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate structure
            if 'recommendations' not in result:
                result['recommendations'] = []
            if 'analysis' not in result:
                result['analysis'] = "Analysis generated by Groq API"
            
            # Enrich recommendations with full candidate data
            for rec in result.get('recommendations', []):
                rec_id = rec.get('intervention_id')
                # Find matching candidate to add missing fields
                for candidate in candidates:
                    if candidate.get('intervention_id') == rec_id:
                        rec['final_score'] = candidate.get('final_score', 0.0)
                        rec['relevance_score'] = candidate.get('relevance_score', 0.0)
                        rec['feasibility_score'] = candidate.get('feasibility_score', 0.0)
                        rec['supporting_evidence'] = candidate.get('supporting_evidence', [])
                        rec['kg_paths'] = candidate.get('kg_paths', [])
                        rec['problem_ids'] = site_summary.get('problem_ids', [])
                        rec['technical_complexity'] = candidate.get('technical_complexity', 'Unknown')
                        rec['time_to_implement'] = candidate.get('time_to_implement', 'Unknown')
                        rec['estimated_cost_level'] = candidate.get('estimated_cost_level', 'Unknown')
                        break
            
            logger.info(f"✅ Groq API returned {len(result.get('recommendations', []))} recommendations")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Groq response as JSON (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"⏳ Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                logger.error(f"Response text: {response_text[:500]}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse Groq API response after multiple attempts"
                )
                
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any([
                '429' in error_str,
                'rate limit' in error_str,
                'too many requests' in error_str
            ])
            
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"⏳ Rate limit hit. Retrying in {delay} seconds... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            else:
                logger.error(f"❌ Groq API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Groq API error after {MAX_RETRIES} attempts: {str(e)}"
                    )
    
    # Should never reach here due to raises above
    raise HTTPException(
        status_code=500,
        detail="Unexpected error in Groq API call"
    )

@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest) -> Dict[str, Any]:
    """
    Main endpoint for generating recommendations.
    Uses RAG pipeline for retrieval + Groq API for generation.
    """
    if pipeline_instance is None:
        logger.error("❌ API called but Pipeline is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Pipeline service unavailable due to initialization failure"
        )
        
    logger.info("=" * 80)
    logger.info("🔥 NEW REQUEST RECEIVED")
    logger.info(f"📝 User Query: {request.user_query}")
    logger.info(f"📚 Chat History Length: {len(request.chat_history)}")

    # Check for off-topic queries
    irrelevant_keywords = ["eligibility", "loan", "mudra", "yojana", "financial", "personal", "bank"]
    if any(k.lower() in request.user_query.lower() for k in irrelevant_keywords):
        logger.warning(f"⚠️ OFF-TOPIC QUERY: {request.user_query}")
        return {
            "analysis": "This query is outside the scope of Road Safety Interventions.",
            "recommendations": [],
            "additional_notes": "The pipeline is specialized for road safety analysis only. Please ask about traffic safety, accident prevention, or road infrastructure improvements.",
            "turn_type": "OFF_TOPIC"
        }

    try:
        # 1. Build site summary from query
        simulated_site_summary: Dict[str, Any] = {
            "description": request.user_query, 
            "location": "NH-44, Junction near District Hospital, Mumbai",
            "problems": [request.user_query], 
            "problem_ids": ["P001", "P002", "P003"],
            "road_type": "National Highway",
            "crash_types": ["CT001", "CT005"],
            "severity": "High",
            "traffic_volume": "Very High",
            "speed_limit": "80 km/h"
        }
        
        # 2. Run RAG pipeline to get candidates (retrieval + scoring only)
        logger.info("🚀 Running RAG Pipeline (Retrieval + Scoring)...")
        rag_result = pipeline_instance.run_pipeline(
            site_summary=simulated_site_summary,
            k_candidates=request.max_recommendations * 2,
            max_recommendations=request.max_recommendations,
            use_grok=False  # We'll call Groq separately for better control
        )
        
        candidates = rag_result.get('candidates', [])
        
        if not candidates:
            logger.warning("⚠️ No candidates retrieved from RAG pipeline")
            return {
                "analysis": "No suitable interventions found for this query.",
                "recommendations": [],
                "additional_notes": "Try rephrasing your query or providing more details about the road safety issue.",
                "turn_type": "NO_RESULTS"
            }
        
        # 3. Call Groq API for intelligent analysis and generation
        logger.info("🤖 Calling Groq API for recommendation generation...")
        groq_result = call_groq_api_for_analysis(
            site_summary=simulated_site_summary,
            candidates=candidates,
            user_query=request.user_query,
            chat_history=request.chat_history
        )
        
        # 4. Determine turn type
        turn_type = "FIRST_TURN" if len(request.chat_history) <= 1 else "FOLLOW_UP"
        
        # 5. Add metadata
        groq_result['turn_type'] = turn_type
        groq_result['pipeline_metadata'] = {
            'total_candidates_retrieved': len(candidates),
            'groq_model': GROQ_MODEL,
            'groq_used': True,
            'rag_method': 'faiss_vector_search',
            'kg_enhanced': rag_result.get('pipeline_metadata', {}).get('kg_enhanced', False)
        }
        
        logger.info(f"✅ SUCCESS - Returning {len(groq_result.get('recommendations', []))} recommendations")
        logger.info("=" * 80)
        
        return groq_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Request processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Request processing failed: {str(e)}"
        )

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline_instance is not None,
        "groq_configured": groq_client is not None,
        "model": GROQ_MODEL,
        "api_version": "3.1",
        "components": {
            "rag_pipeline": pipeline_instance is not None,
            "groq_api": GROQ_AVAILABLE and GROQ_API_KEY is not None,
            "vector_search": pipeline_instance is not None and hasattr(pipeline_instance, 'faiss_index'),
            "knowledge_graph": pipeline_instance is not None and hasattr(pipeline_instance, 'kg_service')
        }
    }

# To run: uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload