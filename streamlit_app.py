import streamlit as st
import json
from datetime import datetime
import html
import re
import requests
from typing import List, Dict, Any
import os
import time

# Backend URL Configuration
# Use localhost/127.0.0.1 for local development
BACKEND_URL = "http://127.0.0.1:8000/recommendations"
HEALTH_CHECK_URL = "http://127.0.0.1:8000/health"

# Page configuration
st.set_page_config(
    page_title="Road Safety RAG System - Groq",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #f55036 0%, #f97316 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: 20%;
    }
    .ai-message {
        background-color: #f3f4f6;
        color: black;
        margin-right: 20%;
    }
    .recommendation-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s;
    }
    .recommendation-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-card p {
        line-height: 1.6;
    }
    .recommendation-card strong {
        font-weight: 600;
    }
    .priority-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .priority-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .priority-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .confidence-score {
        color: #2563eb;
        font-weight: 700;
        font-size: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .turn-indicator {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .groq-badge {
        background: linear-gradient(135deg, #f55036 0%, #f97316 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .rate-limit-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'backend_status' not in st.session_state:
    st.session_state.backend_status = "Unknown"

if 'groq_configured' not in st.session_state:
    st.session_state.groq_configured = False

if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("XAI_API_KEY", "")

if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0

if 'last_error' not in st.session_state:
    st.session_state.last_error = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🚦 Road Safety RAG System</h1>
    <p style="margin: 0; opacity: 0.9;">AI-Powered Intervention Recommendations with Groq</p>
    <span class="groq-badge">⚡ Powered by Llama 3.3 70B</span>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("---")
    
    # Groq API Configuration Section
    st.subheader("🔑 Groq API Setup")
    
    # API Key Input
    api_key_input = st.text_input(
        "Enter Groq API Key",
        value=st.session_state.api_key,
        type="password",
        help="Your API key will be sent securely to the backend"
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if api_key_input:
            st.session_state.groq_configured = True
            st.success("✅ API Key updated")
        else:
            st.session_state.groq_configured = False
    
    # Check if API key is set
    if st.session_state.api_key:
        st.success("✅ API Key configured")
        st.session_state.groq_configured = True
    else:
        st.warning("⚠️ No API key provided")
        
        with st.expander("📝 How to get Groq API Key"):
            st.markdown("""
            1. Visit [Groq Console](https://console.groq.com/)
            2. Sign in with your account (or create one)
            3. Navigate to **API Keys** section
            4. Click **"Create API Key"**
            5. Copy your API key (starts with `gsk_`)
            6. Paste it in the field above
            
            **Note:** The API key should start with `gsk_`
            
            **Free Tier:**
            - Groq offers generous free tier limits
            - Fast inference speed (up to 750 tokens/second)
            - Multiple models available
            - Check current limits at [console.groq.com](https://console.groq.com)
            """)
    
    st.divider()
    
    st.subheader("🎯 Model Settings")
    
    st.markdown("**RAG Model:** Sentence Transformers")
    st.markdown("**LLM Model:** Llama 3.3 70B Versatile")
    st.markdown("**Backend:** FastAPI + Groq SDK")
    st.markdown("**API Endpoint:** Groq (https://api.groq.com)")
    
    top_k = st.slider("Top K Recommendations", 1, 10, 5)
    
    # Retry Configuration
    st.markdown("**Retry Settings (for rate limiting)**")
    max_retries = st.slider("Max Retries", 1, 5, 3)
    initial_delay = st.slider("Initial Delay (seconds)", 1, 10, 2)
    
    st.divider()
    
    st.subheader("💡 Example Queries")
    example_queries = [
        "High accident rate at intersection with poor visibility",
        "There is a large pothole causing vehicle damage",
        "Frequent speeding violations on residential street",
        "Pedestrian safety concerns near school zone",
        "Multiple accidents during rain on curved road",
        "Sharp curve with inadequate warning signs",
        "School zone with no speed bumps",
        "Narrow bridge with no barriers causing accidents",
        "Unmarked pedestrian crossing near hospital",
        "Heavy truck traffic damaging road surface"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{query[:30]}"):
            st.session_state.example_query = query
    
    st.divider()
    
    st.subheader("📊 System Status")
    
    # Check backend status
    def check_backend_health():
        try:
            response = requests.get(HEALTH_CHECK_URL, timeout=2)
            if response.status_code == 200:
                data = response.json()
                pipeline_ready = data.get('pipeline_ready', False)
                groq_configured = data.get('groq_configured', False)
                
                if pipeline_ready and groq_configured:
                    return "🟢 Online (Groq Ready)", True
                elif pipeline_ready:
                    return "🟡 Online (Groq Unavailable)", False
                else:
                    return "🟠 Online (Pipeline Not Ready)", False
            else:
                return "🔴 Offline", False
        except requests.exceptions.RequestException:
            return "🔴 Offline", False
    
    backend_status, groq_ready = check_backend_health()
    st.session_state.backend_status = backend_status
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.write(backend_status)
    
    if groq_ready:
        st.success("⚡ Llama 3.3 70B Ready")
    elif st.session_state.api_key:
        st.warning("⚠️ Backend not configured for Groq")
    else:
        st.error("⚠️ Groq API Key Required")
    
    # Display retry information
    if st.session_state.retry_count > 0:
        st.info(f"🔄 Retries this session: {st.session_state.retry_count}")
    
    if st.session_state.last_error:
        with st.expander("⚠️ Last Error", expanded=False):
            st.error(st.session_state.last_error)
    
    st.info(f"🔗 Backend: `{BACKEND_URL}`")
    
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.retry_count = 0
        st.session_state.last_error = None
        st.rerun()

# Helper function for HTML escaping
def escape_html(text):
    """Safely escape HTML characters."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace('\x00', '')
    text = html.escape(text)
    text = text.replace('\n', '<br>')
    return text

def format_chat_history_for_backend(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format chat history for backend compatibility.
    Extracts only the essential fields needed by the backend.
    """
    formatted_history = []
    
    for msg in messages:
        if msg['role'] == 'user':
            formatted_history.append({
                "role": "user",
                "content": msg['content'],
                "timestamp": msg['timestamp']
            })
        elif msg['role'] == 'assistant':
            # For assistant messages, extract the analysis text
            content = msg.get('content', {})
            if isinstance(content, dict):
                # Use the 'analysis' field, which is usually the brief summary
                analysis = content.get('analysis', 'Assistant response')
            else:
                analysis = str(content)
            
            formatted_history.append({
                "role": "assistant",
                "content": analysis,
                "timestamp": msg['timestamp']
            })
    
    return formatted_history

def exponential_backoff_request(url: str, data: dict, max_retries: int = 3, initial_delay: int = 2, timeout: int = 60):
    """
    Make a request with exponential backoff for rate limiting.
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, timeout=timeout)
            
            # If successful, return response
            if response.status_code == 200:
                return response, None
            
            # Handle 429 rate limiting
            elif response.status_code == 429:
                st.session_state.retry_count += 1
                
                if attempt < max_retries - 1:
                    error_msg = f"Rate limited (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    st.warning(error_msg)
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    error_detail = response.json().get('detail', 'Rate limit exceeded')
                    return None, f"Rate limit exceeded after {max_retries} attempts: {error_detail}"
            
            # Handle other errors
            elif response.status_code == 503:
                error_detail = response.json().get('detail', 'Service unavailable')
                return None, f"Service unavailable: {error_detail}"
            else:
                error_detail = response.text[:500] if response.text else "Unknown error"
                return None, f"Backend error (Status {response.status_code}): {error_detail}"
                
        except requests.exceptions.ReadTimeout:
            return None, f"Request timed out after {timeout} seconds"
        except requests.exceptions.ConnectionError:
            return None, f"Cannot connect to backend server at {url}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    return None, "Max retries exceeded"

def query_rag_system(user_query: str, chat_history: List[Dict[str, Any]], top_k: int = 5, max_retries: int = 3, initial_delay: int = 2):
    """
    Queries the RAG backend service powered by Groq API with exponential backoff.
    """
    
    try:
        # Format chat history for backend
        formatted_history = format_chat_history_for_backend(chat_history)
        
        # Prepare request data
        request_data = {
            "user_query": user_query,
            "max_recommendations": top_k,
            "chat_history": formatted_history
        }
        
        # Add API key to request if provided (optional - backend can use env var)
        if st.session_state.api_key:
            request_data["groq_api_key"] = st.session_state.api_key
        
        # Log request for debugging
        st.session_state.last_request = request_data
        
        # Make POST request with exponential backoff
        response, error = exponential_backoff_request(
            BACKEND_URL,
            request_data,
            max_retries=max_retries,
            initial_delay=initial_delay,
            timeout=60
        )
        
        if error:
            st.session_state.last_error = error
            st.error(f"❌ {error}")
            
            # Return error response
            return {
                "analysis": "❌ Request Failed",
                "recommendations": [],
                "additional_notes": error,
                "turn_type": "ERROR"
            }
        
        if response and response.status_code == 200:
            result = response.json()
            
            # Store raw response for debugging
            st.session_state.last_raw_response = result
            st.session_state.last_error = None
            
            # Extract recommendations
            backend_recs = result.get('recommendations', [])
            
            # Format for frontend display
            frontend_output = {
                "analysis": result.get('analysis', "Analysis from RAG Pipeline"),
                "recommendations": [],
                "additional_notes": result.get('additional_notes', ''),
                "turn_type": result.get('turn_type', 'UNKNOWN'),
                "pipeline_metadata": result.get('pipeline_metadata', {}),
                "debug": {
                    "backend_rec_count": len(backend_recs),
                    "turn_type": result.get('turn_type'),
                    "has_metadata": 'pipeline_metadata' in result,
                    "groq_used": result.get('pipeline_metadata', {}).get('groq_used', False),
                    "model_version": result.get('pipeline_metadata', {}).get('groq_model', 'llama-3.3-70b-versatile')
                }
            }
            
            # Map backend fields to frontend format
            for rec in backend_recs:
                intervention_name = (
                    rec.get('intervention_name') or 
                    rec.get('intervention') or 
                    'Unknown Intervention'
                )
                
                # Use reason as the source for both reason and impact if impact is missing
                reason_text = rec.get('reason') or 'Selection based on relevance score'
                expected_impact = rec.get('expected_impact') or reason_text
                implementation_notes = rec.get('implementation_notes', '')
                estimated_cost = rec.get('estimated_cost', 'Contact authorities for estimate')
                
                confidence = rec.get('confidence', rec.get('final_score', 0.5))
                priority = rec.get('priority', 'Medium').upper()
                
                frontend_output['recommendations'].append({
                    "intervention": intervention_name,
                    "priority": priority,
                    "reason": reason_text,
                    "expected_impact": expected_impact,
                    "implementation_notes": implementation_notes,
                    "estimated_cost": estimated_cost,
                    "confidence": confidence,
                    "rank": rec.get('rank', 0),
                    "intervention_id": rec.get('intervention_id', ''),
                    "relevance_score": rec.get('relevance_score', 0),
                    "feasibility_score": rec.get('feasibility_score', 0)
                })
            
            return frontend_output

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        st.session_state.last_error = error_msg
        st.error(f"❌ {error_msg}")
        return {
            "analysis": "❌ Unexpected Error",
            "recommendations": [],
            "additional_notes": str(e),
            "turn_type": "ERROR"
        }

def clean_analysis_text(text: str) -> str:
    """Remove hardcoded location examples and generic placeholders from analysis text."""
    if not text:
        return text
    
    # Remove specific location references (Mumbai, NH-44, etc.)
    text = re.sub(r'\bon\s+NH-\d+\s+near\s+the\s+District\s+Hospital\s+in\s+Mumbai\b', 'at the described location', text, flags=re.IGNORECASE)
    text = re.sub(r'\bThe\s+site\s+on\s+NH-\d+[^.]*?in\s+Mumbai\s+experiences\b', 'The site experiences', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnear\s+the\s+District\s+Hospital\s+in\s+Mumbai\b', 'at the location', text, flags=re.IGNORECASE)
    text = re.sub(r'\bin\s+Mumbai\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNH-\d+\b', 'the road', text, flags=re.IGNORECASE)
    
    # Clean up any double spaces created by removals
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def render_ai_message_components(ai_content, timestamp):
    """Render AI message with blueprint-compliant formatting."""
    
    with st.container():
        st.markdown(f"**🤖 RAG Pipeline + Groq** • {timestamp}")
        
        # Turn type indicator
        turn_type = ai_content.get('turn_type', 'UNKNOWN')
        
        # Debugging and metadata access
        debug_info = ai_content.get('debug', {})
        pipeline_metadata = ai_content.get('pipeline_metadata', ai_content.get('metadata', {}))
        groq_used = debug_info.get('groq_used', False)
        model_version = debug_info.get('model_version', 'llama-3.3-70b-versatile')
        
        # Turn type visualization
        turn_configs = {
            'FIRST_TURN': {'emoji': '🎯', 'text': 'Initial Analysis', 'color': '#dbeafe'},
            'FOLLOW_UP': {'emoji': '🔄', 'text': 'Follow-up Response', 'color': '#dcfce7'},
            'OFF_TOPIC': {'emoji': '🛑', 'text': 'Off-Topic Query', 'color': '#fee2e2'},
            'ERROR': {'emoji': '❌', 'text': 'Error Occurred', 'color': '#fecaca'},
            'NO_RESULTS': {'emoji': '📭', 'text': 'No Results Found', 'color': '#fef3c7'},
        }
        
        config = turn_configs.get(turn_type, {'emoji': 'ℹ️', 'text': f'Response', 'color': '#f3f4f6'})
        
        groq_indicator = f"⚡ {model_version}" if groq_used else "📊 RAG"
        
        st.markdown(f"""
        <div style="background-color: {config['color']}; padding: 0.5rem 1rem; border-radius: 6px; margin: 0.5rem 0;">
            {config['emoji']} <strong>{config['text']}</strong> • <span style="font-size: 0.875rem;">{groq_indicator}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Parse and extract structured information from analysis
        analysis_text = ai_content.get("analysis", "")
        
        # Clean up any hardcoded location references
        analysis_text = clean_analysis_text(analysis_text)
        
        # Extract Issue Title and Severity (if present in structured format)
        issue_title = ""
        severity_info = ""
        summary_text = analysis_text
        
        if "Issue:" in analysis_text and "Severity:" in analysis_text:
            lines = analysis_text.split('\n')
            for line in lines:
                if line.strip().startswith("Issue:"):
                    issue_title = line.strip()
                elif line.strip().startswith("Summary:"):
                    idx = lines.index(line)
                    summary_text = '\n'.join(lines[idx:])
                    break
        
        # Display structured header if available
        if issue_title:
            st.markdown(f"### {issue_title}")
        
        # Display summary/analysis
        st.markdown("#### 📋 Summary & Analysis")
        st.markdown(summary_text if summary_text else "No analysis provided")
        
        # Recommendations section - Blueprint compliant format
        recommendations = ai_content.get('recommendations', [])
        if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
            
            # Limit to top 3 as per blueprint
            top_recs = recommendations[:3]
            
            st.markdown(f"#### 💡 Top {len(top_recs)} Recommended Interventions (Ranked)")
            
            for i, rec in enumerate(top_recs, 1):
                if not isinstance(rec, dict):
                    continue
                
                # Extract data
                priority = rec.get('priority', 'Medium').upper()
                confidence = rec.get('confidence', 0.5)
                try:
                    confidence_pct = int(float(confidence) * 100)
                except (ValueError, TypeError):
                    confidence_pct = 50
                
                intervention = rec.get('intervention', 'Unknown Intervention')
                reason = clean_analysis_text(rec.get('reason', 'No reason provided'))
                expected_impact = clean_analysis_text(rec.get('expected_impact', 'No impact specified'))
                implementation_notes = clean_analysis_text(rec.get('implementation_notes', ''))
                estimated_cost = rec.get('estimated_cost', '')
                
                # Map priority to timeframe (blueprint requirement)
                timeframe_map = {
                    'HIGH': 'Immediate (within 24-48 hours)',
                    'MEDIUM': 'Short-term (within 7 days)',
                    'LOW': 'Medium-term (1-4 weeks)'
                }
                timeframe = timeframe_map.get(priority, 'Short-term')
                
                # Estimate complexity based on available info
                complexity = "Medium"
                if "immediate" in reason.lower() or "temporary" in reason.lower():
                    complexity = "Low"
                elif "permanent" in reason.lower() or "redesign" in reason.lower():
                    complexity = "High"
                
                # Priority badge styling
                if 'HIGH' in priority:
                    badge_color = '#fee2e2'
                    text_color = '#991b1b'
                    severity_badge = 'HIGH'
                elif 'LOW' in priority:
                    badge_color = '#d1fae5'
                    text_color = '#065f46'
                    severity_badge = 'LOW'
                else:
                    badge_color = '#fef3c7'
                    text_color = '#92400e'
                    severity_badge = 'MEDIUM'
                
                # Render recommendation card (blueprint format)
                st.markdown(f"""
                <div class="recommendation-card" style="border-left: 4px solid {text_color};">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                        <strong style="font-size: 1.15rem; color: #1f2937;">{i}. {escape_html(intervention)}</strong>
                        <div style="display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0;">
                            <span style="background-color: {badge_color}; color: {text_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; white-space: nowrap;">{severity_badge}</span>
                            <span style="color: #2563eb; font-weight: 700; font-size: 0.95rem; white-space: nowrap;">Confidence: {confidence_pct}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Content boxes with proper escaping
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 0.75rem; background-color: #f9fafb; border-radius: 6px; border: 1px solid #e5e7eb;">
                    <p style="margin: 0.25rem 0; color: #374151; font-size: 0.9rem;"><strong>📌 Why:</strong> {escape_html(reason)}</p>
                    <p style="margin: 0.25rem 0; color: #059669; font-size: 0.9rem;"><strong>✅ Impact:</strong> {escape_html(expected_impact)}</p>
                    <p style="margin: 0.25rem 0; color: #7c3aed; font-size: 0.875rem;"><strong>⏱️ Timeframe:</strong> {escape_html(timeframe)}</p>
                    <p style="margin: 0.25rem 0; color: #ea580c; font-size: 0.875rem;"><strong>⚙️ Complexity:</strong> {complexity}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Implementation notes box
                if implementation_notes:
                    st.markdown(f"""
                    <div style="margin: 0.75rem 0; padding: 0.75rem; background-color: #ecfdf5; border-left: 4px solid #059669; border-radius: 6px;">
                        <p style="margin: 0; color: #047857; font-size: 0.9rem;"><strong>🔧 Implementation:</strong> {escape_html(implementation_notes)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cost estimation box
                if estimated_cost:
                    st.markdown(f"""
                    <div style="margin: 0.75rem 0 1.5rem 0; padding: 0.75rem; background-color: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 6px;">
                        <p style="margin: 0; color: #92400e; font-size: 0.9rem;"><strong>💰 Estimated Cost:</strong> {escape_html(estimated_cost)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Add spacing after last item
                    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Supporting Evidence & References
        evidence_ids = pipeline_metadata.get('evidence_ids', [])
        if evidence_ids:
            with st.expander("📚 Supporting Evidence & References", expanded=False):
                st.markdown("**Database References Used:**")
                for eid in evidence_ids[:3]:  # Show top 3
                    st.markdown(f"- `{eid}`")
        
        # Implementation Checklist (for top recommendation only) - ALWAYS SHOW
        if recommendations and len(recommendations) > 0:
            top_rec = recommendations[0]
            impl_notes = top_rec.get('implementation_notes', '')
            intervention_name = top_rec.get('intervention', 'Top Action')
            
            st.markdown("#### ✅ Quick Implementation Checklist (Top Action)")
            st.markdown(f"**Action: {intervention_name}**")
            st.markdown("")
            
            if impl_notes:
                # Try to parse steps from implementation notes
                steps = [s.strip() for s in impl_notes.split('.') if s.strip()]
                if steps:
                    for idx, step in enumerate(steps[:6], 1):  # Max 6 steps as per blueprint
                        st.markdown(f"{idx}. {step}")
                else:
                    # Generate default steps based on intervention type
                    st.markdown(f"1. Conduct site assessment and measure affected area")
                    st.markdown(f"2. Prepare materials and mobilize implementation team")
                    st.markdown(f"3. Execute intervention following IRC standards")
                    st.markdown(f"4. Conduct quality inspection and documentation")
                    st.markdown(f"5. Monitor effectiveness over initial period (7-14 days)")
            else:
                # Generate contextual steps
                priority = top_rec.get('priority', 'MEDIUM').upper()
                if priority == 'HIGH':
                    st.markdown(f"1. Deploy emergency signage and temporary traffic management (within 1-2 hours)")
                    st.markdown(f"2. Alert relevant authorities and mobilize repair crew (within 4-6 hours)")
                    st.markdown(f"3. Implement temporary fix or marking (within 24 hours)")
                    st.markdown(f"4. Schedule permanent intervention (within 48-72 hours)")
                    st.markdown(f"5. Conduct post-implementation safety audit")
                else:
                    st.markdown(f"1. Conduct detailed site survey and measurements")
                    st.markdown(f"2. Obtain necessary approvals and allocate resources")
                    st.markdown(f"3. Prepare site and implement intervention per IRC guidelines")
                    st.markdown(f"4. Perform quality checks and final inspection")
                    st.markdown(f"5. Document intervention and establish monitoring schedule")
            
            st.markdown("")
        
        # Risk & Mitigation Notes - ALWAYS SHOW
        st.markdown("#### ⚠️ Risk & Mitigation Notes")
        
        # Generate contextual risks based on recommendations
        if recommendations and len(recommendations) > 0:
            top_intervention = recommendations[0].get('intervention', '').lower()
            
            if 'marking' in top_intervention or 'lane' in top_intervention:
                st.markdown("""
                **Risk 1:** Road markings may wear out quickly in high-traffic areas  
                → **Mitigation:** Use high-quality thermoplastic or reflective materials; schedule quarterly inspections and touch-ups
                
                **Risk 2:** Poor visibility during adverse weather (rain, fog)  
                → **Mitigation:** Install supplementary cat-eye reflectors; consider raised pavement markers for critical sections
                """)
            elif 'hump' in top_intervention or 'speed' in top_intervention:
                st.markdown("""
                **Risk 1:** Speed humps may cause discomfort or vehicle damage if poorly designed  
                → **Mitigation:** Follow IRC:SP:84-2019 specifications strictly; install advance warning signs; use appropriate materials
                
                **Risk 2:** Emergency vehicle delays during critical situations  
                → **Mitigation:** Design humps with appropriate height (50-100mm); provide alternative routes where feasible
                """)
            elif 'lighting' in top_intervention or 'light' in top_intervention:
                st.markdown("""
                **Risk 1:** Power supply interruptions affecting lighting system  
                → **Mitigation:** Install battery backup systems; establish rapid response maintenance protocol
                
                **Risk 2:** Light pollution affecting nearby residential areas  
                → **Mitigation:** Use directional lighting; comply with illumination standards; install timers where appropriate
                """)
            elif 'barrier' in top_intervention or 'fence' in top_intervention:
                st.markdown("""
                **Risk 1:** Barriers may create visual obstruction or become hazards themselves  
                → **Mitigation:** Follow IRC:SP:73 guidelines; use appropriate height and spacing; install reflective elements
                
                **Risk 2:** Maintenance access restrictions  
                → **Mitigation:** Design with removable sections at strategic locations; maintain clear access pathways
                """)
            else:
                st.markdown("""
                **Risk 1:** Implementation delays due to resource or material unavailability  
                → **Mitigation:** Pre-identify approved vendors; maintain buffer stock of critical materials; develop contingency timeline
                
                **Risk 2:** Temporary disruption to traffic flow during implementation  
                → **Mitigation:** Schedule work during off-peak hours; implement proper traffic management plan; provide advance public notice
                """)
        else:
            st.markdown("""
            **Risk 1:** Implementation delays due to resource constraints  
            → **Mitigation:** Prepare contingency plans and alternative timelines; pre-identify approved contractors
            
            **Risk 2:** Temporary measures may fail in adverse weather conditions  
            → **Mitigation:** Regular inspection and reinforcement after rain/storms; use weather-resistant materials
            """)
        
        st.markdown("")
        
        # Confidence Score Summary
        if recommendations:
            avg_confidence = sum(rec.get('confidence', 0.5) for rec in recommendations[:3]) / min(len(recommendations), 3)
            confidence_level = "HIGH" if avg_confidence >= 0.7 else "MEDIUM" if avg_confidence >= 0.4 else "LOW"
            
            st.markdown(f"""
            <div style="background-color: #eff6ff; padding: 0.75rem; border-left: 4px solid #3b82f6; border-radius: 4px; margin-top: 1rem;">
                <strong>📊 Overall Confidence: {confidence_level} ({int(avg_confidence * 100)}%)</strong><br/>
                <span style="font-size: 0.875rem; color: #1e40af;">Based on database match quality and evidence strength</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional notes
        additional_notes = ai_content.get('additional_notes', '')
        if additional_notes:
            st.markdown(f"""
            <div style="background-color: #dbeafe; padding: 0.75rem; border-radius: 6px; margin-top: 1rem;">
                <strong>📝 Additional Notes:</strong> {escape_html(additional_notes)}
            </div>
            """, unsafe_allow_html=True)
        
        # Metadata footer (compact)
        if pipeline_metadata:
            metadata_timestamp = pipeline_metadata.get('timestamp', timestamp)
            pipeline_version = pipeline_metadata.get('pipeline_version', '1.2')
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0.5rem; background-color: #f9fafb; border-radius: 4px; font-size: 0.75rem; color: #6b7280;">
                <strong>Metadata:</strong> pipeline_version: {pipeline_version} | timestamp: {metadata_timestamp} | evidence_count: {len(evidence_ids)}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

# Display rate limiting warning if there have been retries
if st.session_state.retry_count > 0:
    st.markdown(f"""
    <div class="rate-limit-warning">
        <strong>⚠️ Rate Limiting Detected</strong><br/>
        You've hit the API rate limit {st.session_state.retry_count} time(s) this session.<br/>
        <small>Consider waiting a few seconds between requests to avoid rate limiting.</small>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        if message['role'] == 'user':
            escaped_content = escape_html(message['content'])
            escaped_timestamp = escape_html(message['timestamp'])
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You</strong> • {escaped_timestamp}<br/>
                {escaped_content}
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                render_ai_message_components(message['content'], message['timestamp'])
            except Exception as e:
                st.error(f"Error rendering message {idx}: {e}")
                with st.expander("🔍 Debug - Raw Message Content"):
                    st.json(message)

# Input section
st.divider()

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Describe the road safety issue...",
        key="user_input",
        placeholder="e.g., There is a large pothole causing vehicle damage",
        value=st.session_state.get('example_query', ''),
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("🚀 Send", use_container_width=True)

# Handle example query selection
if 'example_query' in st.session_state:
    del st.session_state.example_query

# Process input
if send_button and user_input:
    
    # Check backend status first
    if "Offline" in st.session_state.backend_status:
        st.error("⚠️ Backend is offline. Please start the backend server first.")
        st.code("uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    elif not st.session_state.api_key:
        st.error("⚠️ Please provide a Groq API key in the sidebar configuration.")
    else:
        # Add user message
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message) 
        
        # Query RAG system
        with st.spinner(f"🔍 Analyzing with RAG Pipeline + Groq..."):
            result = query_rag_system(
                user_input, 
                st.session_state.messages,
                top_k=top_k,
                max_retries=max_retries,
                initial_delay=initial_delay
            )
            
            # Add AI message
            ai_message = {
                'role': 'assistant',
                'content': result,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.messages.append(ai_message)
            
        st.rerun()

# No welcome message - empty chat to start

# Footer
st.divider()

footer_col1, footer_col2 = st.columns([3, 1])

with footer_col1:
    st.caption("💡 Blueprint-Compliant Output | RAG + Llama 3.3 70B (Groq) | Evidence-Based Recommendations")

with footer_col2:
    if st.button("🔍 Debug Info"):
        with st.expander("Debug Information", expanded=True):
            st.write("**Backend Status:**", st.session_state.backend_status)
            st.write("**Groq Configured:**", st.session_state.groq_configured)
            st.write(f"**Backend URL:** `{BACKEND_URL}`")
            st.write(f"**Health Check URL:** `{HEALTH_CHECK_URL}`")
            st.write("**Message Count:**", len(st.session_state.messages))
            st.write("**Model Version:** Llama 3.3 70B Versatile")
            st.write("**API Provider:** Groq")
            st.write("**Retry Count:**", st.session_state.retry_count)
            
            if hasattr(st.session_state, 'last_request'):
                st.write("**Last Request Sent:**")
                # Redact API key in debug output
                debug_request = st.session_state.last_request.copy()
                if 'groq_api_key' in debug_request:
                    debug_request['groq_api_key'] = '***REDACTED***'
                st.json(debug_request)
            
            if hasattr(st.session_state, 'last_raw_response'):
                st.write("**Last Raw Response:**")
                st.json(st.session_state.last_raw_response)