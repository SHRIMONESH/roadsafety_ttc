# build_kg_artifact.py

import networkx as nx
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define Paths ---
# CRITICAL: Project root path and artifact path
PROJECT_ROOT = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
KG_FILE_PATH = ARTIFACTS_DIR / "knowledge_graph.gml"

# --- Define Minimal Graph Structure (Nodes and Edges) ---
# We define a minimal graph relevant to road safety to satisfy the structure checks.
def create_minimal_kg():
    """Creates a minimal directed graph with relevant nodes and attributes."""
    G = nx.DiGraph() # Use a Directed Graph (DiGraph) to match your service's expected behavior
    
    # 1. Add Nodes (Entities)
    G.add_node("P001", type="Problem", label="Damaged Road Sign")
    G.add_node("I003", type="Intervention", label="Faded Road Sign - Hospital Sign")
    G.add_node("T01", type="Tag", label="Visibility")
    
    # 2. Add Edges (Relationships) with required properties
    # Your pipeline looks for 'evidence_id' and 'weight'
    G.add_edge(
        "P001", 
        "I003", 
        relation="addresses", 
        weight=0.8, 
        evidence_id="E001"
    )
    G.add_edge(
        "P001", 
        "T01", 
        relation="related_to", 
        weight=1.0, 
        evidence_id="E002"
    )
    
    logger.info(f"Created KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- Main Execution ---
if __name__ == "__main__":
    
    if not ARTIFACTS_DIR.exists():
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created missing directory: {ARTIFACTS_DIR}")
        
    G = create_minimal_kg()
    
    # Write the graph to the GML file format
    try:
        nx.write_gml(G, str(KG_FILE_PATH))
        logger.info(f"✅ SUCCESSFULLY CREATED KG FILE at: {KG_FILE_PATH.resolve()}")
        logger.info("Relaunch your Uvicorn backend now.")
    except Exception as e:
        logger.error(f"❌ FAILED to write GML file. Ensure NetworkX is installed correctly: {e}")