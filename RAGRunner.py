import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import faiss
import numpy as np
import json
import traceback # Added for cleaner error reporting
from typing import List, Dict, Optional, Any

# Assuming these imports are correct for your project
from app.embeddings import EmbeddingManager


class RAGRunner:
    """
    Retrieval-Augmented Generation Runner for Road Safety Interventions.
    Uses FAISS for semantic search and provides pipeline-compatible retrieval.
    """
    
    def __init__(self, 
                 interventions_path: str,
                 faiss_index_path: str,
                 id_map_path: str):
        """
        Initialize RAG system with interventions database and FAISS index.
        """
        
        print(f"🔧 Initializing RAG Runner...")
        print(f"   - Interventions: {interventions_path}")
        print(f"   - FAISS Index: {faiss_index_path}")
        print(f"   - ID Map: {id_map_path}")
        
        # Initialize embedding manager
        try:
            self.embedder = EmbeddingManager()
            print("   ✅ Embedding manager initialized")
        except Exception as e:
            print(f"   ❌ Failed to initialize embedding manager: {e}")
            raise
        
        # Load interventions database
        try:
            with open(interventions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.interventions = data if isinstance(data, list) else [data]
            
            self.interventions_db = {
                inv['intervention_id']: inv 
                for inv in self.interventions 
                if 'intervention_id' in inv
            }
            
            print(f"   ✅ Loaded {len(self.interventions)} interventions")
            print(f"   ✅ Created lookup DB with {len(self.interventions_db)} entries")
            
        except Exception as e:
            print(f"   ❌ Failed to load interventions: {e}")
            raise
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(faiss_index_path)
            print(f"   ✅ Loaded FAISS index: {self.index.ntotal} vectors")
            
        except Exception as e:
            print(f"   ❌ Failed to load FAISS index: {e}")
            print("   💡 Ensure FAISS index was created with faiss.write_index()")
            raise
        
        # Load ID mapping
        try:
            with open(id_map_path, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
            
            print(f"   ✅ Loaded ID map: {len(self.id_map)} mappings")
            
            # Validate ID map consistency
            if len(self.id_map) != self.index.ntotal:
                print(f"   ⚠️ Warning: ID map size ({len(self.id_map)}) != index size ({self.index.ntotal})")
            
        except Exception as e:
            print(f"   ❌ Failed to load ID map: {e}")
            raise
        
        print("✅ RAG Runner initialization complete\n")
    
    def retrieve_top_k(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top K relevant interventions using semantic search.
        """
        
        if not query or not query.strip():
            print("⚠️ Warning: Empty query provided to retrieve_top_k")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_single(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                
                if idx == -1:
                    continue
                
                if str(idx) not in self.id_map:
                    print(f"   ⚠️ Warning: Index {idx} not found in id_map")
                    continue
                
                intervention_id = self.id_map[str(idx)]
                intervention = self.interventions_db.get(intervention_id)
                
                if intervention is None:
                    print(f"   ⚠️ Warning: Intervention ID {intervention_id} not found in database")
                    continue
                
                # Convert L2 distance to similarity score
                similarity = float(1 - (distance ** 2 / 2))
                
                results.append({
                    **intervention,
                    'similarity': similarity
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error in retrieve_top_k: {e}")
            traceback.print_exc()
            return []
    
    def retrieve_interventions(
        self, 
        site_summary: Dict[str, Any], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        PIPELINE-COMPATIBLE RETRIEVAL WRAPPER
        """
        
        print(f"\n{'='*80}")
        print(f"🔍 retrieve_interventions called")
        print(f"   - top_k: {top_k}")
        
        # ========== QUERY CONSTRUCTION ==========
        query_parts = []
        
        if not isinstance(site_summary, dict):
            print(f"⚠️ Warning: site_summary is not a dict, got {type(site_summary)}")
            if isinstance(site_summary, str):
                query_text = site_summary
            else:
                print("❌ Error: Cannot process site_summary")
                return []
        else:
            # Extract relevant fields to build query text
            if 'location' in site_summary and site_summary['location']:
                loc = site_summary['location']
                if isinstance(loc, str) and loc.strip():
                    query_parts.append(f"Location: {loc}")
            
            if 'name' in site_summary and site_summary['name']:
                query_parts.append(f"Site: {site_summary['name']}")
            
            problems = site_summary.get('problems', site_summary.get('issues', []))
            if isinstance(problems, list) and problems:
                problems_str = ", ".join(str(p) for p in problems if p)
                if problems_str:
                    query_parts.append(f"Problems: {problems_str}")
            
            crash_types = site_summary.get('crash_types', site_summary.get('common_crash_types', []))
            if isinstance(crash_types, list) and crash_types:
                crash_types_str = ", ".join(str(ct) for ct in crash_types if ct)
                if crash_types_str:
                    query_parts.append(f"Crash types: {crash_types_str}")
            
            if 'road_type' in site_summary and site_summary['road_type']:
                query_parts.append(f"Road type: {site_summary['road_type']}")
            
            if 'severity' in site_summary and site_summary['severity']:
                query_parts.append(f"Severity: {site_summary['severity']}")
            
            if 'traffic_volume' in site_summary and site_summary['traffic_volume']:
                query_parts.append(f"Traffic volume: {site_summary['traffic_volume']}")
            
            if 'description' in site_summary and site_summary['description']:
                desc = str(site_summary['description'])
                if desc.strip():
                    query_parts.append(desc)
            
            # Build final query
            query_text = ". ".join([str(x) for x in query_parts if x])
        
        # Validate query
        if not query_text or not query_text.strip():
            print("❌ Error: Empty query generated from site_summary")
            return []
        
        print(f"📝 Generated query: {query_text[:200]}...")
        
        # ========== FAISS SEARCH AND PROCESSING ==========
        
        # Call the core retrieval function
        retrieved_results = self.retrieve_top_k(query_text, top_k)
        
        # Normalize and prepare for pipeline output
        final_results = []
        for rank, intervention in enumerate(retrieved_results, 1):
            # Ensure the similarity score is standardized as 'relevance_score'
            # to meet the pipeline's expectations for Stage 5 (Scoring)
            intervention['relevance_score'] = intervention.pop('similarity') # FIX: Rename score
            
            # Add snippet for context
            intervention['snippet'] = self._create_snippet(intervention)
            
            # Add metadata (for debugging/tracking)
            intervention['candidate_meta'] = {
                "source": "faiss",
                "rank": rank,
            }
            
            final_results.append(intervention)
            
        print(f"✅ Retrieved {len(final_results)} valid interventions")
        
        if final_results:
            top = final_results[0]
            print(f"   📌 Top result: [{top['intervention_id']}] {top['intervention_name']}")
            print(f"   💯 Relevance: {top['relevance_score']:.3f}")
        else:
            print("   ⚠️ No valid results returned")
            
        print(f"{'='*80}\n")
        
        return final_results
    
    def _create_snippet(self, intervention: Dict[str, Any]) -> str:
        """
        Create a short snippet for display purposes.
        """
        
        description = intervention.get(
            'intervention_description',
            intervention.get('description', '')
        )
        
        if description:
            snippet = description[:150]
            if len(description) > 150:
                snippet += "..."
            return snippet
        
        return intervention.get('intervention_name', 'No description available')
    
    def get_intervention_by_id(self, intervention_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific intervention by its ID.
        """
        return self.interventions_db.get(intervention_id)
    
    def get_all_interventions(self) -> List[Dict[str, Any]]:
        """
        Get all interventions in the database.
        """
        return self.interventions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        """
        if not hasattr(self, 'index') or not hasattr(self.index, 'd'):
            dim = 'N/A'
            total = 'N/A'
        else:
            dim = self.index.d
            total = self.index.ntotal

        return {
            "total_interventions": len(self.interventions),
            "database_entries": len(self.interventions_db),
            "faiss_vectors": total,
            "id_map_entries": len(self.id_map),
            "embedding_dimension": dim
        }