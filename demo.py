"""
Demo script for RAG system - WITH DATA VALIDATION
"""
import os
from dotenv import load_dotenv
from core.rag_runner import RAGRunner
from core.data_processor import DataProcessor

def run_demo(gemini_api_key: str, 
             csv_path: str,
             interventions_path: str,
             faiss_index_path: str,
             id_map_path: str):
    """Run demonstration with enhanced error handling"""
    
    # Initialize RAG
    print("Initializing RAG system...")
    rag = RAGRunner(
        gemini_api_key=gemini_api_key,
        interventions_path=interventions_path,
        faiss_index_path=faiss_index_path,
        id_map_path=id_map_path
    )
    
    # Load accident data with validation
    print(f"\nLoading accident data from {csv_path}...")
    try:
        df = DataProcessor.process_accident_data(csv_path)
        print(f"Loaded {len(df)} records\n")
        
        if len(df) == 0:
            print("ERROR: No data loaded from CSV")
            return
            
    except Exception as e:
        print(f"ERROR loading accident data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process first few records
    print("\n" + "="*80)
    print("GENERATING RECOMMENDATIONS")
    print("="*80)
    
    num_cases = min(3, len(df))
    
    for idx in range(num_cases):
        row = df.iloc[idx]
        print(f"\n{'='*80}")
        print(f"CASE #{idx + 1}")
        print(f"{'='*80}")
        
        # Create site summary
        try:
            site_summary = DataProcessor.create_site_summary(row)
            
            if not site_summary or not site_summary.strip():
                print("⚠️  WARNING: Empty site summary generated")
                print(f"Row sample: {dict(list(row.items())[:3])}")
                site_summary = "Insufficient data for this record"
            
            print(f"\nSite Summary:")
            print(site_summary)
            print(f"(Length: {len(site_summary)} characters)")
            
        except Exception as e:
            print(f"ERROR creating site summary: {e}")
            continue
        
        # Generate recommendations
        print("\nGenerating recommendations...")
        try:
            result = rag.generate_recommendations(site_summary, top_k=5)
            
            print(f"\nStatus: {result.polish_status}")
            print(f"Recommendations ({len(result.recommendations)}):")
            
            if len(result.recommendations) == 0:
                print("  ⚠️  No recommendations generated")
            else:
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"\n{i}. {rec.intervention_name}")
                    print(f"   Confidence: {rec.confidence:.2f}")
                    print(f"   Reason: {rec.reason}")
                    print(f"   Evidence: {', '.join(rec.evidence_ids)}")
                    
        except Exception as e:
            print(f"ERROR generating recommendations: {e}")
            continue

def main():
    """Main entry point"""
    load_dotenv()
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("⚠️  WARNING: Using hardcoded API key")
        gemini_api_key = "AIzaSyBSFiO7Z-TJwuRlOQQ9FSndnH_bKHrlkos"
    
    # Paths
    CSV_PATH = "data/raw/accidents.csv"
    INTERVENTIONS_PATH = "data/artifacts/interventions.json"
    FAISS_INDEX_PATH = "data/artifacts/faiss.index"
    ID_MAP_PATH = "data/artifacts/id_map.json"
    
    # Validate files exist
    missing = []
    for path, name in [
        (CSV_PATH, "Accidents CSV"),
        (INTERVENTIONS_PATH, "Interventions JSON"),
        (FAISS_INDEX_PATH, "FAISS Index"),
        (ID_MAP_PATH, "ID Map")
    ]:
        if not os.path.exists(path):
            missing.append(f"  ❌ {name}: {path}")
    
    if missing:
        print("ERROR: Missing required files:")
        print("\n".join(missing))
        return
    
    # Run demo
    try:
        run_demo(
            gemini_api_key=gemini_api_key,
            csv_path=CSV_PATH,
            interventions_path=INTERVENTIONS_PATH,
            faiss_index_path=FAISS_INDEX_PATH,
            id_map_path=ID_MAP_PATH
        )
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()