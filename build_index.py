"""
Enhanced FAISS Index Rebuilder
- Validates intervention schema (intervention_id, intervention_name, intervention_description)
- Creates rich embeddings from multiple fields
- Builds proper ID map (index -> intervention_id)
- Comprehensive verification and testing
- Diagnostic output for troubleshooting
"""

import json
import os
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

def load_interventions(path: str) -> Tuple[List[Dict], List[str]]:
    """
    Load and validate interventions from JSON file.
    
    Returns:
        Tuple of (valid_interventions, error_messages)
    """
    errors = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            interventions = json.load(f)
    except FileNotFoundError:
        errors.append(f"File not found: {path}")
        return [], errors
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return [], errors
    
    if not isinstance(interventions, list):
        errors.append("Interventions file must contain a JSON array")
        return [], errors
    
    valid_interventions = []
    seen_ids = set()
    
    for i, intervention in enumerate(interventions):
        # Check required fields
        intervention_id = intervention.get('intervention_id')
        intervention_name = intervention.get('intervention_name')
        intervention_description = intervention.get('intervention_description')
        
        if not intervention_id:
            errors.append(f"Intervention at index {i} missing 'intervention_id'")
            continue
        
        # Check for duplicate IDs
        if intervention_id in seen_ids:
            errors.append(f"Duplicate intervention_id: {intervention_id}")
            continue
        seen_ids.add(intervention_id)
        
        # Must have either name or description
        if not intervention_name and not intervention_description:
            errors.append(f"Intervention {intervention_id} missing both name and description")
            continue
        
        valid_interventions.append(intervention)
    
    return valid_interventions, errors


def create_embedding_text(intervention: Dict) -> str:
    """
    Create rich text representation for embedding from intervention fields.
    
    Combines:
    - intervention_name
    - intervention_description  
    - category
    - subcategory
    - problem
    - crashtype_id
    """
    parts = []
    
    # Core fields
    if intervention.get('intervention_name'):
        parts.append(intervention['intervention_name'])
    
    if intervention.get('intervention_description'):
        parts.append(intervention['intervention_description'])
    
    # Context fields
    if intervention.get('category'):
        parts.append(f"Category: {intervention['category']}")
    
    if intervention.get('subcategory'):
        parts.append(f"Type: {intervention['subcategory']}")
    
    if intervention.get('problem'):
        parts.append(f"Addresses: {intervention['problem']}")
    
    if intervention.get('crashtype_id'):
        parts.append(f"Crash Type: {intervention['crashtype_id']}")
    
    return ". ".join(parts)


def rebuild_faiss_index(
    interventions_path: str = "data/artifacts/interventions.json",
    output_dir: str = "data/artifacts",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
):
    """
    Rebuild FAISS index and ID map from interventions.
    
    Args:
        interventions_path: Path to interventions.json
        output_dir: Directory to save index and id_map
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
    """
    print("="*70)
    print("REBUILDING FAISS INDEX - ENHANCED VERSION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and validate interventions
    print(f"\n📂 Loading interventions from: {interventions_path}")
    valid_interventions, errors = load_interventions(interventions_path)
    
    if errors:
        print("\n⚠️  Validation Warnings/Errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    
    if not valid_interventions:
        print("\n❌ No valid interventions found. Cannot build index.")
        return False
    
    print(f"✓ Loaded {len(valid_interventions)} valid interventions")
    
    # Step 2: Show sample of intervention IDs
    print("\n📋 Sample Intervention IDs:")
    sample_ids = [inv['intervention_id'] for inv in valid_interventions[:10]]
    print(f"   {', '.join(sample_ids)}")
    if len(valid_interventions) > 10:
        print(f"   ... and {len(valid_interventions) - 10} more")
    
    # Step 3: Load embedding model
    print(f"\n🤖 Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Step 4: Create embedding texts and ID map
    print("\n🔢 Preparing embeddings...")
    texts_to_embed = []
    id_map = {}
    
    for idx, intervention in enumerate(valid_interventions):
        intervention_id = intervention['intervention_id']
        text = create_embedding_text(intervention)
        
        texts_to_embed.append(text)
        id_map[str(idx)] = intervention_id
        
        # Show first 3 examples
        if idx < 3:
            print(f"\n   Example {idx}:")
            print(f"   ID: {intervention_id}")
            print(f"   Text: {text[:150]}...")
    
    print(f"\n✓ Prepared {len(texts_to_embed)} texts for embedding")
    
    # Step 5: Generate embeddings
    print("\n⚡ Generating embeddings...")
    try:
        embeddings = model.encode(
            texts_to_embed, 
            batch_size=batch_size,
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return False
    
    # Step 6: Build FAISS index
    print("\n🔍 Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatL2 for exact search
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add to index
    index.add(embeddings.astype('float32'))
    print(f"✓ FAISS index built with {index.ntotal} vectors (dimension: {dimension})")
    
    # Step 7: Save FAISS index
    index_path = os.path.join(output_dir, "faiss.index")
    print(f"\n💾 Saving FAISS index to: {index_path}")
    try:
        faiss.write_index(index, index_path)
        file_size = os.path.getsize(index_path)
        print(f"✓ FAISS index saved ({file_size:,} bytes)")
    except Exception as e:
        print(f"❌ Failed to save FAISS index: {e}")
        return False
    
    # Step 8: Save ID map
    id_map_path = os.path.join(output_dir, "id_map.json")
    print(f"💾 Saving ID map to: {id_map_path}")
    try:
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map, f, indent=2)
        file_size = os.path.getsize(id_map_path)
        print(f"✓ ID map saved ({file_size:,} bytes)")
    except Exception as e:
        print(f"❌ Failed to save ID map: {e}")
        return False
    
    # Step 9: Verification
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    # Reload and verify
    try:
        loaded_index = faiss.read_index(index_path)
        with open(id_map_path, 'r') as f:
            loaded_id_map = json.load(f)
        
        print(f"  ✓ FAISS Index Size: {loaded_index.ntotal}")
        print(f"  ✓ Dimension: {dimension}")
        print(f"  ✓ ID Map Entries: {len(loaded_id_map)}")
        print(f"  ✓ Index matches ID map: {loaded_index.ntotal == len(loaded_id_map)}")
        
        # Show ID map sample
        print(f"\n  ID Map Sample (first 5 entries):")
        for i in range(min(5, len(loaded_id_map))):
            print(f"    \"{i}\": \"{loaded_id_map[str(i)]}\"")
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False
    
    # Step 10: Test retrieval
    print("\n" + "="*70)
    print("🧪 TESTING RETRIEVAL")
    print("="*70)
    
    test_queries = [
        "damaged road sign",
        "pedestrian crossing",
        "speed hump traffic calming",
        "rear end collision"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-"*70)
        
        try:
            query_embedding = model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            k = 3
            distances, indices = index.search(query_embedding.astype('float32'), k)
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                intervention_id = id_map.get(str(idx))
                intervention = next(
                    (inv for inv in valid_interventions if inv['intervention_id'] == intervention_id), 
                    None
                )
                
                if intervention:
                    similarity = 1 - (dist / 2)  # Convert L2 distance to similarity
                    print(f"  {i}. [{intervention_id}] {intervention['intervention_name']}")
                    print(f"     Similarity: {similarity:.3f} | Category: {intervention.get('category', 'N/A')}")
                else:
                    print(f"  {i}. [ERROR] ID {intervention_id} not found in database!")
        
        except Exception as e:
            print(f"  ❌ Test query failed: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("✓ FAISS INDEX REBUILD COMPLETE")
    print("="*70)
    print("\n📍 Output Files:")
    print(f"   • FAISS Index: {index_path}")
    print(f"   • ID Map:      {id_map_path}")
    print("\n🚀 Next Steps:")
    print("   1. Verify the test queries above return relevant results")
    print("   2. Run your pipeline: python core/pipeline.py")
    print("   3. Check for 'Intervention with ID XXX not found' warnings")
    
    return True


def check_existing_files():
    """Check status of existing artifact files."""
    print("\n📋 CHECKING EXISTING FILES")
    print("="*70)
    
    files_to_check = {
        "Interventions": "data/artifacts/interventions.json",
        "FAISS Index": "data/artifacts/faiss.index",
        "ID Map": "data/artifacts/id_map.json",
        "Knowledge Graph": "data/artifacts/kg.jsonld",
        "Site Summary": "data/inputs/site_summary.json"
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            
            # Show entry count for JSON files
            extra_info = ""
            if path.endswith('.json') or path.endswith('.jsonld'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        extra_info = f", {len(data)} entries"
                    elif isinstance(data, dict):
                        extra_info = f", {len(data)} keys"
                except:
                    pass
            
            print(f"  ✓ {name:20s}: {size:>10,} bytes{extra_info}")
        else:
            print(f"  ✗ {name:20s}: NOT FOUND")
            all_exist = False
    
    print("="*70)
    return all_exist


def diagnose_mismatch(
    interventions_path: str = "data/artifacts/interventions.json",
    id_map_path: str = "data/artifacts/id_map.json"
):
    """
    Diagnose ID mismatches between interventions and ID map.
    """
    print("\n🔍 DIAGNOSING ID MISMATCH")
    print("="*70)
    
    # Load interventions
    try:
        with open(interventions_path, 'r') as f:
            interventions = json.load(f)
        intervention_ids = {inv['intervention_id'] for inv in interventions}
        print(f"✓ Interventions file has {len(intervention_ids)} unique IDs")
    except Exception as e:
        print(f"❌ Failed to load interventions: {e}")
        return
    
    # Load ID map
    try:
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
        mapped_ids = set(id_map.values())
        print(f"✓ ID map has {len(mapped_ids)} unique IDs")
    except FileNotFoundError:
        print(f"⚠️  ID map not found - needs to be rebuilt")
        return
    except Exception as e:
        print(f"❌ Failed to load ID map: {e}")
        return
    
    # Find mismatches
    in_map_not_in_interventions = mapped_ids - intervention_ids
    in_interventions_not_in_map = intervention_ids - mapped_ids
    
    print(f"\n📊 Mismatch Analysis:")
    print(f"  • IDs in both:          {len(intervention_ids & mapped_ids)}")
    print(f"  • IDs only in map:      {len(in_map_not_in_interventions)}")
    print(f"  • IDs only in database: {len(in_interventions_not_in_map)}")
    
    if in_map_not_in_interventions:
        print(f"\n⚠️  IDs in map but NOT in interventions:")
        for i, missing_id in enumerate(sorted(in_map_not_in_interventions)[:10], 1):
            print(f"    {i}. {missing_id}")
        if len(in_map_not_in_interventions) > 10:
            print(f"    ... and {len(in_map_not_in_interventions) - 10} more")
    
    if in_interventions_not_in_map:
        print(f"\n⚠️  IDs in interventions but NOT in map:")
        for i, missing_id in enumerate(sorted(in_interventions_not_in_map)[:10], 1):
            print(f"    {i}. {missing_id}")
        if len(in_interventions_not_in_map) > 10:
            print(f"    ... and {len(in_interventions_not_in_map) - 10} more")
    
    if in_map_not_in_interventions or in_interventions_not_in_map:
        print(f"\n💡 Solution: Rebuild the FAISS index and ID map")
        print(f"   python scripts/rebuild_faiss.py")
    else:
        print(f"\n✅ No ID mismatches found!")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index and diagnose issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check existing files
  python rebuild_faiss.py --check
  
  # Diagnose ID mismatches
  python rebuild_faiss.py --diagnose
  
  # Rebuild index
  python rebuild_faiss.py
  
  # Rebuild with custom paths
  python rebuild_faiss.py -i data/my_interventions.json -o data/my_artifacts
        """
    )
    
    parser.add_argument(
        "--interventions", "-i",
        default="data/artifacts/interventions.json",
        help="Path to interventions.json"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/artifacts",
        help="Output directory for index and id_map"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Only check existing files, don't rebuild"
    )
    parser.add_argument(
        "--diagnose", "-d",
        action="store_true",
        help="Diagnose ID mismatches between files"
    )
    parser.add_argument(
        "--model", "-m",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_existing_files()
    elif args.diagnose:
        check_existing_files()
        diagnose_mismatch(args.interventions, os.path.join(args.output, "id_map.json"))
    else:
        check_existing_files()
        print()
        success = rebuild_faiss_index(
            args.interventions, 
            args.output,
            args.model,
            args.batch_size
        )
        
        if not success:
            print("\n❌ Rebuild failed. Please check errors above.")
            exit(1)