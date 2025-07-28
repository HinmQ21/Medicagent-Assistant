#!/usr/bin/env python3
"""
Script to reset Qdrant collection for embedding dimension mismatch
"""

import os
import shutil
from pathlib import Path
from config import Config

def reset_qdrant_collection():
    """Reset the Qdrant collection by deleting the local database"""
    config = Config()
    
    # Path to local Qdrant database
    qdrant_path = Path(config.rag.vector_local_path)
    
    print(f"🔍 Checking Qdrant database at: {qdrant_path}")
    
    if qdrant_path.exists():
        print(f"📁 Found existing Qdrant database")
        print(f"🗑️  Deleting: {qdrant_path}")
        
        try:
            # Remove the entire directory
            shutil.rmtree(qdrant_path)
            print("✅ Successfully deleted old Qdrant database")
            
            # Also clean up docs database if it exists
            docs_path = Path(config.rag.doc_local_path)
            if docs_path.exists():
                print(f"🗑️  Also deleting docs database: {docs_path}")
                shutil.rmtree(docs_path)
                print("✅ Successfully deleted old docs database")
                
            # Clean up parsed content directory
            parsed_path = Path(config.rag.parsed_content_dir)
            if parsed_path.exists():
                print(f"🗑️  Also deleting parsed content: {parsed_path}")
                shutil.rmtree(parsed_path)
                print("✅ Successfully deleted old parsed content")
                
        except Exception as e:
            print(f"❌ Error deleting database: {e}")
            return False
    else:
        print("ℹ️  No existing Qdrant database found")
    
    print("\n🎉 Qdrant reset complete!")
    print("💡 You can now run the ingestion script again:")
    print("   python ingest_rag_data.py --file ./data/raw/covid_chest_xray_2024.pdf")
    
    return True

if __name__ == "__main__":
    print("🔄 Qdrant Collection Reset Tool")
    print("=" * 50)
    
    # Ask for confirmation
    response = input("⚠️  This will delete all existing RAG data. Continue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        reset_qdrant_collection()
    else:
        print("❌ Operation cancelled")
