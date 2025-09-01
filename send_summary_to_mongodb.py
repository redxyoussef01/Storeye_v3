import json
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime
import os

# ---- Configuration (edit these variables) ----
# MongoDB Configuration
MONGODB_URI = "mongodb+srv://yourais00_db_user:8zQcQOsAWYDeiDHy@coffe.mz4ouct.mongodb.net/"
DATABASE_NAME = "storeyes_analysis"  # Database name
COLLECTION_NAME = "analysis_results"  # Collection name

# Input file path
SUMMARY_JSON_PATH = "./detections_output/objects_classification_output/global_summary.json"

def load_summary_data(json_path):
    """Load data from the global summary JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded summary data from {json_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Summary file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}")
        return None
    except Exception as e:
        print(f"Error loading summary data: {e}")
        return None

def prepare_mongodb_document(summary_data):
    """Prepare the summary data for MongoDB insertion"""
    
    # Extract key information
    analysis_summary = summary_data.get('analysis_summary', {})
    class_distribution = summary_data.get('class_distribution', {})
    video_summaries = summary_data.get('video_summaries', {})
    detailed_results = summary_data.get('detailed_results', [])
    
    # Create MongoDB document
    document = {
        "timestamp": datetime.now(),
        "analysis_date": analysis_summary.get('analysis_timestamp', 'Unknown'),
        "analysis_method": analysis_summary.get('analysis_method', 'Unknown'),
        
        # Overall statistics
        "overall_stats": {
            "total_hand_folders_processed": analysis_summary.get('total_hand_folders_processed', 0),
            "folders_with_objects": analysis_summary.get('folders_with_objects', 0),
            "total_objects_counted": analysis_summary.get('total_objects_counted', 0),
            "average_objects_per_folder": analysis_summary.get('average_objects_per_folder', 0)
        },
        
        # Class distribution
        "class_distribution": class_distribution,
        
        # Video summaries
        "video_summaries": video_summaries,
        
        # Detailed results
        "detailed_results": detailed_results,
        
        # Metadata
        "metadata": {
            "source_file": SUMMARY_JSON_PATH,
            "insertion_timestamp": datetime.now(),
            "version": "1.0"
        }
    }
    
    return document

def connect_to_mongodb():
    """Connect to MongoDB and return client and database objects"""
    try:
        # Connect to MongoDB Atlas with proper connection options
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=10000,          # 10 second socket timeout
            maxPoolSize=10,                 # Connection pool size
            retryWrites=True,               # Enable retry writes
            w='majority'                    # Write concern
        )
        
        # Test the connection
        client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB Atlas")
        print(f"   Connection string: {MONGODB_URI.split('@')[1]}")  # Hide credentials in output
        
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        return client, db, collection
        
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        print(f"   Please check your connection string and network connectivity")
        return None, None, None

def insert_to_mongodb(collection, document):
    """Insert document into MongoDB collection"""
    try:
        # Insert the document
        result = collection.insert_one(document)
        print(f"✅ Successfully inserted document with ID: {result.inserted_id}")
        return True, result.inserted_id
        
    except Exception as e:
        print(f"❌ Error inserting document to MongoDB: {e}")
        return False, None

def create_indexes(collection):
    """Create useful indexes for the collection"""
    try:
        # Create indexes for better query performance
        collection.create_index("timestamp")
        collection.create_index("analysis_date")
        collection.create_index("overall_stats.total_objects_counted")
        collection.create_index("metadata.insertion_timestamp")
        print("✅ Successfully created indexes")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create indexes: {e}")

def reset_collection(collection):
    """Reset/clear the entire collection"""
    try:
        # Delete all documents in the collection
        result = collection.delete_many({})
        print(f"🗑️  Successfully reset collection: deleted {result.deleted_count} documents")
        return True
    except Exception as e:
        print(f"❌ Error resetting collection: {e}")
        return False

def get_collection_stats(collection):
    """Get statistics about the collection"""
    try:
        total_documents = collection.count_documents({})
        print(f"📊 Collection statistics:")
        print(f"   Total documents: {total_documents}")
        
        if total_documents > 0:
            # Get the most recent document
            latest_doc = collection.find_one(
                sort=[("timestamp", -1)]
            )
            if latest_doc:
                latest_time = latest_doc.get("timestamp", "Unknown")
                print(f"   Latest document timestamp: {latest_time}")
        
        return total_documents
    except Exception as e:
        print(f"⚠️  Warning: Could not get collection statistics: {e}")
        return 0

def main():
    """Main function to send summary to MongoDB"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Send analysis summary to MongoDB")
    parser.add_argument("--reset", action="store_true", help="Reset/clear the collection before inserting new data")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics without inserting data")
    args = parser.parse_args()
    
    print("🗄️  Starting MongoDB summary operations...")
    
    # Check configuration
    if not MONGODB_URI:
        print("❌ Error: Please configure MONGODB_URI in the script")
        return
    
    # Connect to MongoDB
    client, db, collection = connect_to_mongodb()
    if not client:
        print("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    try:
        # Show current collection statistics
        print("\n📊 Current collection status:")
        get_collection_stats(collection)
        
        if args.stats_only:
            print("📊 Stats-only mode: No data will be inserted")
            return
        
        # Reset collection if requested
        if args.reset:
            print("\n🗑️  Resetting collection as requested...")
            if reset_collection(collection):
                print("✅ Collection reset successful")
            else:
                print("❌ Collection reset failed. Continuing anyway...")
        
        # Load summary data
        summary_data = load_summary_data(SUMMARY_JSON_PATH)
        if not summary_data:
            print("❌ Failed to load summary data. Exiting.")
            return
        
        # Prepare document for MongoDB
        print("\n📝 Preparing document for MongoDB...")
        document = prepare_mongodb_document(summary_data)
        
        # Insert into MongoDB
        print("📤 Inserting document into MongoDB...")
        success, document_id = insert_to_mongodb(collection, document)
        
        if success:
            print(f"\n🎉 Successfully uploaded analysis results to MongoDB!")
            print(f"   Database: {DATABASE_NAME}")
            print(f"   Collection: {COLLECTION_NAME}")
            print(f"   Document ID: {document_id}")
            
            # Create indexes for better performance
            create_indexes(collection)
            
            # Show updated statistics
            print("\n📊 Updated collection status:")
            get_collection_stats(collection)
            
        else:
            print("❌ Failed to upload analysis results to MongoDB")
            
    except Exception as e:
        print(f"❌ Error during MongoDB operation: {e}")
        
    finally:
        # Close MongoDB connection
        if client:
            client.close()
            print("🔌 MongoDB connection closed")

if __name__ == "__main__":
    print("=" * 60)
    print("🗄️  MongoDB Summary Upload Tool")
    print("=" * 60)
    print("Usage:")
    print("  python send_summary_to_mongodb.py              # Upload summary data")
    print("  python send_summary_to_mongodb.py --reset      # Reset collection and upload")
    print("  python send_summary_to_mongodb.py --stats-only # Show collection stats only")
    print("=" * 60)
    print()
    
    main()
