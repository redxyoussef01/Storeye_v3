#!/usr/bin/env python3
"""
Test script for MongoDB connection and basic operations
"""

from pymongo import MongoClient
import json
from datetime import datetime

# Configuration
MONGODB_URI = "mongodb+srv://yourais00_db_user:8zQcQOsAWYDeiDHy@coffe.mz4ouct.mongodb.net/"
DATABASE_NAME = "storeyes_analysis"
COLLECTION_NAME = "analysis_results"

def test_connection():
    """Test MongoDB connection"""
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
        print("✅ MongoDB Atlas connection successful!")
        print(f"   Connection string: {MONGODB_URI.split('@')[1]}")  # Hide credentials in output
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"   Please check your connection string and network connectivity")
        return None

def test_database_operations(client):
    """Test basic database operations"""
    if not client:
        return False
    
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Test document insertion
        test_document = {
            "test": True,
            "timestamp": datetime.now(),
            "message": "This is a test document",
            "data": {
                "sample": "value",
                "number": 42
            }
        }
        
        result = collection.insert_one(test_document)
        print(f"✅ Test document inserted with ID: {result.inserted_id}")
        
        # Test document retrieval
        retrieved_doc = collection.find_one({"_id": result.inserted_id})
        if retrieved_doc:
            print("✅ Test document retrieved successfully")
            print(f"   Message: {retrieved_doc.get('message')}")
            print(f"   Number: {retrieved_doc.get('data', {}).get('number')}")
        
        # Test document deletion
        delete_result = collection.delete_one({"_id": result.inserted_id})
        if delete_result.deleted_count > 0:
            print("✅ Test document deleted successfully")
        
        # Test collection count
        total_docs = collection.count_documents({})
        print(f"✅ Collection contains {total_docs} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations failed: {e}")
        return False

def list_databases(client):
    """List all databases"""
    try:
        databases = client.list_database_names()
        print(f"📋 Available databases: {databases}")
        return databases
    except Exception as e:
        print(f"❌ Failed to list databases: {e}")
        return []

def list_collections(client, db_name):
    """List collections in a database"""
    try:
        db = client[db_name]
        collections = db.list_collection_names()
        print(f"📋 Collections in {db_name}: {collections}")
        return collections
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")
        return []

def main():
    """Main test function"""
    print("🧪 Testing MongoDB Connection and Operations")
    print("=" * 50)
    
    # Test connection
    client = test_connection()
    if not client:
        print("❌ Cannot proceed without MongoDB connection")
        return
    
    # List databases
    print("\n📊 Database Information:")
    databases = list_databases(client)
    
    # List collections in our database
    if DATABASE_NAME in databases:
        collections = list_collections(client, DATABASE_NAME)
    else:
        print(f"📝 Database '{DATABASE_NAME}' will be created when first document is inserted")
    
    # Test operations
    print("\n🔧 Testing Database Operations:")
    success = test_database_operations(client)
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! MongoDB is ready to use.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    # Close connection
    client.close()
    print("🔌 MongoDB connection closed")

if __name__ == "__main__":
    main()
