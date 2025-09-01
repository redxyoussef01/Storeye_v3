# MongoDB Integration for Storeyes Analysis

This document explains how to use the MongoDB integration to store analysis results instead of sending emails.

## Overview

The `send_summary_to_mongodb.py` script reads the analysis results from the JSON file and stores them in a MongoDB database for later retrieval and analysis.

## Prerequisites

1. **MongoDB Installation**: Make sure MongoDB is installed and running on your system
2. **Python Dependencies**: Install the required packages

## Installation

1. Install the MongoDB dependencies:
```bash
pip install -r requirements_mongodb.txt
```

2. Or install manually:
```bash
pip install pymongo dnspython
```

## Configuration

Edit the configuration variables in `send_summary_to_mongodb.py`:

```python
# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017/"  # Change to your MongoDB connection string
DATABASE_NAME = "storeyes_analysis"  # Database name
COLLECTION_NAME = "analysis_results"  # Collection name
```

### MongoDB Connection String Examples

- **Local MongoDB**: `mongodb://localhost:27017/`
- **MongoDB Atlas**: `mongodb+srv://username:password@cluster.mongodb.net/`
- **MongoDB with Authentication**: `mongodb://username:password@localhost:27017/`

## Usage

Run the script to upload analysis results to MongoDB:

```bash
python send_summary_to_mongodb.py
```

## Database Structure

The script creates documents with the following structure:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "analysis_date": "2024-01-01",
  "analysis_method": "intersection_analysis",
  "overall_stats": {
    "total_hand_folders_processed": 100,
    "folders_with_objects": 75,
    "total_objects_counted": 500,
    "average_objects_per_folder": 5.0
  },
  "class_distribution": {
    "object1": 200,
    "object2": 150,
    "object3": 150
  },
  "video_summaries": {
    "video1": {
      "total_objects_count": 100,
      "total_hand_folders": 20,
      "hand_folders_with_objects": 15
    }
  },
  "detailed_results": [
    {
      "folder_name": "folder1",
      "object_count": 10,
      "images_total": 50,
      "images_with_objects": 25
    }
  ],
  "metadata": {
    "source_file": "./detections_output/objects_classification_output/global_summary.json",
    "insertion_timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0"
  }
}
```

## Indexes

The script automatically creates the following indexes for better query performance:

- `timestamp`: For time-based queries
- `analysis_date`: For date-based filtering
- `overall_stats.total_objects_counted`: For object count queries
- `metadata.insertion_timestamp`: For insertion time queries

## Querying the Data

You can query the stored data using MongoDB commands or tools like MongoDB Compass:

### Example Queries

1. **Get all analysis results**:
```javascript
db.analysis_results.find()
```

2. **Get results from a specific date**:
```javascript
db.analysis_results.find({"analysis_date": "2024-01-01"})
```

3. **Get results with more than 100 objects**:
```javascript
db.analysis_results.find({"overall_stats.total_objects_counted": {$gt: 100}})
```

4. **Get the latest analysis**:
```javascript
db.analysis_results.find().sort({"timestamp": -1}).limit(1)
```

5. **Get analysis summary by date**:
```javascript
db.analysis_results.aggregate([
  {
    $group: {
      _id: "$analysis_date",
      total_objects: {$sum: "$overall_stats.total_objects_counted"},
      avg_objects_per_folder: {$avg: "$overall_stats.average_objects_per_folder"}
    }
  }
])
```

## Integration with Pipeline

To integrate this with your analysis pipeline, you can:

1. **Replace the email sending** in your main pipeline with MongoDB upload
2. **Add it as an additional step** after email sending
3. **Use it for backup** of analysis results

### Example Integration

```python
# In your main pipeline script
from send_summary_to_mongodb import main as upload_to_mongodb

# After analysis is complete
if analysis_successful:
    upload_to_mongodb()
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure MongoDB is running and the connection string is correct
2. **Authentication Error**: Check username/password in the connection string
3. **Permission Error**: Ensure the MongoDB user has write permissions to the database

### Debug Mode

To enable debug logging, you can modify the script to include more detailed error messages:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

1. **Use Environment Variables** for sensitive information:
```python
import os
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
```

2. **Use MongoDB Atlas** for production environments
3. **Enable MongoDB authentication** for security
4. **Use SSL/TLS** for encrypted connections

## Performance Tips

1. **Use indexes** (already implemented in the script)
2. **Batch inserts** for multiple documents
3. **Use connection pooling** for high-frequency operations
4. **Monitor database size** and implement data retention policies
