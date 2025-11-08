"""
Database module for storing prediction results.
Uses SQLite for simple persistence.
"""
import sqlite3
from datetime import datetime
from src import config


def init_database(db_path=None):
    """
    Initialize the SQLite database and create tables if they don't exist.
    
    Args:
        db_path: Path to the database file
    
    Returns:
        SQLite connection object
    """
    if db_path is None:
        db_path = config.DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp DATETIME NOT NULL,
        image_path TEXT,
        probabilities TEXT
    )
    ''')
    
    conn.commit()
    print(f"Database initialized at: {db_path}")
    
    return conn


def save_prediction(filename, predicted_class, confidence, 
                image_path=None, probabilities=None, db_path=None):
    """
    Save a prediction result to the database.
    
    Args:
        filename: Name of the image file
        predicted_class: Predicted class label
        confidence: Confidence score (0-1)
        image_path: Full path to the image file (optional)
        probabilities: Dictionary of class probabilities (optional)
     db_path: Path to the database file
    
    Returns:
        ID of the inserted record
    """
    if db_path is None:
        db_path = config.DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    prob_str = str(probabilities) if probabilities else None
    
    cursor.execute('''
   INSERT INTO predictions (filename, predicted_class, confidence, timestamp, image_path, probabilities)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, predicted_class, confidence, timestamp, image_path, prob_str))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id


def get_all_predictions(db_path=None, limit=None):
    """
    Retrieve all predictions from the database.
    
    Args:
        db_path: Path to the database file
      limit: Maximum number of records to retrieve
    
    Returns:
        List of prediction records
    """
    if db_path is None:
        db_path = config.DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = 'SELECT * FROM predictions ORDER BY timestamp DESC'
    if limit:
        query += f' LIMIT {limit}'
    
    cursor.execute(query)
    records = cursor.fetchall()
    conn.close()
    
    return records


def get_predictions_by_class(predicted_class, db_path=None):
    """
    Retrieve predictions for a specific class.
    
    Args:
        predicted_class: Class label to filter by
        db_path: Path to the database file
 
    Returns:
   List of prediction records
    """
    if db_path is None:
     db_path = config.DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
      SELECT * FROM predictions 
        WHERE predicted_class = ?
        ORDER BY timestamp DESC
    ''', (predicted_class,))
    
    records = cursor.fetchall()
    conn.close()
    
    return records


def get_statistics(db_path=None):
    """
    Get statistics about predictions.
    
    Args:
        db_path: Path to the database file
    
    Returns:
      Dictionary with statistics
    """
    if db_path is None:
        db_path = config.DB_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
 # Total predictions
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    
 # Predictions by class
    cursor.execute('''
     SELECT predicted_class, COUNT(*) as count 
 FROM predictions 
  GROUP BY predicted_class
    ''')
    by_class = dict(cursor.fetchall())
    
    # Average confidence
    cursor.execute('SELECT AVG(confidence) FROM predictions')
    avg_confidence = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_predictions': total,
        'predictions_by_class': by_class,
        'average_confidence': avg_confidence if avg_confidence else 0
    }


def clear_database(db_path=None):
    """
    Clear all records from the predictions table.
    
    Args:
        db_path: Path to the database file
    """
    if db_path is None:
        db_path = config.DB_PATH
  
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    
    print("Database cleared!")


if __name__ == "__main__":
    # Test database functionality
    print("=" * 50)
    print("TESTING DATABASE")
    print("=" * 50)
    
    # Initialize database
    init_database()
    
    # Test saving a prediction
    record_id = save_prediction(
        filename="test_image.jpg",
        predicted_class="robot",
        confidence=0.95,
        probabilities={'human': 0.05, 'robot': 0.95}
    )
    print(f"\nSaved prediction with ID: {record_id}")
    
    # Get all predictions
    predictions = get_all_predictions()
    print(f"\nTotal predictions in database: {len(predictions)}")
  
  # Get statistics
    stats = get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  By class: {stats['predictions_by_class']}")
    print(f"  Average confidence: {stats['average_confidence']:.2%}")
