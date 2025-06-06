import sqlite3
from typing import List, Dict, Any

def get_user_interactions(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all interactions for a specific user.
    
    Args:
        user_id (str): The ID of the user to retrieve interactions for
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing interaction data
        Each dictionary contains: ID, UserID, Channel, and Interaction
    """
    try:
        # Connect to the database
        conn = sqlite3.connect("customer_data.db")
        cursor = conn.cursor()
        
        # Query to get all interactions for the specified user
        query = """
        SELECT ID, UserID, Channel, Interaction 
        FROM user_interactions 
        WHERE UserID = ?
        ORDER BY ID
        """
        
        cursor.execute(query, (user_id,))
        interactions = cursor.fetchall()
        
        # Convert the results to a list of dictionaries
        result = []
        for interaction in interactions:
            result.append({
                'ID': interaction[0],
                'UserID': interaction[1],
                'Channel': interaction[2],
                'Interaction': interaction[3]
            })
        
        return result
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_user_transactions(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all transactions for a specific user.
    
    Args:
        user_id (str): The ID of the user to retrieve transactions for
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing transaction data
        Each dictionary contains: ID, UserID, Type, Amount, Date, ToPerson, 
        AccountNumber, and Description
    """
    try:
        # Connect to the database
        conn = sqlite3.connect("customer_data.db")
        cursor = conn.cursor()
        
        # Query to get all transactions for the specified user
        query = """
        SELECT ID, UserID, Type, Amount, Date, ToPerson, AccountNumber, Description 
        FROM user_transactions 
        WHERE UserID = ?
        ORDER BY Date
        """
        
        cursor.execute(query, (user_id,))
        transactions = cursor.fetchall()
        
        # Convert the results to a list of dictionaries
        result = []
        for transaction in transactions:
            result.append({
                'ID': transaction[0],
                'UserID': transaction[1],
                'Type': transaction[2],
                'Amount': transaction[3],
                'Date': transaction[4],
                'ToPerson': transaction[5],
                'AccountNumber': transaction[6],
                'Description': transaction[7]
            })
        
        return result
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Example usage
if __name__ == "__main__":
    # Example: Get interactions for user U001
    user_id = "U001"
    
    # Get interactions
    interactions = get_user_interactions(user_id)
    print(f"\nInteractions for user {user_id}:")
    for interaction in interactions:
        print(f"Channel: {interaction['Channel']}, Interaction: {interaction['Interaction']}")
    
    # Get transactions
    transactions = get_user_transactions(user_id)
    print(f"\nTransactions for user {user_id}:")
    for transaction in transactions:
        print(f"Date: {transaction['Date']}, Type: {transaction['Type']}, Amount: ${transaction['Amount']}, Description: {transaction['Description']}")