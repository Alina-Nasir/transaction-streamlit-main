"""
MySQL Validation Script for Installer
This script validates MySQL credentials and creates the database if needed.
"""
import sys
import json

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    print("ERROR: mysql-connector-python is not installed")
    sys.exit(1)


def validate_and_setup_mysql(host, port, database, user, password):
    """
    Validate MySQL credentials and create database if it doesn't exist.
    Returns: 0 on success, 1 on failure
    """
    try:
        # Test connection to MySQL server (without specifying database)
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password
        )
        
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"SUCCESS: Database '{database}' is ready")
        
        conn.close()
        
        # Now test connection to the specific database
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            database=database,
            user=user,
            password=password
        )
        
        print(f"SUCCESS: Connected to MySQL database '{database}' successfully")
        conn.close()
        
        return 0
        
    except MySQLError as e:
        print(f"ERROR: MySQL connection failed - {str(e)}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error - {str(e)}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("ERROR: Invalid arguments")
        print("Usage: validate_mysql.py <host> <port> <database> <user> <password>")
        sys.exit(1)
    
    host = sys.argv[1]
    port = sys.argv[2]
    database = sys.argv[3]
    user = sys.argv[4]
    password = sys.argv[5]
    
    exit_code = validate_and_setup_mysql(host, port, database, user, password)
    sys.exit(exit_code)
