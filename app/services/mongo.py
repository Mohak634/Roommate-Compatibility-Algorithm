import os
import bcrypt
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

# Load environment variables
load_dotenv()

def connect_to_mongo():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise EnvironmentError("MONGODB_URI not found in .env file.")
    try:
        client = MongoClient(uri, server_api=ServerApi("1"))
        client.admin.command("ping")
        print("✅ Connected to MongoDB.")
        return client
    except Exception as e:
        print("❌ MongoDB connection failed:", str(e))
        return None

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def signup_user(username, email, password):
    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}

    try:
        db = client["dormmate"]
        users_collection = db["users"]

        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)

        user = {
            "username": username,
            "email": email,
            "password_hash": hash_password(password),
            "created_at": datetime.utcnow(),
            "role": "user",
            "profile_complete": False
        }

        users_collection.insert_one(user)
        return {"success": True, "message": "User created successfully."}
    except DuplicateKeyError:
        return {"success": False, "message": "Username or email already exists."}
    except Exception as e:
        return {"success": False, "message": f"Error creating user: {str(e)}"}
    finally:
        client.close()


def login_user(username_or_email, password):
    client = connect_to_mongo()
    if not client:
        return {"success": False, "message": "MongoDB connection failed."}

    try:
        db = client["dormmate"]
        users_collection = db["users"]

        user = users_collection.find_one({
            "$or": [
                {"username": username_or_email},
                {"email": username_or_email}
            ]
        })

        if not user:
            return {"success": False, "message": "User not found."}

        if verify_password(password, user["password_hash"]):
            return {
                "success": True,
                "message": "Login successful.",
                "user": {
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                    "profile_complete": user["profile_complete"]
                }
            }
        else:
            return {"success": False, "message": "Incorrect password."}
    finally:
        client.close()

