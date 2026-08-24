import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from app.config import settings

logger = logging.getLogger(__name__)

_firebase_app = None

def initialize_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    if not os.path.isabs(cred_path):
        # Resolve relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cred_path = os.path.join(base_dir, cred_path)

    if os.path.exists(cred_path):
        logger.info(f"Initializing Firebase Admin SDK using credentials file: {cred_path}")
        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred, {
            'projectId': settings.FIREBASE_PROJECT_ID,
            'storageBucket': settings.FIREBASE_STORAGE_BUCKET
        })
    else:
        logger.warning(f"Credentials file not found at {cred_path}. Initializing default app credentials.")
        _firebase_app = firebase_admin.initialize_app(options={
            'projectId': settings.FIREBASE_PROJECT_ID,
            'storageBucket': settings.FIREBASE_STORAGE_BUCKET
        })

    return _firebase_app

def get_db():
    initialize_firebase()
    return firestore.client()

def get_auth_client():
    initialize_firebase()
    return auth

def get_storage_bucket():
    initialize_firebase()
    return storage.bucket()

def upload_to_storage(local_path: str, blob_path: str, content_type: str = "video/mp4") -> str:
    """Uploads a local file to Firebase Storage and returns public media download URL."""
    import uuid
    import urllib.parse
    try:
        bucket = get_storage_bucket()
        bucket_name = settings.FIREBASE_STORAGE_BUCKET
        token = str(uuid.uuid4())
        
        blob = bucket.blob(blob_path)
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        with open(local_path, "rb") as f:
            blob.upload_from_file(f, content_type=content_type)
            
        encoded_path = urllib.parse.quote(blob_path, safe="")
        return f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media&token={token}"
    except Exception as err:
        logger.warning(f"Firebase Storage upload fallback for {blob_path}: {str(err)}")
        return f"https://storage.googleapis.com/{settings.FIREBASE_STORAGE_BUCKET}/{blob_path}"

