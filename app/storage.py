from google.cloud import storage

BUCKET_NAME = 'soovik-documents'

def download_from_gcs(dest_name: str, local_path: str):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(dest_name)
        blob.download_to_filename(local_path)
        print(f"Downloaded {dest_name} to {local_path}")
    except Exception as e:
        print(f"Error downloading {dest_name} from GCS: {e}")

def upload_to_gcs(file_path: str, dest_name: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(file_path)
    print(f"Uploaded {file_path} to gs://{BUCKET_NAME}/{dest_name}")