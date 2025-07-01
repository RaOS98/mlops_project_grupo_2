import os
import boto3

from src.utils import DEFAULT_BUCKET

def save_processed_data(df, is_train: bool):
    output_dir = f"data/processed/{'train' if is_train else 'oot'}"
    # Choose filename
    filename = f"clean_{'train' if is_train else 'oot'}_data.csv"
    local_path = os.path.join(output_dir, filename)

    # Ensure local directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save locally
    df.to_csv(local_path, index=False)
    print(f"Data saved locally at: {local_path}")

    # Save to S3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, DEFAULT_BUCKET, local_path)
    print(f"Data uploaded to S3 at: s3://{DEFAULT_BUCKET}/{local_path}")

    return local_path  # optionally return full local path
