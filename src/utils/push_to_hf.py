import os
import argparse
import yaml
from pathlib import Path
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

def push_to_hf(model_repo_id=None, dataset_repo_id=None, token=None, config_path="config.yml"):
    """
    Pushes models and data to their respective Hugging Face Hub repositories.
    """
    # Load from .env if it exists
    load_dotenv()
    
    token = token or os.getenv("HF_TOKEN")
    model_repo_id = model_repo_id or os.getenv("HF_MODEL_REPO_ID") or os.getenv("HF_REPO_ID")
    dataset_repo_id = dataset_repo_id or os.getenv("HF_DATASET_REPO_ID")

    if not token:
        print("Warning: HF_TOKEN not found. You might need to login via 'huggingface-cli login'.")
    else:
        login(token=token)
    
    api = HfApi()
    
    # Load config to get data paths
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Pushing Models and Config
    if model_repo_id:
        print(f"\n--- Uploading to Model Repository: {model_repo_id} ---")
        try:
            api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Error creating/accessing model repo: {e}")
        
        model_files = [
            (config_path, config_path),
            (config.get("model_checkpoint_path"), config.get("model_checkpoint_path")),
        ]
        # Add model checkpoint config if it exists
        if config.get("model_checkpoint_path"):
            checkpoint_dir = Path(config.get("model_checkpoint_path")).parent
            checkpoint_config = checkpoint_dir / "config.yml"
            if checkpoint_config.exists():
                model_files.append((str(checkpoint_config), str(checkpoint_config)))

        # Add specific processed data files as requested by user
        processed_data_files = [
            "data/processed/emotion_artist_table.pkl",
            "data/processed/user_emotion_table.pkl"
        ]
        for pdf in processed_data_files:
            if os.path.exists(pdf):
                # Upload to a folder named 'processed_data' in the repo
                path_in_repo = f"processed_data/{os.path.basename(pdf)}"
                model_files.append((pdf, path_in_repo))

        for local_path, repo_path in model_files:
            if local_path and os.path.exists(local_path):
                print(f"Uploading {local_path} to model repo as {repo_path}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=model_repo_id,
                    repo_type="model",
                )
    else:
        print("\nSkipping Model upload (HF_MODEL_REPO_ID not set).")

    # 2. Pushing Data
    if dataset_repo_id:
        print(f"\n--- Uploading to Dataset Repository: {dataset_repo_id} ---")
        try:
            api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Error creating/accessing dataset repo: {e}")
        
        data_files = [
            config.get("data_train_path"),
            config.get("data_val_path"),
            config.get("raw_data_path"),
            "data/processed/merged_data.parquet",
            "data/processed/spotify_va.parquet",
        ]

        for f_path in data_files:
            if f_path and os.path.exists(f_path):
                print(f"Uploading {f_path} to dataset repo...")
                api.upload_file(
                    path_or_fileobj=f_path,
                    path_in_repo=f_path,
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                )
    else:
        print("\nSkipping Dataset upload (HF_DATASET_REPO_ID not set).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push models and data to Hugging Face Hub")
    parser.add_argument("--model_repo", type=str, help="Hugging Face Model repository ID")
    parser.add_argument("--dataset_repo", type=str, help="Hugging Face Dataset repository ID")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the root config.yml file")
    
    args = parser.parse_args()
    
    push_to_hf(args.model_repo, args.dataset_repo, args.token, args.config)
