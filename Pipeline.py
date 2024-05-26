import os
from clearml import Dataset, Task
from DataPreprocess import preprocess_and_upload_data, load_and_preprocess_image, save_preprocessed_data
from ModelTrain import train_and_log_model
from GitHubUtils import update_model
 
# Set ClearML environment variables
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'I7O2G108OEWA9MU3S2Y5'
os.environ['CLEARML_API_SECRET_KEY'] = 'G5QDGfcv2BvYiorxuiGiQTEPtGLx4ZeZNTEf8GMwg41nTUssoA'
 
def run_pipeline(epochs: int = 5, project_name: str = "AttendanceAI", raw_dataset_name: str = "raw_data",
                 processed_dataset_name: str = "processed_data", env_path: str = "C:/Users/arron/OneDrive/Documents/UTS/Post Graduate/Semester 2/42174 Artificial Intelligence Studio/Assignment/Week 12/.env",
                 repo_url: str = "git@github.com:YourUser/YourRepo.git", branch: str = "LL14-51-Updated-Model-Deployment"):
    # Check if the processed dataset already exists
    try:
        existing_processed_dataset = Dataset.get(
            dataset_project=project_name,
            dataset_name=processed_dataset_name,
            only_completed=True,
            alias="latest"
        )
        print(f"Using existing processed dataset: {existing_processed_dataset.id}")
    except Exception as e:
        print(f"Processed dataset not found: {e}")
        # Preprocess Data Step
        preprocess_and_upload_data(
            raw_dataset_name=raw_dataset_name,
            project_name=project_name,
            processed_dataset_name=processed_dataset_name
        )
 
    # Train Model Step
    model_id = train_and_log_model(
        project_name=project_name,
        task_name="Train Model",
        dataset_project=project_name,
        dataset_name=processed_dataset_name,
        test_size=0.25,
        random_state=42,
        initial_lr=1e-4,
        drop=0.5,
        epochs_drop=10.0,
        num_epochs=epochs,
        batch_size=32,
    )
 
    # Update model in GitHub
    update_model(
        model_id=model_id,
        env_path=env_path,
        repo_url=repo_url,
        branch=branch,
        project_name=project_name
    )
 
    print("Pipeline executed. Check ClearML for progress.")
 
if __name__ == "__main__":
    run_pipeline()
