from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="runs/detect/train-7/weights/best.pt",
    path_in_repo="best.pt",
    repo_id="Khalil200383/RoadGuard-AI",
    repo_type="model"
)

