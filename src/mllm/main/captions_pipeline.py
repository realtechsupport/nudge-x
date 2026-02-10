import os
from mllm.captions_generate import Captions
from dotenv import load_dotenv
from mllm.prompts import questions
from mllm.config.settings import mllm_model, USE_NDVI, USE_UDM
from mllm.config import validate_env
load_dotenv()

# Validate required environment variables before running
validate_env()

# Step 1: Generate and save captions only
images_folder_path = os.getenv("IMAGE_DIR")  # Path to the test images folder
batch_size = 2

captions = Captions(
    mllm_model=mllm_model, 
    images_folder_path=images_folder_path, 
    batch_size=batch_size,
    questions=questions,
    use_ndvi=USE_NDVI,
    use_udm=USE_UDM
)
captions.run()
