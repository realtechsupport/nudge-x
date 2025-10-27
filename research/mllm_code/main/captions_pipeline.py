import sys
import os
from mllm_code.exception import mllmException
from mllm_code.captions_generate import Captions
from dotenv import load_dotenv
from mllm_code.prompts import questions
from mllm_code.config.settings import mllm_model
load_dotenv()


# Step 1: Generate and save captions only
images_folder_path = os.getenv("IMAGE_DIR") # Path to the test images folder
batch_size = 2

captions = Captions(
    mllm_model=mllm_model, 
    images_folder_path=images_folder_path, 
    batch_size=batch_size,
    questions=questions
)
captions.run()








