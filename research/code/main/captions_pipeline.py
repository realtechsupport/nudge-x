import sys
from exception import mllmException
from captions_generate import Captions
from dotenv import load_dotenv
from prompts import questions
load_dotenv()

try:
    # Step 1: Generate and save captions only
    mllm_model = "LLAMA"
    images_folder_path = "/Users/saikrishna/Desktop/MLLM/research/data/test"  # Path to the test images folder
    batch_size = 2
    
    captions = Captions(
        mllm_model=mllm_model, 
        images_folder_path=images_folder_path, 
        batch_size=batch_size,
        questions=questions
    )
    captions.run()
except Exception as e:
    raise mllmException(e, sys)







