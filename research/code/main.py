import sys
from exception import mllmException
from captions_pipeline import Captions
from dotenv import load_dotenv
from prompts import questions
load_dotenv()

try:
    mllm_model = "LLAMA"
    images_folder_path = "" # Path to the test images folder
    batch_size = 5
    
    captions = Captions(
        mllm_model=mllm_model, 
        images_folder_path=images_folder_path, 
        batch_size=batch_size,
        questions=questions
    )
    captions.run()
except Exception as e:
    raise mllmException(e, sys)
