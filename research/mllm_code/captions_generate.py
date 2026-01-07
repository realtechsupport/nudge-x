from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
import os
import json
import time
from typing import List, Tuple, Optional
from mllm_code.prompts import system_prompt, questions, multi_shot_examples
from mllm_code.mllm_helper import *
from mllm_code.evaluation import CaptionEvaluator
from mllm_code.database_pipeline.database_operations import create_table_if_not_exists, save_filename_and_captions, create_pipeline_run
from mllm_code.config.settings import (
    LLAMA_MODEL_NAME, LLAMA_INVOKE_URL, LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_MAX_TOKENS, LLAMA_FREQUENCY_PENALTY,
    PROMPT_VERSION
)

# Determine number of shots from the multi_shot_examples
def _count_shots(examples_str: str) -> int:
    """Count the number of Question/Answer pairs in multi-shot examples."""
    if not examples_str:
        return 0
    return examples_str.count("Question:")

class Captions:
    """
    Class to generate captions, evaluate them, and save in batches.
    Each saved tuple = (basename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude).
    """

    def __init__(self, mllm_model: str, images_folder_path: str, questions: List[str], batch_size: int = 5, prompt_version: str = PROMPT_VERSION):
        self.mllm_model = mllm_model
        self.images_folder_path = images_folder_path
        self.questions = questions
        self.batch_size = batch_size
        self.prompt_version = prompt_version
        self.run_id: Optional[str] = None

        self._model_selector = {
            "LLAMA": self._llama,
            "KOSMOS": self._kosmos
        }

        if images_folder_path.startswith("gs://"):
            # --- Handle GCS prefix ---
            print(f"Detected GCS path: {images_folder_path}")
            self.client = storage.Client()
            self.bucket_name = images_folder_path.split("/")[2]
            self.bucket = self.client.bucket(self.bucket_name)
            self.prefix = "/".join(images_folder_path.split("/")[3:])
            self.image_files = self._list_gcs_images()
        else:
            # --- Handle local path ---
            print(f"Detected local path: {images_folder_path}")
            self.image_files = [
                os.path.join(images_folder_path, f)
                for f in os.listdir(images_folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        if not self.image_files:
            raise ValueError(f"No images found in {images_folder_path}")
        
    def _list_gcs_images(self):
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        return [
            blob.name for blob in blobs
            if blob.name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def _load_image_from_gcs(self, blob_name: str):
        """Download a single image as PIL.Image."""
        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        return Image.open(BytesIO(image_bytes)), os.path.basename(blob_name)

    def _load_images_in_batch(self, blob_names, max_workers=8):
        """Parallel download of multiple GCS images."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._load_image_from_gcs, blob_names))
        return results

    def _save_caption(self, captions_with_metadata: List[Tuple[str, str, str, str, str, bool, bool, str, Optional[float], Optional[float]]]):
        """Save a batch of captions with evaluation metadata to DB."""
        create_table_if_not_exists()
        save_filename_and_captions(captions_with_metadata, run_id=self.run_id)

    def _batch_iterator(self, iterable, batch_size):
        """Yield successive batches of size `batch_size` from iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def run(self):
        """Run the selected model."""
        if self.mllm_model not in self._model_selector:
            raise ValueError(f"Unsupported model: {self.mllm_model}")
        
        # Create pipeline run record before starting
        create_table_if_not_exists()
        num_shots = _count_shots(multi_shot_examples)
        
        self.run_id = create_pipeline_run(
            prompt_version=self.prompt_version,
            model_name=LLAMA_MODEL_NAME if self.mllm_model == "LLAMA" else "kosmos-2",
            temperature=LLAMA_TEMPERATURE,
            frequency_penalty=LLAMA_FREQUENCY_PENALTY,
            top_p=LLAMA_TOP_P,
            num_shots=num_shots
        )
        
        if self.run_id:
            print(f"✅ Pipeline run created with ID: {self.run_id}")
        else:
            print("⚠️ Warning: Could not create pipeline run record, continuing without run_id")
        
        self._model_selector[self.mllm_model]()

    def _llama(self):
        """Run caption generation with LLaMA model in batches."""
        print("Running LLAMA -4\n")

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...\n")
            captions_with_metadata = []

            if self.images_folder_path.startswith("gs://"):
                # batch contains blob names (strings), image_objs contains (pil_image, basename) tuples
                image_objs = self._load_images_in_batch(batch)
                for blob_name, (pil_image, basename) in zip(batch, image_objs):
                    for question in self.questions:
                        print(f"Processing image: {basename}")
                        # Pass blob_name (string path) to LlamaPromptGenerator for metadata extraction
                        prompt, location, basename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(blob_name, question)
                        # Pass pil_image to LlamaCaptionGenerator for image processing
                        caption = LlamaCaptionGenerator(
                            pil_image, system_prompt, prompt,
                            LLAMA_MODEL_NAME, LLAMA_INVOKE_URL,
                            LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_MAX_TOKENS, LLAMA_FREQUENCY_PENALTY
                        )

                        print("Caption generated: ", caption)
                        print("Evaluating caption...")
                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
                        captions_with_metadata.append((basename, mine_name, location, country, caption, is_accepted, True, question, latitude, longitude))
            # --- Local path flow --- #
            else:
                for image_file in batch:
                    for question in self.questions:
                        print(f"Processing image: {image_file}")
                        prompt, location, basename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(image_file, question)
                        caption = LlamaCaptionGenerator(
                            image_file, system_prompt, prompt,
                            LLAMA_MODEL_NAME, LLAMA_INVOKE_URL,
                            LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_MAX_TOKENS, LLAMA_FREQUENCY_PENALTY
                        )

                        print("Caption generated: ", caption)
                        print("Evaluating caption...")
                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
                        captions_with_metadata.append((basename, mine_name, location, country, caption, is_accepted, True, question, latitude, longitude))

            
            self._save_caption(captions_with_metadata)

    def _kosmos(self):
        print("Running Kosmos-2\n")
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...\n")
            captions_with_metadata = []

            if self.images_folder_path.startswith("gs://"):
                image_objs = self._load_images_in_batch(batch)
                for pil_image, basename in image_objs:
                    for question in self.questions:
                        prompt, location, _ = KosmosPromptGenerator(pil_image, question)
                        caption = KosmosCaptionGenerator(pil_image, prompt, invoke_url)

                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question)
                        captions_with_metadata.append((basename, None, location, None, caption, is_accepted, True, question))
            else:
                for image_file in batch:
                    for question in self.questions:
                        prompt, location, basename = KosmosPromptGenerator(image_file, question)
                        caption = KosmosCaptionGenerator(image_file, prompt, invoke_url)

                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question)
                        captions_with_metadata.append((basename, None, location, None, caption, is_accepted, True, question))
            self._save_caption(captions_with_metadata)

    def evaluation(self, caption: str, max_retries: int = 3) -> bool:
        """Evaluate caption and return acceptance flag with retry logic."""
        evaluator = CaptionEvaluator(
            gemini_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=""
        )
        
        for attempt in range(max_retries):
            try:
                result = evaluator.evaluate(
                    caption=caption,
                    model="gemini",
                    weights={
                        "Environmental_Focus": 1/5,
                        "Specificity_Terminology": 1/5,
                        "Processes_Patterns": 1/5,
                        "Adherence_to_Constraints": 1/5,
                        "Conciseness": 1/5  
                    },
                    threshold=3.0
                )
                required_keys = {"scores", "decision"}
                if not result or not required_keys.issubset(result.keys()):
                    raise RuntimeError(f"Incomplete evaluation result: {result}")
                print("scores: \n", json.dumps(result["scores"], indent=4))
                print("decision: ", result["decision"], "\n")
                return bool(result["decision"])
                
            except (KeyError, RuntimeError, ValueError, AttributeError) as e:
                # Retry on KeyError (missing fields in JSON), RuntimeError (invalid JSON structure), 
                # ValueError (empty/invalid response), or AttributeError (module errors)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Evaluation attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, raise the error
                    print(f"Evaluation failed after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Evaluation failed after {max_retries} retries") from e
            except Exception as e:
                # For other unexpected errors, don't retry
                print(f"Unexpected evaluation error: {e}")
                raise RuntimeError("Evaluation failed") from e
