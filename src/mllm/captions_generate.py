from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
import os
import json
import time
from typing import List, Tuple, Optional
from mllm.prompts import system_prompt, questions, multi_shot_examples
from eo.mllm_helper import (
    LlamaPromptGenerator_mines, LlamaCaptionGenerator, KosmosPromptGenerator,
    KosmosCaptionGenerator, find_matching_auxiliary_images, compress_image,
    has_metadata_for_image,
)
from mllm.evaluation import CaptionEvaluator
from database_pipeline.database_operations import create_table_if_not_exists, save_filename_and_captions, create_pipeline_run
from mllm.config.settings import (
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

    def __init__(
        self,
        mllm_model: str,
        images_folder_path: str,
        questions: List[str],
        batch_size: int = 5,
        prompt_version: str = PROMPT_VERSION,
        use_ndvi: bool = True,
        use_udm: bool = True
    ):
        """
        Initialize the Captions generator.
        
        Args:
            mllm_model: Model to use ("LLAMA" or "KOSMOS")
            images_folder_path: Path to images (local or gs://)
            questions: List of questions to ask about each image
            batch_size: Number of images to process per batch
            prompt_version: Version of prompts to use
            use_ndvi: Whether to include NDVI images if available (default: True)
            use_udm: Whether to include UDM binary classifier images if available (default: True)
        
        Examples:
            # RGB only (ignore NDVI and UDM even if they exist)
            Captions(..., use_ndvi=False, use_udm=False)
            
            # RGB + NDVI only
            Captions(..., use_ndvi=True, use_udm=False)
            
            # RGB + UDM only
            Captions(..., use_ndvi=False, use_udm=True)
            
            # RGB + NDVI + UDM (default)
            Captions(..., use_ndvi=True, use_udm=True)
        """
        self.mllm_model = mllm_model
        self.images_folder_path = images_folder_path
        self.questions = questions
        self.batch_size = batch_size
        self.prompt_version = prompt_version
        self.use_ndvi = use_ndvi
        self.use_udm = use_udm
        self.run_id: Optional[str] = None
        # (filename, image_mode, error_message)
        self.failed_cases: List[Tuple[str, str, str]] = []
        
        # Log the image combination being used
        aux_images = []
        if use_ndvi:
            aux_images.append("NDVI")
        if use_udm:
            aux_images.append("UDM")
        if aux_images:
            print(f"Image mode: RGB + {' + '.join(aux_images)}")
        else:
            print("Image mode: RGB only")

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
            # Store all images for auxiliary lookup
            self.all_image_files = self._list_gcs_images()
            # Filter to only RGB images for primary processing
            self.image_files = self._filter_rgb_images(self.all_image_files)
        else:
            # --- Handle local path ---
            print(f"Detected local path: {images_folder_path}")
            self.all_image_files = [
                os.path.join(images_folder_path, f)
                for f in os.listdir(images_folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            # Filter to only RGB images for primary processing
            self.image_files = self._filter_rgb_images(self.all_image_files)

        # Only process images that have metadata in METADATA_TSV; ignore the rest
        before = len(self.image_files)
        self.image_files = [p for p in self.image_files if has_metadata_for_image(p)]
        skipped = before - len(self.image_files)
        if skipped:
            print(f"Skipped {skipped} RGB image(s) with no metadata. Processing {len(self.image_files)} image(s).")
        if not self.image_files:
            raise ValueError(
                f"No RGB images with metadata found in {images_folder_path}. "
                "Expected naming: minename_rgb_date.png and mine name must match a row in METADATA_TSV."
            )
        
    def _list_gcs_images(self):
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        return [
            blob.name for blob in blobs
            if blob.name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def _filter_rgb_images(self, image_files: List[str]) -> List[str]:
        """
        Filter list of images to only include RGB images.
        RGB images follow naming: minename_rgb_date.png
        """
        rgb_images = []
        for img_path in image_files:
            basename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(basename)[0]
            parts = name_without_ext.split("_")
            # Check if this is an RGB image: minename_rgb_date
            if len(parts) >= 3 and parts[1].lower() == "rgb":
                rgb_images.append(img_path)
        return rgb_images

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
            print(f"Pipeline run created with ID: {self.run_id}")
        else:
            print("Warning: Could not create pipeline run record, continuing without run_id.")
        
        self._model_selector[self.mllm_model]()

    def _llama(self):
        """Run caption generation with LLaMA model in batches."""
        print("Running LLaMA-4")

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...")
            captions_with_metadata = []

            if self.images_folder_path.startswith("gs://"):
                # batch contains blob names (strings), image_objs contains (pil_image, basename) tuples
                image_objs = self._load_images_in_batch(batch)
                for blob_name, (pil_image, basename) in zip(batch, image_objs):
                    # Find auxiliary images (NDVI, UDM) for this RGB image
                    aux_images = find_matching_auxiliary_images(blob_name, self.all_image_files)
                    
                    # Load auxiliary images if found AND enabled
                    ndvi_image = None
                    udm_image = None
                    
                    if self.use_ndvi and aux_images['ndvi']:
                        print(f"  Using NDVI image: {os.path.basename(aux_images['ndvi'])}")
                        ndvi_image, _ = self._load_image_from_gcs(aux_images['ndvi'])
                    
                    if self.use_udm and aux_images['udm']:
                        print(f"  Using UDM overlay: {os.path.basename(aux_images['udm'])}")
                        udm_image, _ = self._load_image_from_gcs(aux_images['udm'])
                    
                    for question in self.questions:
                        print(f"Processing image: {basename}...")
                        # Pass blob_name (string path) to LlamaPromptGenerator for metadata extraction
                        prompt, location, basename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(blob_name, question)
                        # Multi-step fallback logic:
                        # 1) Try with current config (RGB + NDVI/UDM as available) at default quality.
                        # 2) On failure, log and retry with lower quality.
                        # 3) On failure, log and drop UDM (RGB + NDVI).
                        # 4) On failure, drop NDVI as well (RGB only). On failure, log and skip.

                        base_quality = 70
                        low_quality = 50

                        def _attempt_gcs(rgb_img, ndvi_img, udm_img, quality: int) -> Optional[str]:
                            mode_components = ["RGB"]
                            if ndvi_img is not None:
                                mode_components.append("NDVI")
                            if udm_img is not None:
                                mode_components.append("UDM")
                            image_mode_local = "+".join(mode_components)
                            try:
                                return LlamaCaptionGenerator(
                                    rgb_img, system_prompt, prompt,
                                    LLAMA_MODEL_NAME, LLAMA_INVOKE_URL,
                                    LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_MAX_TOKENS, LLAMA_FREQUENCY_PENALTY,
                                    second_image_file_path_or_image=ndvi_img,
                                    third_image_file_path_or_image=udm_img,
                                    first_image_label="RGB",
                                    second_image_label="NDVI",
                                    third_image_label="UDM",
                                    quality=quality,
                                )
                            except Exception as e:
                                error_msg_local = str(e)
                                print(
                                    f"[RUN_ID={self.run_id}] Caption generation FAILED for image '{basename}' "
                                    f"(mode={image_mode_local}): {error_msg_local}"
                                )
                                self.failed_cases.append((basename, image_mode_local, error_msg_local))
                                return None

                        # Step 1: default quality, all available images
                        caption = _attempt_gcs(pil_image, ndvi_image, udm_image, base_quality)
                        if caption is None:
                            # Step 2: lower quality, same set of images
                            caption = _attempt_gcs(pil_image, ndvi_image, udm_image, low_quality)
                        if caption is None and (ndvi_image is not None or udm_image is not None):
                            # Step 3: RGB + NDVI (drop UDM)
                            caption = _attempt_gcs(pil_image, ndvi_image, None, low_quality)
                        if caption is None:
                            # Step 4: RGB only
                            caption = _attempt_gcs(pil_image, None, None, low_quality)
                        if caption is None:
                            # All fallbacks failed; skip this image/question
                            continue

                        print("Evaluating caption...")
                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
                        captions_with_metadata.append((basename, mine_name, location, country, caption, is_accepted, True, question, latitude, longitude))
            # --- Local path flow --- #
            else:
                for image_file in batch:
                    # Find auxiliary images (NDVI, UDM) for this RGB image
                    aux_images = find_matching_auxiliary_images(image_file, self.all_image_files)
                    
                    # Only use auxiliary images if enabled
                    ndvi_path = aux_images['ndvi'] if self.use_ndvi else None
                    udm_path = aux_images['udm'] if self.use_udm else None
                    
                    if ndvi_path:
                        print(f"  Using NDVI image: {os.path.basename(ndvi_path)}")
                    if udm_path:
                        print(f"  Using UDM overlay: {os.path.basename(udm_path)}")
                    
                    for question in self.questions:
                        print(f"Processing image: {image_file}...")
                        prompt, location, basename, country, mine_name, latitude, longitude = LlamaPromptGenerator_mines(image_file, question)

                        base_quality = 70
                        low_quality = 50

                        def _attempt_local(rgb_path: str, ndvi_path_local, udm_path_local, quality: int) -> Optional[str]:
                            mode_components = ["RGB"]
                            if ndvi_path_local:
                                mode_components.append("NDVI")
                            if udm_path_local:
                                mode_components.append("UDM")
                            image_mode_local = "+".join(mode_components)
                            try:
                                return LlamaCaptionGenerator(
                                    rgb_path, system_prompt, prompt,
                                    LLAMA_MODEL_NAME, LLAMA_INVOKE_URL,
                                    LLAMA_TEMPERATURE, LLAMA_TOP_P, LLAMA_MAX_TOKENS, LLAMA_FREQUENCY_PENALTY,
                                    second_image_file_path_or_image=ndvi_path_local,
                                    third_image_file_path_or_image=udm_path_local,
                                    first_image_label="RGB",
                                    second_image_label="NDVI",
                                    third_image_label="UDM",
                                    quality=quality,
                                )
                            except Exception as e:
                                error_msg_local = str(e)
                                print(
                                    f"[RUN_ID={self.run_id}] Caption generation FAILED for image '{image_file}' "
                                    f"(mode={image_mode_local}): {error_msg_local}"
                                )
                                self.failed_cases.append((image_file, image_mode_local, error_msg_local))
                                return None

                        # Step 1: default quality, all available images
                        caption = _attempt_local(image_file, ndvi_path, udm_path, base_quality)
                        if caption is None:
                            # Step 2: lower quality, same set of images
                            caption = _attempt_local(image_file, ndvi_path, udm_path, low_quality)
                        if caption is None and (ndvi_path or udm_path):
                            # Step 3: RGB + NDVI (drop UDM)
                            caption = _attempt_local(image_file, ndvi_path, None, low_quality)
                        if caption is None:
                            # Step 4: RGB only
                            caption = _attempt_local(image_file, None, None, low_quality)
                        if caption is None:
                            # All fallbacks failed; skip this image/question
                            continue

                        print("Evaluating caption...")
                        is_accepted = self.evaluation(caption)
                        # Tuple: (filename, mine_name, location, country, caption, is_accepted, is_evaluated, question, latitude, longitude)
                        captions_with_metadata.append((basename, mine_name, location, country, caption, is_accepted, True, question, latitude, longitude))

            self._save_caption(captions_with_metadata)

        if self.failed_cases:
            print("\nThe following caption generations failed and were skipped:")
            for filename, image_mode, error in self.failed_cases:
                print(
                    f"- RUN_ID={self.run_id}, image={filename}, mode={image_mode}, reason={error}"
                )

            # Persist failures to a log file under data/logs so that it is not
            # ignored by the global *.log rule in .gitignore.
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                logs_dir = os.path.join(project_root, "data", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                run_tag = self.run_id or "unknown_run"
                log_path = os.path.join(logs_dir, f"caption_failures_{run_tag}.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Caption generation failures for run_id={self.run_id}\n")
                    for filename, image_mode, error in self.failed_cases:
                        f.write(f"image={filename}\tmode={image_mode}\treason={error}\n")
                print(f"Failure log written to: {log_path}")
            except Exception as e:
                print(f"Warning: could not write failures log file: {e}")

    def _kosmos(self):
        print("Running Kosmos-2")
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...")
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
                print(f"Evaluation decision: {result['decision_text']}")
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
