import os
import json
from typing import List, Tuple
from mllm_code.prompts import system_prompt, questions
from mllm_code.mllm_helper import LlamaPromptGenerator, LlamaCaptionGenerator
from mllm_code.evaluation import CaptionEvaluator
from mllm_code.database_pipeline.database_operations import create_table_if_not_exists, save_filename_and_captions

class Captions:
    """
    Class to generate captions, evaluate them, and save in batches.
    Each saved tuple = (basename, location, caption, is_accepted, is_evaluated).
    """

    def __init__(self, mllm_model: str, images_folder_path: str, questions: List[str], batch_size: int = 5):
        self.mllm_model = mllm_model
        self.images_folder_path = images_folder_path
        self.questions = questions
        self.batch_size = batch_size

        self._model_selector = {
            "LLAMA": self._llama,
            "KOSMOS": self._kosmos
        }

        # Load all images in the folder
        self.image_files = [
            os.path.join(images_folder_path, f)
            for f in os.listdir(images_folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        # Validate that images were found
        if not self.image_files:
            raise ValueError(
                f"No images found in the specified folder!\n"
                f"   Searched in: {images_folder_path}\n"
                f"   Supported formats: .png, .jpg, .jpeg"
            )

    def _save_caption(self, captions_with_metadata: List[Tuple[str, str, str, bool, bool]]):
        """Save a batch of captions with evaluation metadata to DB."""
        create_table_if_not_exists()
        save_filename_and_captions(captions_with_metadata)

    def _batch_iterator(self, iterable, batch_size):
        """Yield successive batches of size `batch_size` from iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def run(self):
        """Run the selected model."""
        if self.mllm_model not in self._model_selector:
            raise ValueError(f"Unsupported model: {self.mllm_model}")
        self._model_selector[self.mllm_model]()

    def _llama(self):
        """Run caption generation with LLaMA model in batches."""
        print("Running LLAMA -4\n")
        model_name = 'meta/llama-4-maverick-17b-128e-instruct'
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...\n")
            captions_with_metadata = []

            for image_file in batch:
                for question in self.questions:
                    print(f"Processing image: {image_file}\n")
                    prompt, location, basename = LlamaPromptGenerator(image_file, question)
                    

                    caption = LlamaCaptionGenerator(image_file, system_prompt, prompt, model_name, invoke_url)
                    print("Caption generated: ", caption)
                    print("\n")
                    print("Evaluating caption\n")
                    # Evaluate caption
                    is_accepted = self.evaluation(caption)
                    is_evaluated = True
                    print("Caption evaluated\n")
                    # Collect metadata
                    captions_with_metadata.append((basename, location, caption, is_accepted, is_evaluated))
            #print("captions_with_metadata", captions_with_metadata)
            # save batch after processing
            self._save_caption(captions_with_metadata)

    def _kosmos(self):
        """Run caption generation with Kosmos-2 model in batches."""
        print("Running Kosmos-2")
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

        for batch in self._batch_iterator(self.image_files, self.batch_size):
            print(f"\nProcessing batch of {len(batch)} images...\n")
            captions_with_metadata = []

            for image_file in batch:
                for question in self.questions:
                    prompt, location, basename = KosmosPromptGenerator(image_file, question)
                    caption = KosmosCaptionGenerator(image_file, prompt, invoke_url)

                    # Evaluate caption
                    is_accepted = self.evaluation(caption)
                    is_evaluated = True

                    # Collect metadata
                    captions_with_metadata.append((basename, location, caption, is_accepted, is_evaluated))

            # save batch after processing
            self._save_caption(captions_with_metadata)

    def evaluation(self, caption: str) -> bool:
        """Evaluate caption and return acceptance flag."""
        try:
            evaluator = CaptionEvaluator(
                gemini_api_key=os.getenv("GOOGLE_API_KEY"),
                anthropic_api_key=""
            )

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
        except Exception as e:
            raise RuntimeError("Evaluation failed") from e

