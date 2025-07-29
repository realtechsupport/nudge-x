class Captions:
    """
    Class to generate captions and save.
    """
    def __init__(self, mllm_model: str, image_file_path: str):
        self.mllm_model = mllm_model
        self.image_file_path = image_file_path

        self._model_selector = {
            "LLAMA": self._llama,
            "KOSMOS": self._kosmos
        }
    
    def _save_caption(self, caption, accept: bool):
      connect_db()
      create_table_if_not_exists()
      save_image_and_captions(self.image_file_path, caption, accept)

    def run(self):
      if self.mllm_model not in self._model_selector:
          raise ValueError(f"Unsupported model: {self.mllm_model}")
      self._model_selector[self.mllm_model]()

    def _llama(self):
        print("Running LLAMA -4")
        model_name = 'meta/llama-4-maverick-17b-128e-instruct'
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        for question in questions:
            prompt = LlamaPromptGenerator(self.image_file_path, question)
            print(prompt)
            caption = LlamaCaptionGenerator(self.image_file_path, SYSTEM_PROMPT, prompt, model_name, invoke_url)
            print(caption)
            print("-----------------------------------------------------------------" * 2)

    def _kosmos(self):
        print("Running Kosmos-2")
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"

        for question in questions:
            prompt = KosmosPromptGenerator(location, common_prompt, question)
            print(prompt)
            caption = KosmosCaptionGenerator(self.image_file_path, prompt, invoke_url)
            print(caption)
            print("-----------------------------------------------------------------" * 2)


    def evaluation(self):
      evaluator = CaptionEvaluator(
      gemini_api_key= os.getenv("GOOGLE_API_KEY"),
      anthropic_api_key=""
      )

      result = evaluator.evaluate(
      caption=caption,
      model="gemini",
      weights={ # weights are adjustable paramters and can be adjusted based on the requirement.
          "Environmental_Focus": 0.3,
          "Scientific_Accuracy_Plausibility": 0.3,
          "Specificity_Terminology": 0.2,
          "Processes_Patterns_Changes": 0.1,
          "Adherence_to_Constraints": 0.05,
          "Conciseness": 0.05
      },
      threshold=2.0

      )
      is_accepted = bool(result)
      _save_caption(caption, is_accepted = is_accepted)


