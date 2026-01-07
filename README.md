Instructions to run: (Work in your own user in VM)

Clone repo:
  
   git clone --no-checkout https://github.com/realtechsupport/nudge-x.git  
   Make sure you are in nudge-x Ex: your_username@nudge-x:~/nudge-x$  
   git sparse-checkout init --cone  
   git sparse-checkout set research .env.example .gitignore  
   git checkout main  


Make sure to create your own .env file. The template is found in .env.example file  
Make sure to add your_venv_name into .gitignore  


MacOS/Linux:
1. python3 -m venv your_venv_name
2. source your_venv_name/bin/activate
3. pip install -r requirements.txt


Windows:
1. python -m venv my_venv
2. my_venv\Scripts\activate
3. pip install -r requirements.txt
If killed while installing due to torch, use python -m pip install --index-url https://download.pytorch.org/whl/cpu torch  

Always run any file inside research. 
cd research 

Example: To run captions_pipeline
MacOS:
python3 -m mllm_code.main.captions_pipeline 

Windows: 
python -m mllm_code.main.captions_pipeline 

Use this command in terminal to check number of images in the nundge-bucket
gsutil ls gs://nudge-bucket | grep -E '\.png$|\.jpg$|\.jpeg$' | wc -l
 
Use below command to connect to database using Terminal  

psql -h localhost -U Admin1 -d captions_db  
Enter password: 
The following commands helps to navigate throught the db  
\dt - List tables in the current database (captions_db)
\x -  For expanded display mode (Human redable print)
\q	- Quit the psql session
You can run SQL commands like (always use ; at end);



UI:

Interactive Globe version built using Gemini 2.5 Flash.
Link: https://g.co/gemini/share/4e04b325b8eb 


Reference material for LLM as a judge: 

1. https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
2. https://www.evidentlyai.com/llm-guide/llm-as-a-judge
3. https://towardsdatascience.com/llm-as-a-judge-a-practical-guide/
4. https://arxiv.org/pdf/2411.15594
