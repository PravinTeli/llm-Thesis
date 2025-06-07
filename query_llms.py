import ollama
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import logging
import sys
import time
from tqdm import tqdm

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('file.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_questions(json_file: str) -> List[Dict]:
    """
    Load and flatten categorized questions from JSON file.
    
    Returns:
        List of questions with category information added
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    for category, category_questions in data['categorized_questions'].items():
        for question in category_questions:
            question['category'] = category
            questions.append(question)
    questions.sort(key=lambda x: x['id'])
    return questions

models = ["qwen", "deepseek-r1", "mistral", "llama3.2", "gemma", "phi"]

def warm_up_model(model: str, max_retries: int = 3) -> bool:
    """
    Warm up a model with a simple query to ensure it's loaded.
    """
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = ollama.generate(model=model, prompt="Hi", options={"num_predict": 1})
            if response and "response" in response:
                elapsed = time.time() - start_time
                return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return False

def query_model(model: str, prompt: str) -> Optional[str]:
    """
    Query an Ollama model with error handling and timeouts.
    """
    try:
        start_time = time.time()
        response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 500})
        elapsed = time.time() - start_time
        
        if response and "response" in response:
            return response["response"]
        else:
            return None
    except Exception:
        return None

def process_batch(questions: List[Dict], models: List[str], output_file: str) -> None:
    """
    Process a batch of questions across multiple models in parallel.
    """
    warmed_up_models = [m for m in models if warm_up_model(m)]
    if not warmed_up_models:
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Question ID", "Category", "Question (EN)", "Question (ZH)", 
            "Model", "Response (EN)", "Response (ZH)"
        ])

        # Calculate total number of queries
        total_queries = len(questions) * len(warmed_up_models) * 2  # *2 for EN and ZH
        progress_bar = tqdm(total=total_queries, desc="Processing queries")

        responses = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_params = {}
            
            # Submit all queries
            for model in warmed_up_models:
                for q in questions:
                    future_en = executor.submit(query_model, model, q["text_en"])
                    future_zh = executor.submit(query_model, model, q["text_zh"])
                    future_to_params[future_en] = {**q, "model": model, "is_english": True}
                    future_to_params[future_zh] = {**q, "model": model, "is_english": False}

            # Track responses
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                q_id, model = params["id"], params["model"]
                
                if (q_id, model) not in responses:
                    responses[(q_id, model)] = {"en": None, "zh": None}
                
                try:
                    result = future.result()
                    key = "en" if params["is_english"] else "zh"
                    responses[(q_id, model)][key] = result
                    
                    progress_bar.update(1)
                    
                    # Write to CSV if we have both responses
                    if all(responses[(q_id, model)].values()):
                        writer.writerow([
                            q_id,
                            params["category"],
                            params["text_en"],
                            params["text_zh"],
                            model,
                            responses[(q_id, model)]["en"],
                            responses[(q_id, model)]["zh"]
                        ])
                        file.flush()
                except Exception:
                    progress_bar.update(1)

        progress_bar.close()

if __name__ == "__main__":
    try:
        logging.info("started")
        questions = load_questions('questions.json')
        total_questions = len(questions)
        total_categories = len(set(q['category'] for q in questions))
        
        # Save in the same directory as the script
        output_file = 'responses.csv'
        process_batch(questions, models, output_file)
        logging.info("completed")
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        sys.exit(1)
