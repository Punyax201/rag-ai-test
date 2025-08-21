# main.py
# Entry point for your AI model project

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

if __name__ == "__main__":
    print("Hello, AI World!")
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(huggingface_dataset_name)
    example_indices = [40, 200]

    dash_line = '-'.join('' for x in range(100))

    # Load model and tokenizer
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    prompt = ''

    for i, index in enumerate(range(0, 10)):

        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        prompt = f"""
    Summarize the following dialogue:

    {dialogue}

    What is going on?
    """
        if index != 10:
            prompt += f"{summary}\n"

        inputs = tokenizer(prompt, return_tensors='pt')
        model_output = model.generate(
            inputs['input_ids'],
            max_new_tokens=100)
        
        output = tokenizer.decode(model_output[0], skip_special_tokens=True)

        print(dash_line)
        print('Example ', i + 1)
        print(dash_line)
        print('INPUT DIALOGUE:')
        print(dataset['test'][index]['dialogue'])
        print(dash_line)
        print('BASELINE HUMAN SUMMARY:')
        print(dataset['test'][index]['summary'])
        print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')
        print(dash_line)
        print()

    
    # Example: create dummy data and fit a simple model
    # X = np.array([[1], [2], [3], [4]])
    # y = np.array([2, 4, 6, 8])
    # model = LinearRegression()
    # model.fit(X, y)
    # print("Model coefficient:", model.coef_)
