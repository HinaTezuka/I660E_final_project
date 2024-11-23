import sys

from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import get_dataset_config_names, load_dataset

from funcs import (
    make_prompts_for_amazon_datasets,
    is_valid_rating,
    calculate_mae,
    calculate_rmse,
)

""" model cinfigs """
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "vinai/RecGPT-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

""" datasets """
amazon_toys_ds = load_dataset(path="smartcat/Amazon_Toys_and_Games_2018", name="reviews", split="train")
"""
Dataset({
    features: ['reviewerID', 'reviewerName', 'overall', 'reviewTime', 'asin', 'reviewText', 'summary'],
    num_rows: 1398
})
"""

""" keep correct label as dict """
labels = defaultdict(float)
for review in amazon_toys_ds:
    reviewerID = review['reviewerID']
    rating = review['overall'][0]
    labels[reviewerID] = rating

""" make prompts """
prompts = make_prompts_for_amazon_datasets(amazon_toys_ds)
""" dict for tracking each MAE and RMSE """
eval_values = defaultdict(list)

""" evaluation """
for reviewerID, prompt in prompts.items():
    # input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device) # original
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    max_length = 2048
    input_ids = input_ids[:, :max_length]
    output_ids = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.batch_decode(output_ids[:, input_ids.size(1) :], skip_special_tokens=True)[0].strip()
    # flag for judging whether response from the model is rating or not
    is_valid_response = is_valid_rating(response)
    if is_valid_response:
        actual = labels[reviewerID] # actual value (label)
        predicted = float(response[:3]) # predicted value(if is_valid_response is True, response="?.?/5.0", so extract first"?.?" rating as a float.)
        """ evaluation """
        eval_values["actual"].append(actual)
        eval_values["predicted"].append(predicted)

if __name__ == "__main__":
    """ results """
    # calculate mean of MAE_list and RMSE_list respectively.
    MAE = calculate_mae(eval_values["actual"], eval_values["predicted"])
    RMSE = calculate_rmse(eval_values["actual"], eval_values["predicted"])
    print(f"final result (MAE, RMSE): {MAE, RMSE}")
