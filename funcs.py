import sys
import re
import math
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_prompts_for_amazon_beauty(datasets) -> defaultdict(str):
    prompts = defaultdict(str)
    for review in datasets:
        review_text = review['reviewText'][0]
        reviewerID = review['reviewerID']
        # print(review_text)
        # print(reviewerID)
        # # sys.exit()
        prompt = f"""
        ### Instruction:
        Predict rating for the item.
        Given the interaction history of a user with beauty products as follows:

        Review: {review_text}; Rating:
        """
        prompts[reviewerID] = prompt

    return prompts

def is_valid_rating(response):
    return bool(re.match(r"^\d(\.\d{1})?\/5\.0$", response))

# MAE
# def calculate_mae(actual: float, predicted: float) -> float:
#     return abs(actual - predicted)
def calculate_mae(actual:float, predicted: float) -> float:
    return mean_absolute_error(actual, predicted)

# RMAE
# def calculate_rmse(actual: float, predicted: float) -> float:
#     return math.sqrt((actual - predicted) ** 2)
def calculate_rmse(actual: float, predicted: float) -> float:
    return mean_squared_error(actual, predicted, squared=True)

def calculate_final_result(eval_values: defaultdict(float)) -> (float, float):
    """
    return the final result of MAE and RMSE (take means).
    """
    return np.array(eval_values["MAE"]).mean(), np.array(eval_values["RMSE"]).mean()
