import re
import os
import random
import pandas as pd
import numpy as np

def wildcards_match(s, pattern):
    regex_pattern = "^" + pattern.replace("%", ".*").replace("_", ".") + "$"
    return re.match(regex_pattern, s) is not None

import random

def select_wildcard(prob_dict):
    choice = random.random()
    total = 0.0
    for key, value in prob_dict.items():
        total += value
        if choice < total:
            return key
    return key

import random

def generate_wildcard_pattern(
    s,
    insert_char_prob=0.5,
    wildcard_prob={'%': 0.20, '_': 0.80},
    insert_tail_prob=0.95,
    percent_skip_prob=0.1
):
    pattern = []
    char_flag = False
    i = 0
    s = str(s)
    while i < len(s):
        pattern.append(s[i])
        
        if random.random() < insert_char_prob:
            wildcard = select_wildcard(wildcard_prob)
            pattern.append(wildcard)
            
            if wildcard == '%' and random.random() < percent_skip_prob:
                remaining_chars = len(s) - i - 1
                if remaining_chars > 0:
                    skip = random.randint(0, remaining_chars)
                    i += skip
            else:
                i += 1
        
        i += 1
    
    return ''.join(pattern)


def generateListOfWildCards(strList,insert_char_prob=0.5, wildcard_prob={'%': 0.20, '_': 0.80} , insert_tail_prob=0.95):
    wildStrings = []
    for si in strList:
        wildStrings.append(generate_wildcard_pattern( si,insert_char_prob, wildcard_prob, insert_tail_prob))
    return wildStrings


if __name__ == "__main__":
    insert_char_probs = [0.5, 0.1]
    wildcard_probs = [{'%' : 0.80, '_' : 0.20}, {'%' : 0.20, '_' : 0.80}]
    csv_file_path = '../data/lineitem10000.csv'
    df = pd.read_csv(csv_file_path, header=None)
    sL = df[0].tolist()
    
    os.makedirs('./queries', exist_ok=True)
    
    for insert_char_prob in insert_char_probs:
        for wildcard_prob in wildcard_probs:
            first_num = insert_char_prob * 10
            secon_num = wildcard_prob['%'] * 10
            
            filename = f'./queries/lineitem10000_ipct{first_num}_pct{secon_num}.npy'
            
            wildstrs = generateListOfWildCards(sL, insert_char_prob, wildcard_prob)
            
            np.save(filename, np.array(wildstrs))
            
            print(f"Saved: {filename}")

    
    
