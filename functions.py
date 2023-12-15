import numpy as np
import pandas as pd
import re


def unpack(json_obj):
    unpacked_json = []
    for key, value in json_obj.items():
        for element in value:
            unpacked_json.append(element)

    return unpacked_json


def split_items(data: list, indices_a: list, indices_b: list):
    output_a = [data[index] for index in indices_a]
    output_b = [data[index] for index in indices_b]
    return output_a, output_b


def extract_values(kvp_set, key_name):
    temp_list = []
    for i in range(len(kvp_set)):
        temp_list.append(kvp_set[i].get(key_name))
    return temp_list


def clean_punctuation_measurements(obj: list, replacements: list, spaces: bool):
    if not spaces:
        replacements.append((' ', ''))
    for i in range(len(obj)):
        for edit in replacements:
            obj[i] = obj[i].replace(*edit)
        obj[i] = re.sub(r'[^\w\s]', '', obj[i])
    return obj


def skim_text(texts: list, remove: set):
    for i in range(len(texts)):
        for item in remove:
            texts[i] = texts[i].lower().replace(item, '')
    return texts


def extract_quantity(string_list, measurement, n_char):
    result = []
    for i in range(len(string_list)):
        text = string_list[i]
        loc = text.find(measurement)
        if loc == -1:
            result.append(None)
        else:
            temp = text[(loc - n_char):loc].strip()
            result.append(temp)
    return result


def pair_match(str_list):
    dim = len(str_list)
    matrix = np.full((dim, dim), True)
    for i in range(dim):
        for j in range(len(str_list)):
            if (str_list[i] is not None) and (str_list[j] is not None):
                matrix[i][j] = str_list[i] == str_list[j]
    return matrix


def shingle_matrix(str_list: list, k):
    shingle_set = set()
    for string in str_list:
        if len(string) <= k:
            shingle_set.add(string)
            break
        for i in range(len(string)+1-k):
            shingle = string[i:i+k]
            shingle_set.add(shingle)
    shingle_list = list(shingle_set)

    bool_matrix = np.full((len(shingle_set), len(str_list)), False)
    for m in range(len(shingle_list)):
        curr_shingle = shingle_list[m]
        for n in range(len(str_list)):
            desc = str_list[n]
            if len(desc) <= k:
                if curr_shingle == desc:
                    bool_matrix[m][n] = True
                continue
            if curr_shingle in desc:
                bool_matrix[m][n] = True

    return bool_matrix


def min_hash(sparse_matrix: np.ndarray, n_perm):
    n_shingles, n_items = np.shape(sparse_matrix)
    sig_matrix = np.zeros((n_perm, n_items))

    for i in range(n_perm):
        perm = np.random.permutation(sparse_matrix)
        for j in range(n_items):
            for k in range(n_shingles):
                if perm[k, j]:
                    sig_matrix[i, j] = k
                    break

    return sig_matrix


def lsh(sig_matrix, n_bands):
    sig_matrix = pd.DataFrame(sig_matrix)
    if np.shape(sig_matrix)[0] % n_bands != 0:
        print('Warning: not all rows used for bands')
    candidate = []

    for q, subset in enumerate(np.array_split(sig_matrix, n_bands, axis=0)):
        band = []
        for col in subset.columns:
            block = [str(int(signature)) for signature in subset.iloc[:, col]]
            identifier = '.'.join(block)
            band.append(identifier)

        for i in range(len(band)-1):
            for j in range(i+1, len(band)):
                if band[i] == band[j]:
                    candidate.append((i, j))

    candidate_list = list(set(candidate))
    return candidate_list


def accuracy_measures(prediction_list, actual_list):
    prediction_list = set(prediction_list)  # in LSH: candidates
    actual_list = set(actual_list)

    duplicate = [1 for prediction in prediction_list if prediction in actual_list]
    duplicates_found = sum(duplicate)

    if len(prediction_list) == 0:
        precision = 0
        recall = 0
    else:
        precision = duplicates_found/len(prediction_list)  # in LSH: pair quality
        recall = duplicates_found/len(actual_list)  # in LSH: pair completeness

    if (precision+recall) == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def jaccard_sim_str(str1, str2, k):
    strings_shingled = shingle_matrix([str1, str2], k)
    bool_vector1, bool_vector2 = strings_shingled[:, 0], strings_shingled[:, 1]

    intersection = sum([int(bin_value) for i, bin_value in enumerate(bool_vector1) if bool_vector2[i]])
    union = len(bool_vector1)
    return intersection/union


def kvp_sim(kvp_set1: dict, kvp_set2: dict, k, q):
    key_set1, key_set2 = kvp_set1.keys(), kvp_set2.keys()
    key_sims = np.array([jaccard_sim_str(key1, key2, k) for key1 in key_set1 for key2 in key_set2])
    val_sims = np.array([jaccard_sim_str(kvp_set1[key1], kvp_set2[key2], q) for key1 in key_set1 for key2 in key_set2])
    sim, total_weight = sum(key_sims*val_sims), sum(key_sims)

    if total_weight == 0:
        return 0

    return sim / total_weight
