import core as c
import functions as f
import numpy as np
import pandas as pd
import json
import time

begin_time = time.time()
n_bootstraps, n_hash = 10, 420
char_to_int = [ord(letter) for i, letter in enumerate('Frasincar')]
np.random.seed(sum(char_to_int))
K, P, Q = range(2, 8), range(2, 6), range(1, 5)
B = [0, 30, 35, 42, 60, 70, 84, 105, 140, 210, 420]


with open('TVs-all-merged.json') as data:
    data_json = json.load(data)
full_data = f.unpack(data_json)
d_size = len(full_data)
ids, titles, shops, brands, scr_sizes, refr_rates, features = c.preprocess(full_data)

results_LSH = []
for k in K:
    for b in B:
        print(f'F1*-scores for k={k}, b={b}:')
        if b != 0:
            t = (1 / b) ** (1 / (n_hash / b))
        else:
            t = 0
        for r in range(n_bootstraps):
            try:
                start_time = time.time()
                bootstrap = set([int((np.random.rand() * d_size) % d_size) for i in range(d_size)])
                bootstrap_size = len(bootstrap)
                train_indices, test_indices = list(bootstrap), [i for i in range(d_size) if i not in bootstrap]
                tr_ids, te_ids = f.split_items(ids, train_indices, test_indices)
                tr_titles, te_titles = f.split_items(titles, train_indices, test_indices)
                tr_shops, te_shops = f.split_items(shops, train_indices, test_indices)
                tr_brands, te_brands = f.split_items(brands, train_indices, test_indices)

                tr_cps, tr_aps, tr_fc, tr_pms = c.preselection(tr_ids, tr_titles, tr_shops, k, n_hash, b, lsh=bool(b))

                iteration_time = round((time.time() - start_time), 1)
                results_LSH.append(
                    {'k-shingle': k, 'n_bands': b, 'threshold': t,
                     'bootstrap': r+1, 'execution time (sec)': iteration_time,
                     'fraction of comparisons': tr_fc,
                     'pair quality': tr_pms[0], 'pair completeness': tr_pms[1], 'f1*': tr_pms[2]})
                print(f'{round(tr_pms[2], 4)}, time: {int(iteration_time)}s')
            finally:
                continue

results_LSH = pd.DataFrame(results_LSH)
results_LSH.to_excel('results-LSH.xlsx', index=False, float_format="%.6f")

avg_results_LSH = results_LSH.groupby(['k-shingle', 'n_bands']).mean()
max_f1_star_LSH = pd.Series.argmax(avg_results_LSH[['f1*']])
k_opt = K[max_f1_star_LSH // len(B)]  # k_opt = 4
b_opt = B[max_f1_star_LSH % len(B)]  # b_opt = 60
t_opt = (1 / b_opt) ** (1 / (n_hash / b_opt))
print(f'LSH complete, optimal F1* at k = {k_opt} and b = {b_opt} (making t={t_opt})')

results_full = []
for p in P:
    for q in Q:
        print(f'F1-scores for p={p}, q={q}:')
        for r in range(n_bootstraps):
            start_time = time.time()
            bootstrap = set([int((np.random.rand() * d_size) % d_size) for i in range(d_size)])
            bootstrap_size = len(bootstrap)
            train_indices, test_indices = list(bootstrap), [i for i in range(d_size) if i not in bootstrap]
            tr_ids, te_ids = f.split_items(ids, train_indices, test_indices)
            tr_titles, te_titles = f.split_items(titles, train_indices, test_indices)
            tr_shops, te_shops = f.split_items(shops, train_indices, test_indices)
            tr_brands, te_brands = f.split_items(brands, train_indices, test_indices)
            tr_scr_sizes, te_scr_sizes = f.split_items(scr_sizes, train_indices, test_indices)
            tr_refr_rates, te_refr_rates = f.split_items(refr_rates, train_indices, test_indices)
            tr_features, te_features = f.split_items(features, train_indices, test_indices)

            tr_cps, tr_aps, tr_fc, tr_pms = c.preselection(tr_ids, tr_titles, tr_shops, k_opt, n_hash, b_opt)
            tr_labels, tr_data = c.candidate_datagen(tr_cps, tr_aps, p, q,
                                                     tr_titles, tr_brands, tr_scr_sizes, tr_refr_rates, tr_features)
            model = c.train(tr_labels, tr_data)
            tr_ams = c.test(model, tr_aps, tr_cps, tr_data)

            iteration_time = round((time.time() - start_time))
            results_full.append(
                {'key-shingle': p, 'value-shingle': q,
                 'bootstrap': r + 1, 'execution time (sec)': iteration_time,
                 'fraction of comparisons': tr_fc,
                 'pair quality': tr_pms[0], 'pair completeness': tr_pms[1], 'f1*': tr_pms[2],
                 'precision': tr_ams[0], 'recall': tr_ams[1], 'f1': tr_ams[2]})
            print(f'{round(tr_ams[2], 4)}, time: {int(iteration_time)}s')

results_full = pd.DataFrame(results_full)
results_full.to_excel('results-full.xlsx', index=False, float_format="%.6f")

avg_results_full = results_LSH.groupby(['key-shingle', 'value-shingle']).mean()
max_f1_star_full = pd.Series.argmax(avg_results_LSH[['f1']])
p_opt = P[max_f1_star_LSH // len(Q)]  # p_opt = 4
q_opt = Q[max_f1_star_LSH % len(Q)]  # q_opt = 3
print(f'Classification complete, optimal F1 at p = {p_opt} and q = {q_opt} (for k={k_opt} and b={b_opt}')

k_opt, b_opt, p_opt, q_opt = 4, 60, 4, 3
results_oos = []
print(f'F1-scores of sample for k={k_opt}, b={b_opt}, p={p_opt}, q={q_opt}:')
for r in range(n_bootstraps*10):  # More repetitions here to have more certainty in the estimates
    start_time = time.time()
    bootstrap = set([int((np.random.rand() * d_size) % d_size) for i in range(d_size)])
    bootstrap_size = len(bootstrap)
    train_indices, test_indices = list(bootstrap), [i for i in range(d_size) if i not in bootstrap]
    tr_ids, te_ids = f.split_items(ids, train_indices, test_indices)
    tr_titles, te_titles = f.split_items(titles, train_indices, test_indices)
    tr_shops, te_shops = f.split_items(shops, train_indices, test_indices)
    tr_brands, te_brands = f.split_items(brands, train_indices, test_indices)
    tr_scr_sizes, te_scr_sizes = f.split_items(scr_sizes, train_indices, test_indices)
    tr_refr_rates, te_refr_rates = f.split_items(refr_rates, train_indices, test_indices)
    tr_features, te_features = f.split_items(features, train_indices, test_indices)

    tr_cps, tr_aps, tr_fc, tr_pms = c.preselection(tr_ids, tr_titles, tr_shops, k_opt, n_hash, b_opt)
    te_cps, te_aps, te_fc, te_pms = c.preselection(te_ids, te_titles, te_shops, k_opt, n_hash, b_opt)

    tr_labels, tr_data = c.candidate_datagen(tr_cps, tr_aps, p_opt, q_opt,
                                             tr_titles, tr_brands, tr_scr_sizes, tr_refr_rates, tr_features)
    te_labels, te_data = c.candidate_datagen(te_cps, te_aps, p_opt, q_opt,
                                             te_titles, te_brands, te_scr_sizes, te_refr_rates, te_features)

    model = c.train(tr_labels, tr_data)
    te_ams = c.test(model, te_aps, te_cps, te_data)

    iteration_time = round((time.time() - start_time))
    results_oos.append(
        {'bootstrap': r + 1, 'execution time (sec)': iteration_time,
         'fraction of comparisons': te_fc,
         'pair quality': te_pms[0], 'pair completeness': te_pms[1], 'f1*': te_pms[2],
         'precision': te_ams[0], 'recall': te_ams[1], 'f1': te_ams[2]})
    print(f'{round(te_ams[2], 4)}, time: {int(iteration_time)}s')

results_oos = pd.DataFrame(results_oos)
results_oos.to_excel('results-oos.xlsx', index=False, float_format="%.6f")

mean_f1_oos, std_f1_oos = results_oos['f1'].mean(), results_oos['f1'].std()
print(f'Out-of-sample testing complete, mean and standard deviation of F1-score: {mean_f1_oos} ({std_f1_oos})')

print(f'Total runtime: {round((time.time()-begin_time)//60)}m {round((time.time()-begin_time)%60)}s')
