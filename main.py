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
b = B[max_f1_star_LSH % len(B)]  # b_opt = 60
k = K[max_f1_star_LSH // len(B)]  # k_opt = 4
t = (1 / b) ** (1 / (n_hash / b))


results_full = []
print(f'LSH complete, optimal F1* at k = {k} and b = {b}')
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

            tr_cps, tr_aps, tr_fc, tr_pms = c.preselection(tr_ids, tr_titles, tr_shops, k, n_hash, b)
            te_cps, te_aps, te_fc, te_pms = c.preselection(te_ids, te_titles, te_shops, k, n_hash, b)

            tr_labels, tr_data = c.candidate_datagen(tr_cps, tr_aps, p, q,
                                                     tr_titles, tr_brands, tr_scr_sizes, tr_refr_rates, tr_features)
            te_labels, te_data = c.candidate_datagen(te_cps, te_aps, p, q,
                                                     te_titles, te_brands, te_scr_sizes, te_refr_rates, te_features)

            model = c.train(tr_labels, tr_data)
            te_ams = c.test(model, te_aps, te_cps, te_data)  # for OOS-performance, use te_ equivalents

            iteration_time = round((time.time() - start_time))
            results_full.append(
                {'key-shingle': p, 'value-shingle': q, 'threshold': t,
                 'bootstrap': r + 1, 'execution time (seconds)': iteration_time,
                 'fraction of comparisons': tr_fc,
                 'pair quality': te_pms[0], 'pair completeness': te_pms[1], 'f1*': te_pms[2],
                 'precision': te_ams[0], 'recall': te_ams[1], 'f1': te_ams[2]})  # similarly here
            print(f'{round(te_ams[2], 4)} time: {int(iteration_time)}s')

results_full = pd.DataFrame(results_full)
results_full.to_excel('results-full.xlsx', index=False, float_format="%.6f")

print(f'Total runtime: {round((time.time()-begin_time)//60)}m {round((time.time()-begin_time)%60)}s')
