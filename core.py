import functions as f
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import imblearn.over_sampling as sm


def preprocess(full_kvp):
    brand_set = {"philips", "supersonic", "samsung", "sansui", "sanyo", "schneider electric", "seiki digital",
                 "sèleco", "setchell carlson", "sharp", "siemens", "skyworth", "sony", "soyo", "cge", "philco-ford",
                 "howard radio", "healthkit", "cortron", "vestel", "supersonic", "toshiba", "coby", "panasonic",
                 "vizio", "naxa", "viewsonic", "avue", "insignia", "sunbritetv", "optoma", "westinghouse", "dynex",
                 "sceptre", "tcl", "curtisyoung", "compaq", "upstar", "azend", "seiki", "contex", "affinity", "hiteker",
                 "epson", "elo", "gpx", "sigmac", "venturer", "elite", "acer", "admiral", "aiwa", "akai", "alba",
                 "amstrad", "andrea", "apex", "apple", "arcam", "arise india", "aga", "audiovox", "awa", "baird",
                 "bang & olufsen", "beko", "benq", "binatone", "blaupunkt", "bpl group", "brionvega", "bush",
                 "canadian general electric", "changhong", "chimei", "compal electronics", "conar instruments",
                 "continental edison", "cossor", "craig", "curtis mathes", "daewoo", "dell", "delmonico", "dumont",
                 "durabrand", "dynatron", "english electric", "ekco", "electrohome", "element", "emerson", "emi",
                 "farnsworth", "ferguson", "ferranti", "finlux", "fisher electronics", "fujitsu", "funai", "geloso",
                 "general electric", "goldstar", "goodmans industries", "google", "gradiente", "grundig", "haier",
                 "hallicrafters", "hannspree", "heath company", "hinari", "hmv", "hisense", "hitachi", "hoffman",
                 "itel", "itt", "jensen", "jvc", "kenmore", "kent television", "kloss video", "kogan",
                 "kolster-brandes", "konka", "lanix", "le.com", "lg", "loewe", "luxor", "magnavox", "marantz",
                 "marconiphone", "matsui", "memorex", "micromax", "metz", "mitsubishi", "mivar", "motorola", "muntz",
                 "murphy radio", "nec", "nokia", "nordmende", "onida", "orion", "packard bell", "pensonic", "philco",
                 "philips", "pioneer", "planar systems", "polaroid", "proline", "proscan", "pye", "pyle", "quasar",
                 "radioshack", "rauland-borg", "rca", "realistic", "rediffusion", "saba", "salora"}
    shop_set = {"newegg.com", "best buy", "amazon", "thenerds.net"}

    replacements = [('-', ''), ('/', ''), (':', ''), ('–', ''), (';', ''), ('+', ''),
                    ('(', ''), (')', ''), ('[', ''), (']', ''),
                    ('.', " "), (',', " "), ('  ', " "),  ("'", " "),
                    ('Yes', '1'), ('No', '0'),
                    ('Inch', 'inch'), ('\"', 'inch'), ('inches', 'inch'), ('-inch', 'inch'), (' inch', 'inch'),
                    ('Hz', 'hz'), (' Hz', 'hz'), (' hz', 'hz'), ('-hz', 'hz'), ('hertz', 'hz'), ('Hertz', 'hz'),
                    ]  # Can add more later, like with nits etc

    model_ids = f.extract_values(full_kvp, 'modelID')
    shops = f.extract_values(full_kvp, 'shop')
    feature_maps = f.extract_values(full_kvp, 'featuresMap')
    titles = f.extract_values(full_kvp, 'title')

    titles_clean = f.clean_punctuation_measurements(titles.copy(), replacements, spaces=True)
    titles_lsh = f.clean_punctuation_measurements(
        f.skim_text(f.skim_text(titles.copy(), brand_set), shop_set), replacements, spaces=False)

    brands = []
    for title in titles:
        match = False
        for brand in brand_set:
            match = brand.lower() in title.lower()
            if match:
                brands.append(brand)
                break
        if not match:
            brands.append(None)

    screen_sizes = f.extract_quantity(titles_clean, 'inch', 2)
    refresh_rates = f.extract_quantity(titles_clean, 'hz', 3)

    return model_ids, titles_lsh, shops, brands, screen_sizes, refresh_rates, feature_maps


def preselection(model_ids, titles, shops,  k_shingle_length, hashes, bands, lsh=True):
    n_items = len(titles)
    shop_match = f.pair_match(shops)
    candidate_pairs = [(i, j) for i in range(n_items - 1) for j in range(i + 1, n_items) if not shop_match[i][j]]
    tot_comparisons = len(candidate_pairs)

    if lsh:
        titles_shingled = f.shingle_matrix(titles, k_shingle_length)
        sig_matrix = f.min_hash(titles_shingled, hashes)
        candidate_pairs = f.lsh(sig_matrix, bands)
        candidate_pairs = [(i, j) for (i, j) in candidate_pairs if not shop_match[i][j]]

    frac_comp = len(candidate_pairs) / tot_comparisons
    actual_pairs = [(i, j) for i in range(n_items-1) for j in range(i+1, n_items) if model_ids[i] == model_ids[j]]
    pair_quality_measures = f.accuracy_measures(candidate_pairs, actual_pairs)

    return candidate_pairs, actual_pairs, frac_comp, pair_quality_measures


def candidate_datagen(candidate_pairs, actual_pairs, k_shingle_len, q_shingle_len,
                      titles, brands, screen_sizes, refresh_rates, feature_maps):
    brand_match = f.pair_match(brands)
    screen_size_match = f.pair_match(screen_sizes)
    refresh_rate_match = f.pair_match(refresh_rates)

    classification_labels = np.array([int((i, j) in actual_pairs) for (i, j) in candidate_pairs])

    classification_features = np.zeros((len(candidate_pairs), 5))
    classification_features[:, 0] = [f.jaccard_sim_str(titles[i], titles[j], k_shingle_len)
                                     for (i, j) in candidate_pairs]
    classification_features[:, 1] = [f.kvp_sim(feature_maps[i], feature_maps[j], k_shingle_len, q_shingle_len)
                                     for (i, j) in candidate_pairs]
    classification_features[:, 2] = [int(brand_match[i, j]) for (i, j) in candidate_pairs]
    classification_features[:, 3] = [int(screen_size_match[i, j]) for (i, j) in candidate_pairs]
    classification_features[:, 4] = [int(refresh_rate_match[i, j]) for (i, j) in candidate_pairs]

    return classification_labels, classification_features


def train(labels, features):
    smote = sm.SMOTE(sampling_strategy='minority', k_neighbors=(min(len(labels)-1, 5)))
    
    try:
        features_smote, labels_smote = smote.fit_resample(features, labels)
    except ValueError:
        print('note, next value computed without smote')
        features_smote, labels_smote = features, labels

    model = GradientBoostingClassifier()  # Many other models can be used here
    model.fit(features_smote, labels_smote)
    return model


def test(model, actual_pairs, candidate_pairs, classification_features):
    predictions = model.predict(classification_features)
    prediction_list = []
    for n in range(len(candidate_pairs)):
        if predictions[n] == 1:
            prediction_list.append(candidate_pairs[n])

    outcome = f.accuracy_measures(prediction_list, actual_pairs)
    return outcome
