from sklearn.decomposition import NMF
import pandas as pd


def nmf_topic_modelling(n_nmf_topics, tfidf_mat):  # Experimental only

    # run NMF on TFIDF
    return NMF(n_components=n_nmf_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_mat)


def calculate_weights(nmf, feature_names):
    term_weights_to_sum = 10  # 0 to sum all weights
    term_weights_to_print = 50

    # create list of all or of top n terms & weights for every topic
    top_features = []
    if term_weights_to_sum == 0:
        term_weights = nmf.components_.sum(axis=0)
        top_features = zip(feature_names, term_weights)
    else:
        for topic_idx, term_weights in enumerate(nmf.components_):
            for idx in term_weights.argsort()[:-term_weights_to_sum - 1:-1]:
                top_features.append((feature_names[idx], term_weights[idx]))

    # sum term weights over topics and print
    top_features_df = pd.DataFrame(top_features, columns=['feature', 'score'])
    top_features_df = top_features_df.groupby(top_features_df.feature).sum(). \
        sort_values(by='score', ascending=False).reset_index()
    print("Term weights extracted from topics (sum over all topics of term weights associated with each topic):")
    print(top_features_df[0:term_weights_to_print])
    print()
