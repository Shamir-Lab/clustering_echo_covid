import sys
import random
import consensus_with_kprototype
import pandas as pd
import json
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging


def cluster_helper(labels, n_clusters, df, name, path=''):
    """
    saves a copy of the clusters labels and create a separated dataframe for each cluster
    :param labels: clusters' labels
    :param n_clusters: number of clusters
    :param df: full data frame - including all clusters
    :param name: name for the file to save
    :param path: path for the file to save
    :return: list of all dataframes, one for each cluster
    """
    clusters = []
    dictionary = {}
    for i in range(n_clusters):
        clusters.append([j for j, e in enumerate(list(map(int, labels))) if e == i])
        dictionary[i] = clusters[i]

    with open(f'{path}clusters_indexes {name} - {n_clusters}.txt', 'w') as convert_file:
        convert_file.write(json.dumps(dictionary))
    dfs = []
    if df is not None:
        for i in range(n_clusters):
            dfs.append(df.iloc[clusters[i]])
    return dfs

def helper_numeric_p_vals(df_holder, func, dfs, col, p_val_numeric, logger):
    """
    run the statistical test for continuous variables and put the results in p_val_numeric
    :param df_holder: the dataframe that holds the values
    :param func: the statistical test function
    :param dfs: dataframes of the different clusters
    :param col: the variable (column)
    :param p_val_numeric: array of p-value results
    :param logger: a logger object
    :return: None
    """
    try:
        s, p = func(*df_holder)
        if pd.isna(p):
            return None
        temp_list = []
        for i in range(len(dfs)):
            temp_list.append(dfs[i][col].mean(axis=0))
            temp_list.append(dfs[i][col].std(axis=0))
            temp_list.append(len(df_holder[i]))
        temp_list.append(p)
        p_val_numeric[col] = temp_list
    except:
        if logger:
            logger.warning("issues in col = ", col)
        return None


def get_pvals(numeric_cols, categorical_cols, outcomes_cols, file_name, func=f_oneway, labels=None, is_cat=True,
              df_all=None, path='', logger = None):
    """
    run anova test for the continuous variables and chi square for the categorical variables, saves the results in three
    csv files.
    :param numeric_cols: continuous variables
    :param categorical_cols: categorical variables
    :param outcomes_cols: outcomes variables
    :param file_name: file name to save the results
    :param func: the function to use for the statistical test for the continuous variables
    :param labels: cluster labels
    :param is_cat: boolean - to include categorical variables in the analysis
    :param df_all: full data farme of all variables
    :param path: path to save the results files
    :param logger: ש logger object
    :return: None
    """
    # separated data frame for each cluster
    dfs = cluster_helper(labels, n_clusters, df_all, name, path)
    p_val_numeric = pd.DataFrame()
    p_val_categ = pd.DataFrame()
    p_val_out = pd.DataFrame()
    df_holder = [None]*len(dfs)

    numric_cols=set(numeric_cols)
    categorial_cols=set(categorical_cols)
    outcomes = set(outcomes_cols)
    for col in numric_cols:
        # for the additional echo cols that are present more than once (only the first was used as an input)
        if not isinstance(dfs[0][col], pd.Series):
            for j in range(len(dfs)):
                for i in range(len(dfs[j][col].columns)):
                    dfs[j][f"{col} {i}"] = dfs[j][col].iloc[:,i]
            for i in range(len(dfs[0][col].columns)):
                col_name = f"{col} {i}"
                for j in range(len(dfs)):
                    df_holder[j] = dfs[j][col_name].dropna()
                helper_numeric_p_vals(df_holder, func, dfs, col_name, p_val_numeric, logger)
        else:
            for i in range(len(dfs)):
                    df_holder[i] = dfs[i][col].dropna()
            helper_numeric_p_vals(df_holder,func, dfs,col,p_val_numeric, logger)

    p_val_numeric = p_val_numeric.transpose()
    cols = []
    for i in range(len(dfs)):
        cols.append(f'mean-{i}')
        cols.append(f'std-{i}')
        cols.append(f'valid-{i}')
    cols.append(f'p-value')
    p_val_numeric.columns = cols
    p_val_numeric = p_val_numeric.sort_values(by="p-value")

    # p-values for the continuous variables
    p_val_numeric.to_csv(file_name+"p_val_numeric.csv")

    if is_cat:
        df_new = pd.concat(dfs)
        labels = [0]*len(dfs[0])+[1]*len(dfs[1])
        labels = pd.Series(labels)
        for col in categorial_cols:
            try:
                sums = [None] * len(dfs)
                if not df_all.empty:
                    table = pd.crosstab(index=labels, columns=df_new[col])
                else:
                    # non binary variable
                    if col.startswith("Admission reason"):
                        table = pd.crosstab(index=labels, columns=df_all[col])
                    else:
                        for i in range(len(dfs)):
                            df_holder[i] = dfs[i][col].dropna()
                            sums[i] = df_holder[i].sum()
                        table = [None]*len(dfs)
                        for i in range(len(dfs)):
                            table[i] = [sums[i], len(df_holder[i]) - sums[i]]
                chi, p, d, x = chi2_contingency(table)
                if pd.isna(p):
                    continue
                if not labels.empty:
                    for i in range(len(dfs)):
                        df_holder[i] = dfs[i][col].dropna()
                        sums[i] = df_holder[i].sum()

                temp_list = []
                for i in range(len(dfs)):
                    temp_list.append(len(df_holder[i]))
                    temp_list.append(sums[i])
                temp_list.append(p)
                temp_list.append(chi)
                p_val_categ[col] = temp_list
            except:
                if logger:
                    logger.warning("issues in col = ", col)

        p_val_categ = p_val_categ.transpose()
        cols = []
        for i in range(len(dfs)):
            cols.append(f'valid {i}')
            cols.append(f'sum {i}')
        cols.append("p-value")
        cols.append("chi stat")
        p_val_categ.columns = cols
        p_val_categ = p_val_categ.sort_values(by="p-value")
        p_val_categ.to_csv(file_name + "_p_val_categ.csv")

    if is_cat:
        df_new = pd.concat(dfs)
        labels = [0]*len(dfs[0])+[1]*len(dfs[1])
        labels = pd.Series(labels)
        for col in outcomes:
            try:
                sums = [None] * len(dfs)
                if not df_all.empty:
                    table = pd.crosstab(index=labels, columns=df_new[col])
                else:
                    for i in range(len(dfs)):
                        df_holder[i] = dfs[i][col].dropna()
                        sums[i] = df_holder[i].sum()
                    table = [None] * len(dfs)
                    for i in range(len(dfs)):
                        table[i] = [sums[i], len(df_holder[i]) - sums[i]]
                chi, p, d, x = chi2_contingency(table)
                if pd.isna(p):
                    continue
                if not labels.empty:
                    for i in range(len(dfs)):
                        df_holder[i] = dfs[i][col].dropna()
                        sums[i] = df_holder[i].sum()
                temp_list = []
                for i in range(len(dfs)):
                    temp_list.append(len(df_holder[i]))
                    temp_list.append(sums[i])
                temp_list.append(p)
                temp_list.append(chi)
                p_val_out[col] = temp_list
            except:
                if logger:
                    logger.warning("issues in col = ", col)

        p_val_out = p_val_out.transpose()
        cols = []
        for i in range(len(dfs)):
            cols.append(f'valid {i}')
            cols.append(f'sum {i}')
        cols.append("p-value")
        cols.append("chi stat")
        p_val_out.columns = cols
        p_val_out = p_val_out.sort_values(by="p-value")

        # p-values for the categorical and outcome variables
        p_val_out.to_csv(file_name + "_p_val_outcomes.csv")

def oneHot(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)


def calc_c_index(df_origin, df_all_num, df_all_cat, clusters, times_name, outcome_name, logger):
    """
    calculate c-index for a cox model based on the clutering results
    :param df_origin: full data frame of all variables
    :param df_all_num: continuous variables data frame
    :param df_all_cat: categorical variables data frame
    :param clusters: clusters' labels
    :param times_name: name of variable that represents the time
    :param outcome_name: outcome variable
    :param logger: a logger object
    :return: c-index result
    """

    df_all = pd.concat([df_all_num, df_all_cat], axis=1)

    raw_times = list(df_origin[times_name])
    times = [x if x and x >= 0 else 0 for x in raw_times]
    events_raw = df_origin[outcome_name]
    events = [1 if x and x > 0 else 0 for x in events_raw]
    for j in range(len(events)):
        if pd.isna(raw_times[j]) or (not raw_times[j]) or (raw_times[j] < 0):
            events[j] = 0

    binary_table = oneHot(clusters)
    for i in range(binary_table.shape[1]):
        df_all[str(i)] = binary_table[:,i]
    df_all["T"] = times
    df_all["E"] = events
    df_all = df_all[["0", "1", "2", "T", "E"]]

    cph = CoxPHFitter()
    cph.fit(df_all, duration_col="T", event_col="E")

    summary_str = cph.summary.to_string()
    logger.info(summary_str)
    c_index = concordance_index(df_all['T'], -cph.predict_partial_hazard(df_all), df_all['E'])
    return c_index


def logrank(df_origin, clusters, times_name, outcome_name):
    """
    log rank test
    :param df_origin: dataframe of full data
    :param clusters: labels of clusters
    :param times_name: name of time variable
    :param outcome_name: name of ourcome variable
    :return: log rank p-value
    """
    raw_times = list(df_origin[times_name])
    times = [x if x and x >= 0 else 0 for x in raw_times]
    events_raw = df_origin[outcome_name]
    events = [1 if x and x > 0 else 0 for x in events_raw]
    for j in range(len(events)):
        if pd.isna(raw_times[j]) or (not raw_times[j]) or (raw_times[j] < 0):
            events[j] = 0
    result = multivariate_logrank_test(times, clusters, events)
    return result.p_value


def make_consensus(n_clusters, df_all_num, df_all_cat,gamma=None, r=1, logger = None):
    """
    perform consensus clustering with k prototype algorithm
    :param n_clusters: number of clusters
    :param df_all_num: data frame of continuous variables
    :param df_all_cat: data frame of categorical variables
    :param gamma: parameter for the k prototype algorithm
    :param r: resampeling rate for the consensus clustering
    :param logger: a logger object
    :return: clusters' labels
    """
    consensus = consensus_with_kprototype.ConsensusCluster(cluster=KMeans, L=n_clusters, K=n_clusters + 1, H=50, resample_proportion=r)
    consensus.fit(df_all_num, df_all_num, df_all_cat, gamma, verbose=False)

    lables = consensus.predict()
    for i in range(n_clusters):
        if logger:
            logger.info(f"len of cluster {i} = {len([j for j, e in enumerate(list(map(int, lables))) if e == i])}")
    return lables


def analyses(labels, df_all, df_all_num, df_all_cat, path, outcomes, logger):
    """
    Perform statistical tests for the values of the different variables in the different clusters, calculate
    the c-index for in hospital mortality and the log rank p-value
    :param labels: clusters' labels
    :param df_all: full data
    :param df_all_num: data with continuous variables
    :param df_all_cat: data with categorical variables
    :param path: path to save the results
    :param logger: a logger object
    :return: c-index value anf log rank p-value
    """
    # run appropriate statistical tests for all variables
    get_pvals(df_all_num.columns, df_all_cat.columns, outcomes, file_name=f'{path}{name} - {n_clusters}', labels=labels,
              df_all=df_all, path=path, logger=logger)

    # c-index for the clustering solution
    c_index = calc_c_index(df_all,df_all_num, df_all_cat, labels, "days in hospital ", "In hospital mortality", logger)

    log_rank_p_value = logrank(df_all, labels, "days in hospital ", "In hospital mortality")
    return c_index, log_rank_p_value


def shuffle_vals(df_num, n, df_cat, start_ind= None):
    """
    shuffle values of either echo variables or non-echo varibles
    :param df_num: dataframe of continuous variables
    :param n: number of continuous variables to shuffle
    :param df_cat: dataframe of categorical variables
    :param start_ind: first echo variable index. if doesn't exist - shuffle the non echo variables
    :return: None
    """
    if start_ind:
        # # to test the total removal of echo
        # num_cols = df_num.columns[start_ind:start_ind+n]
        # df_num.drop(num_cols, axis=1, inplace=True)
        for i in range(n):
            seri = pd.Series(df_num[df_num.columns[start_ind+i]], index=df_num.index).sample(n=len(df_num))
            seri.index = df_num.index
            df_num[df_num.columns[start_ind + i]] = seri
        cat_cols = ["At<100", "bad heart condition (>=2)"]
        # # to test the total removal of echo
        # df_cat.drop(cat_cols, axis=1, inplace=True)
        for col in cat_cols:
            seri = pd.Series(df_cat[col], index=df_cat.index).sample(n=len(df_cat))
            seri.index = df_cat.index
            df_cat[col] = seri
    # suffle non-echo variables
    else:
        ran = range(len(df_num.columns))
        indxs = random.choices(ran, k=n)
        for i in indxs:
            seri = pd.Series(df_num[df_num.columns[i]], index=df_num.index).sample(n=len(df_num))
            seri.index = df_num.index
            df_num[df_num.columns[i]] = seri
        ran = range(len(df_cat.columns))
        indxs = random.choices(ran, k=2)
        for i in indxs:
            seri = pd.Series(df_cat[df_cat.columns[i]], index=df_cat.index).sample(n=len(df_cat))
            seri.index = df_cat.index
            df_cat[df_cat.columns[i]] = seri



def remove_at_random(df_numeric, num_of_vars, df_all, df_all_cat, test_echo=False, echo_start_ind = None, logger=None):
    """
    randomly shuffle the echo variables of an equivalent number of non echo variables and see the new c index and log
    rank
    :param df_numeric: dataframe of the continuous variables
    :param num_of_vars: number of variables to shuffle
    :param df_all: dataframe of the original data
    :param df_all_cat: dataframe of the categorical variables
    :param test_echo: boolean - to shuffle echo - True, non echo - False
    :param echo_start_ind: index of the first echo variable
    :param logger: a logger object
    :return: average c_index, std c_index, average log rank p-value, std log rank p-value
    """
    n_clusters = 4
    gamma = 3
    r = 0.85
    c_indexes = []
    logranks = []
    number_of_iter = 50

    for i in range(number_of_iter):
        df_num_c = df_numeric.copy()
        df_cat_c = df_all_cat.copy()
        if test_echo:
            shuffle_vals(df_num_c, num_of_vars, df_cat_c, echo_start_ind)
        else:
            shuffle_vals(df_num_c, num_of_vars, df_cat_c)

        labels = make_consensus(n_clusters, df_num_c, df_cat_c, gamma, r, logger)

        c_indexes.append(calc_c_index(df_all, df_numeric, df_all_cat, labels, "days in hospital ", "In hospital mortality", logger))
        logranks.append(logrank(df_all, labels, "days in hospital ", "In hospital mortality"))
    np_arr = np.array(c_indexes)
    ave_c = np.mean(np_arr)
    std_c = np.std(np_arr)

    logger.info(f"mean c index = {ave_c}; std = {std_c}")

    np_arr = np.array(logranks)
    ave_lr = np.mean(np_arr)
    std_lr = np.std(np_arr)
    logger.info(f"mean log rank = {ave_lr}; std = {std_lr}")

    return ave_c, std_c, ave_lr, std_lr



if __name__ == '__main__':
    path_to_otiginal_data = sys.argv[1]  # a csv file of the original full data, containing all variables to use for
                                        # clustering and aditional varibles to include in the input data: outcomes and
                                        # aditional echo measurments
    path_to_df_continuous = sys.argv[2]  # a csv file of the continuous data
    path_to_df_categorical = sys.argv[3]  # a csv file of the categorical data
    outcomes = sys.argv[4].split(',')  # list of outcomes to test

    df_full = pd.read_csv(path_to_otiginal_data, index_col=0)
    df_con = pd.read_csv(path_to_df_continuous,index_col=0)
    df_cat = pd.read_csv(path_to_df_categorical, index_col=0)

    n_clusters = 4
    gamma = 3  # weight parameter for k-prototype
    r = 0.85 # resampling rate for consensus clustering
    path_to_save_results = ""
    name = "consensus clustering for echo-covid"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    labels = make_consensus(n_clusters, df_con, df_cat, gamma, r, logger)
    c_index = analyses(labels, df_full, df_con, df_cat, path_to_save_results, outcomes, logger)

    # test the results after shuffling the echo variables or non echo variables
    echo_continuous_variables = 31
    ave_c, std_c, ave_lr, std_lr = remove_at_random(df_con, echo_continuous_variables, df_full, df_cat, test_echo=False,
                                                    echo_start_ind=None, logger=logger)

