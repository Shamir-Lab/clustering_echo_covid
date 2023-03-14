import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def normalize(df):
    """
    z-score normalization
    :param df: the data frame to normalize
    :return: normalized data frame
    """
    for col in df.columns:
        min = df[col].min(axis=0)
        max = df[col].max(axis=0)
        for i in range(len(df[col])):
            df[col].iloc[i] = (df[col].iloc[i]-min)/(max - min)
    return df

def impute_vals(df, imputer):
    """
    impute missing values in the data frame. drop variables with 2/3 or more missing values
    :param df: the data frame with missing values
    :param imputer: the imputation finction to use
    :return: imputed data frame
    """
    cols_to_remove_insx = []
    missing_factor = 2/3
    blank_limit = len(df)*(missing_factor)

    for i in range(len(df.iloc[0])):
        blank = 0
        for j in range(len(df)):
            val = df.iat[j, i]
            try:
                if (pd.isna(val)):
                    blank += 1
            except:
                pass
        if blank>blank_limit:
            cols_to_remove_insx.append(i)
    if len(cols_to_remove_insx)>0:
        df = df.drop(df.columns[cols_to_remove_insx], axis=1, inplace=False)
    if imputer is None:
        return df
    new_df = imputer.fit_transform(df)
    new_df = pd.DataFrame(new_df)
    new_df.columns = df.columns
    return new_df


def create_full_dfs(df_numeric,df_categorical, imputer_num=IterativeImputer(random_state=None, max_iter=50),
                 imputer_cat = SimpleImputer(strategy="most_frequent"), norm_power_trans=True):
    """

    :param df_numeric: the data frame containing the continuous variables
    :param df_categorical: the data frame containing the categorical variables
    :param imputer_num: imputation function for the continuous variables
    :param imputer_cat: imputation function for the categorical variables
    :param normfix: boolean - use the power transform normalization
    :return: full df_numeric and df_categorical data frames
    """
    ## iterative imputer option for categorical variables
    # imp_cat = IterativeImputer(estimator=RandomForestClassifier(),
    #                        initial_strategy='most_frequent',
    #                        max_iter=10, random_state=0)

    df_categorial_imp = impute_vals(df_categorical, imputer_cat)
    df_numeric_imp = impute_vals(df_numeric, imputer_num)


    if norm_power_trans:
        pt = PowerTransformer(method='yeo-johnson', copy=False)
        pt.fit_transform(df_numeric_imp)

        return df_numeric_imp, df_categorial_imp

    else:
        return normalize(df_numeric_imp), df_categorial_imp



if __name__ == '__main__':
    path_to_file = sys.argv[1]  # a csv file of the full data
    conti_variables = sys.argv[2].split(',')  # a list of all continuous variables
    cat_variables = sys.argv[3].split(',')  # a list of all categorical variables
    path_to_save = sys.argv[4]  # path to save the full data frames that are ready to work


    original_data = pd.read_csv(path_to_file, index_col=0)
    df_con = original_data[conti_variables].copy()  # data frame with continuous variables only
    df_cat = original_data[cat_variables].copy()  # data frame with categorical variables only

    df_con, df_cat = create_full_dfs(df_con, df_cat)
    df_con.to_csv(f"{path_to_save}/continuous_data.csv")
    df_cat.to_csv(f"{path_to_save}/categorical_data.csv")
