# library doc string

# TODO: Add logging
# import libraries
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from constants import (
    CAT_COLUMNS,
    DATA_PATH,
    MODEL_OUTPUT_PATH,
    MODEL_SCORE_OUTPUT_PATH,
    PARAM_GRID,
    QUANT_COLUMNS,
    RESULT_FEATURE_IMPORTANCE_PATH,
    RESULT_IMAGE_PATH,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    # encode churn
    df[response] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    for cat in category_lst:
        cat_list = []
        cat_groups = df.groupby(cat).mean()[response]
        for val in df[cat]:
            cat_list.append(cat_groups.loc[val])
        df[f"{cat}_{response}"] = cat_list
    return df


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    CAT_COLS_RENAMED = [f"{elem}_{response}" for elem in CAT_COLUMNS]
    cols_to_keep = QUANT_COLUMNS + CAT_COLS_RENAMED
    X = pd.DataFrame()
    X[cols_to_keep] = df[cols_to_keep]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # Create Reports
    # Train results
    train_report_lr = pd.DataFrame(
        classification_report(y_train, y_train_preds_lr, output_dict=True)
    ).transpose()
    train_report_rf = pd.DataFrame(
        classification_report(y_train, y_train_preds_rf, output_dict=True)
    ).transpose()
    pd.concat([train_report_lr, train_report_rf], axis=0).to_csv(
        path_or_buf=f"{MODEL_SCORE_OUTPUT_PATH}/train_report.csv"
    )
    # Test Results
    test_report_lr = pd.DataFrame(
        classification_report(y_test, y_test_preds_lr, output_dict=True)
    ).transpose()
    test_report_rf = pd.DataFrame(
        classification_report(y_test, y_test_preds_lr, output_dict=True)
    ).transpose()
    pd.concat([test_report_lr, test_report_rf], axis=0).to_csv(
        path_or_buf=f"{MODEL_SCORE_OUTPUT_PATH}/test_report.csv"
    )


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    model_loaded = joblib.load(model)


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
               y_train_preds_rf: train labels predicted (rf)
               y_test_preds_rf: test labels predicted (rf)
               y_train_preds_lr: train labels predicted (lr)
               y_test_preds_lr: test labels predicted (lr)

    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    param_grid = PARAM_GRID
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # Create Images
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(f"{RESULT_IMAGE_PATH}/combined.png")
    # Save models
    joblib.dump(cv_rfc.best_estimator_, f"{MODEL_OUTPUT_PATH}/rfc_model.pkl")
    joblib.dump(lrc, f"{MODEL_OUTPUT_PATH}/logistic_model.pkl")
    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":
    df = import_data(DATA_PATH)
    df_enc = encoder_helper(df=df, category_lst=CAT_COLUMNS, response="Churn")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df_enc, response="Churn"
    )
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
        X_train, X_test, y_train, y_test
    )
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
