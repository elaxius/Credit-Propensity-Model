import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error, roc_curve
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from optbinning import BinningProcess
from joblib import dump
from catboost import CatBoostRegressor

def split_data(df, target_col, drop_cols, test_size=0.3):
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def detect_categorical(X):
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def fit_binning(X_train, y_train, categorical_vars):
    binning_process = BinningProcess(variable_names=X_train.columns.tolist(),
                                     categorical_variables=categorical_vars)
    binning_process.fit(X_train, y_train)
    return binning_process

def create_iv_table(X, binning_process, name):
    iv_table = []
    for var in X.columns:
        bin_var = binning_process.get_binned_variable(var)
        iv_value = bin_var.binning_table.iv
        iv_table.append({"Variable": var, "IV": iv_value})
    iv_df = pd.DataFrame(iv_table).sort_values(by="IV", ascending=False)
    iv_df.to_csv(f"Output/{name}_iv_table.csv", index=False)
    return iv_df

def drop_high_vif(df, threshold=5.0):
    df_vif = sm.add_constant(df.copy())
    while True:
        vif = pd.Series(
            [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])],
            index=df_vif.columns
        )
        max_vif = vif.drop("const").max()
        if max_vif <= threshold:
            break
        drop_col = vif.drop("const").idxmax()
        print(f"Dropping '{drop_col}' with VIF = {max_vif:.2f}")
        df_vif = df_vif.drop(columns=[drop_col])
    return df_vif.drop(columns="const")

def perform_feature_selection(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    sfs = SequentialFeatureSelector(
        lr, n_features_to_select="auto", direction="backward", scoring="roc_auc", cv=3
    )
    sfs.fit(X_train, y_train)
    return X_train.columns[sfs.get_support()].tolist()

def train_model(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train, y_train)
    return lr

def train_regressor(X_train, y_train):
    # Detect categorical features (dtype = 'category')
    cat_features = X_train.select_dtypes(include='category').columns.tolist()

    # Define the model
    catboost_model = CatBoostRegressor(
        loss_function='RMSE'
    )

    # Define the hyperparameters to search over
    param_grid = {
        'l2_leaf_reg':[0,0.2,0.4,0.6,0.8,1],
        'depth': [2, 4, 8, 16],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'iterations': [50, 100, 200, 500]
    }

    # Set up GridSearchCV for cross-validation
    grid_search = GridSearchCV(
        estimator=catboost_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with categorical feature info
    grid_search.fit(X_train, y_train, cat_features=cat_features)

    print(f"✅ Best Parameters: {grid_search.best_params_}")
    best_catboost = grid_search.best_estimator_

    return best_catboost

def evaluate_model(model, X_test, y_test, name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    print("\n🔍 Model Evaluation on Test Set")
    print("AUROC:", roc_auc_score(y_test, y_pred_proba))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(y_test, y_pred_proba):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Output/{name}_roc_curve.png")
    plt.show()

def evaluate_regressor_model(model, X_test, y_test, name):
    # Predict the values
    y_pred = model.predict(X_test)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # MAPE in percentage

    print("\n🔍 Model Evaluation on Test Set")
    print("MAPE:", mape)

    # Plotting the predicted vs actual values (for regression)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted Values ({name})")
    plt.tight_layout()
    plt.savefig(f"Output/{name}_actual_vs_predicted.png")
    plt.show()

def save_model_artifacts(model, binning_process, selected_features, name):
    if model is not None:
        dump(model, f"Model/{name}_model.joblib")
    if binning_process is not None:
        dump(binning_process, f"Model/{name}_binning_process.joblib")
    if selected_features is not None:
        dump(selected_features, f"Model/{name}_selected_features.joblib")

