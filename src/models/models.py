import warnings
import os
import sys
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import cartoframes as cf
import xgboost as xgb
import logging

from catboost import CatBoostRegressor, Pool
from datetime import datetime
from hyperopt.pyll.base import scope
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

sys.path.insert(0, "/Users/44371/Documents/Cases/V7FC/aag-vantage/src")

from utils.carto_helpers import get_creds, set_creds
from constants.global_constants import auv_features, PROJECT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class WhitespaceModel:
    """Model object containing data preprocessing, fitting, hyperparam optimization
    and model tracking utilizing MLFlow."""

    def __init__(self):
        self.data = None
        self.model = None
        self.train = None
        self.test = None
        self.cat_features = None
        self.numeric_features = None
        self.pipe = None
        self.run_name = None
        self.algos = ["catboost", "xgboost", "randomforest"]

    def load_data(
        self,
        data_path: str,
        features: dict = auv_features,
        include: bool = 1,
        catboost_pool: bool = False,
        test_size: float = 0.2,
    ):
        """
        Loads data from csv or carto table for model tuning and evaluation

        Args:
            data_path (str): Path to csv or carto table name with
                enriched dataframe
            features (dict, optional): Dictionary object of features,
                categorical columns, and the target variable.
                Defaults to auv_features in global_constants.
            include (bool, optional): 1/0 indicator if feature list signifies
                features to include (1) or exclude (0) from modeling.
                Defaults to 1.
            catboost_pool (bool, optional): 1/0 indicator if catboost pool
                object should be returned (1).  Defaults to 0.
            test_size (float, optional): Size to use for
                train-test split (between 0 and 1). Defaults to 0.2.
        """
        # SSL fix for on-prem
        os.system(
            'echo quit | openssl s_client -showcerts -connect\
                carto.tools.bain.com:443 >> \
                $(python -c "import certifi; print(certifi.where())")'
        )

        # Read in data, first from carto account and
        # if not found then from local path
        try:
            store_df = cf.read_carto(data_path)
        except Exception:
            try:
                store_df = pd.read_csv(data_path)
            except Exception:
                raise ValueError(
                    f"Unable to find {data_path} in Carto account or in\
                    directory, please check edit the data path"
                )

        # Convert to a dataframe
        store_df = pd.DataFrame(store_df)
        self.data = store_df

        return self.build_data(
            store_df, features, include, catboost_pool, test_size
        )

    def build_data(
        self,
        df: pd.DataFrame,
        features: dict,
        include: bool = 1,
        catboost_pool: bool = False,
        test_size: float = 0.2,
        seed=9,
    ):
        """Prepares raw data for use in training models

        Args:
            df (pd.DataFrame): Dataframe to transform for modeling.
            features (dict): List of features in global_constants,
                signifying the target variable, features to include or exclude,
                and categorical columns.
            include (bool, optional): Signifies whether specified features
                should be included (1) or excluded (0) from the dataset.
                Defaults to 1.
            catboost_pool (bool, optional): Flag to create catboost pool
                object as well for training. Defaults to False.
            test_size (float, optional): Test size to use for train-
                test split.
            seed (int, optional): Seed to use for train-test split.
                Defaults to 9.
        """

        # Instantiate model features and target
        mod_features = features["features"]
        cat_features = features["categorical"]
        target = features["target"]

        if not isinstance(target, str):
            raise TypeError("Must provide string identifier of target column")
        if not isinstance(mod_features, list):
            raise TypeError(
                "Must provide list of features to include\
                                or exclude"
            )

        # Create y
        y = df[target]

        # Subset to columns of interest
        X = df.drop(columns=[target])

        # Uses provided feature list and include flag to subset columns
        if include:
            X = X.loc[:, X.columns.isin(mod_features)]
        else:
            X = X.loc[:, ~X.columns.isin(mod_features)]

        # Create categorical features using provided list,
        # if list is empty, use the dtypes to categorize
        if not cat_features:
            cat_features = list(X.select_dtypes("object"))

        numeric_features = [col for col in X if col not in cat_features]

        # Classify features
        self.cat_features = cat_features
        self.numeric_features = numeric_features

        # Re-order dataframe
        X = X[numeric_features + cat_features]

        # Create train-test split for modeling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # Create dictionary output
        model_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        # Set data
        self.train = [X_train, y_train]
        self.test = [X_test, y_test]

        # If catboost_pool, create pools for model
        if catboost_pool:
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            test_pool = Pool(X_test, y_test, cat_features=cat_features)

            # Add to dictionary output
            model_data["train_pool"] = train_pool
            model_data["test_pool"] = test_pool

            # Set data
            self.train_pool = train_pool
            self.test_pool = test_pool

        return df, model_data

    def gen_pipeline(self, X: pd.DataFrame, model_type):
        """Generates preprocessing pipeline to prepare data for modeling.
        Contains two different pipeline templates, one for Catboost and
        one for non-Catboost models due to differing support for cateogrical
        features.

        Args:
            X (DataFrame): Dataframe with predictors to be processed.
            model_type (str): String representing the model type to use.
        """

        # Generate pre-processing pipeline for non-catboost models
        # That do not support categorical features
        if model_type != "catboost":

            # Generate categorical transformer
            categorical_features = self.cat_features

            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            missing_values=None,
                            strategy="constant",
                            fill_value="missing",
                        ),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Generate numeric transformer for non-categorical columns
            numeric_features = self.numeric_features

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            )

            # Generate preprocessor with the two transformer objects
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            # Generate preprocessing pipeline with built in feature-selector
            sk_pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "feature_selection",
                        FeatureSelector(
                            CatBoostRegressorPipe(logging_level="Silent")
                        ),
                    ),
                ]
            )

        # Generate pre-processing pipeline for catboost models
        # That support categorical features and missing values
        elif model_type == "catboost":

            # Generate preprocessing pipeline with built in feature-selector
            sk_pipe = Pipeline(
                steps=[
                    (
                        "feature_selection",
                        FeatureSelector(
                            CatBoostRegressorPipe(logging_level="Silent")
                        ),
                    ),
                ]
            )

        else:
            raise ValueError("Please select a valid model type")

        return sk_pipe

    def gen_model(self, model_type: str, params):
        """Creates a model object based on user-selection and given parameters

        Args:
            model_type (str): One of xgboost, catboost, or randomforest
                regressors to train and track
        """

        if model_type.lower() == "xgboost":
            if params is not None:
                model = xgb.XGBRegressor(**params)
            else:
                model = xgb.XGBRegressor()

        elif model_type.lower() == "catboost":
            if params is not None:
                model = CatBoostRegressorPipe(**params, logging_level="Silent")
            else:
                model = CatBoostRegressorPipe(logging_level="Silent")

        elif model_type.lower() == "randomforest":
            if params is not None:
                model = RandomForestRegressor(**params)
            else:
                model = RandomForestRegressor()

        # Raise error if non-supported model_type provided
        else:
            raise ValueError(
                "Please select a model type between\
            'xgboost', 'catboost', or 'randomforest'"
            )

        self.model = model

        return model

    def mlflow_train(
        self,
        data: dict,
        model_type: str,
        experiment_id: str,
        params=None,
        run_name: str = None,
        nested: bool = False,
    ) -> Pipeline:
        """Trains different model types using mlflow logging capabilities

        Args:
            data (dict): Dictionary object generated from build_data which
                contains train-test split.
            model_type (str): One of xgboost, catboost, or randomforest
                regressors to train and track
            run_name (str, optional): Run name to track in mlflow.
                Defaults to model_type and datetime concatenation.
            run_hyperopt (bool, optional): Track whether or not this is
                used as part of hyperopt model hyperparam optimization.
                Defaults to False.

        Raises:
            ValueError: "Please select a model type between
                'xgboost', 'catboost', or 'randomforest'"
            ValueError: "Please provide data dictionary object with
                X_train, y_train, X_test, & y_test
                or train_pool & test_pool"

        Returns:
            Pipeline: sklearn pipeline containing data pre-processing steps
                and trained model of model_type
        """

        # Print mlflow tracking uri
        # print("MLflow Tracking URI:", mlflow.get_tracking_uri())

        # Set run_name to model_type and datetime concat by default
        if run_name is None:
            run_name = (
                model_type + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        print(f"Beginning run {run_name}")

        # Start tracking model run
        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment_id, nested=nested
        ) as run:
            # run_id = run.info.run_uuid
            # experiment_id = run.info.experiment_id
            # print("MLflow:")
            # print("  run_id:", run_id)
            # print("  experiment_id:", experiment_id)
            # print(
            #    "  experiment_name:", client.get_experiment(experiment_id).name
            # )

            # Instantiate model objects based on model_type

            # Generate pre-processing pipeline
            pipe = self.gen_pipeline(data["X_train"], model_type)

            # Generate model based on user selection
            model = self.gen_model(model_type, params)

            # Add model to pipeline
            pipe.steps.append(["regressor", model])

            # Cross-validate to optimize hyper parameters
            metrics_cv = None
            if params is not None:
                # Examine cross validation results
                cv_preds = cross_val_predict(
                    pipe, data["X_train"], data["y_train"]
                )
                # Obtain cross validation metrics for model optimization
                metrics_cv = {
                    f"val_{metric}": value
                    for metric, value in self.regression_metrics(
                        data["y_train"], cv_preds
                    ).items()
                }

            # Fit pipe on training data, predict on test set
            pipe.fit(data["X_train"], data["y_train"])

            preds = pipe.predict(data["X_test"])

            # Obtain test set metrics and create dictionary
            metrics_test = {
                f"test_{metric}": value
                for metric, value in self.regression_metrics(
                    data["y_test"], preds
                ).items()
            }

            if metrics_cv is not None:
                metrics = {**metrics_test, **metrics_cv}
            else:
                metrics = {**metrics_test}

            # mlflow logging
            if model_type.lower() == "catboost":
                mlflow.log_params(model.get_all_params())
            else:
                mlflow.log_params(model.get_params())

            mlflow.log_metrics(metrics)

            # SHAP report
            shap_img, shap_fi = self.shap_report(pipe, data, run_name)

            # Log SHAP summary
            mlflow.log_artifact(shap_img)
            mlflow.log_artifact(shap_fi)

            # save pipe
            self.pipe = pipe

            print(f"Test R2: {np.round(metrics['test_R2'], 3)}")
            # print(f"Trained {model_type} model, returning pipeline object")
            return pipe, metrics

    def check_missing(self, df: pd.DataFrame):
        """Generates a summary report on missing values in our dataset

        Args:
            df (pd.DataFrame): Input dataframe to use for analysis
        """

        total_cols = df.shape[1]
        missings = df.isna().sum()
        missings = pd.DataFrame(
            missings[missings > 0].sort_values(ascending=False),
            columns=["n_rows"],
        )

        missings["perc"] = (missings["n_rows"] / df.shape[0]) * 100

        cols_w_miss = len(missings)
        troublesome = missings[missings["perc"] > 5]

        print(
            f"{cols_w_miss} columns have missing data, representing "
            f"{np.round((cols_w_miss/total_cols)*100, 2)}% of "
            f"({total_cols}) total columns"
        )

        print(
            f"{troublesome.shape[0]} columns contain > 5% missing data, "
            "warranting further exploration"
        )

        return missings

    def shap_report(self, pipe, df, run_name):
        """Saves a shap summary plot for variable importance."""

        model = pipe[-1]
        X = df["X_train"]

        # Convert using pipeline object
        X = self.get_intermediate_df(pipe, X)

        # Set up the tree model explainer and compute the shap values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Generate summary, get a list of ranked drivers
        shap.summary_plot(shap_values, X, show=False)
        shap_file = PROJECT_DIR + f"/src/docs/shap/shap_plot_{run_name}.png"
        plt.savefig(shap_file, dpi=300, bbox_inches="tight")
        plt.close()

        fi_file = (
            PROJECT_DIR + f"/src/docs/shap/feature_importance_{run_name}.csv"
        )
        fi = pd.DataFrame(
            np.abs(shap_values).mean(axis=0), index=list(X)
        ).reset_index()
        fi.columns = ["col", "importance"]
        fi.sort_values("importance", ascending=False, inplace=True)
        fi.to_csv(fi_file, index=False)

        return shap_file, fi_file

    def get_corrs(
        self,
        df: pd.DataFrame,
        target: str,
        exclude_cols: list,
        n_display: int = 25,
    ):
        """[summary]

        Args:
            df (pd.DataFrame): [description]
            target (str): [description]
            exclude_cols (list): [description]
            n_display (int, optional): [description]. Defaults to 25.
        """

        cor_df = pd.DataFrame(df.corr()[target])

        cor_df["abs_cor"] = np.abs(cor_df[target])
        cor_df = cor_df.drop(index=target)
        cor_df = cor_df.drop(index=exclude_cols)

        print(cor_df.sort_values("abs_cor", ascending=False).head(n_display))

        pass

    def search_space(self):
        """Reads in pre-defined hyperparameters from global_constants and
            adjusts based on the model type

        Args:
            model_type (str): One of xgboost, catboost, or randomforest
                regressors to train and track

        Returns:
            [type]: [description]
        """

        space = hp.choice(
            "regressor_type",
            [
                {
                    "model_type": "xgboost",
                    "learning_rate": hp.quniform(
                        "learning_rate_xg", 0.05, 0.31, 0.05
                    ),
                    "max_depth": scope.int(
                        hp.quniform("max_depth_xg", 4, 12, 1)
                    ),
                    "min_child_weight": scope.int(
                        hp.quniform("min_child_weight_xg", 1, 10, 1)
                    ),
                    "colsample_bytree": hp.uniform(
                        "colsample_bytree_xg", 0.3, 1
                    ),
                    "subsample": hp.uniform("subsample_xg", 0.4, 1),
                    "n_estimators": scope.int(
                        hp.quniform("n_estimators_xg", 100, 1200, 50)
                    ),
                },
                # {
                #     "model_type": "randomforest",
                #     "max_depth": scope.int(
                #         hp.choice(
                #             "max_depth_rf", np.arange(10, 110, 10, dtype=int)
                #         )
                #     ),
                #     "n_estimators": scope.int(
                #         hp.quniform("n_estimators_rf", 100, 1200, 50)
                #     ),
                #     "max_features": hp.choice(
                #         "max_features_rf", ["auto", "sqrt"]
                #     ),
                #     "min_samples_split": hp.choice(
                #         "min_samples_split_rf", [2, 5, 10]
                #     ),
                #     "min_samples_leaf": hp.choice(
                #         "min_samples_leaf_rf", [1, 2, 4]
                #     ),
                #     "bootstrap": hp.choice("bootstrap_rf", [True, False]),
                # },
                {
                    "model_type": "catboost",
                    "learning_rate": hp.quniform(
                        "learning_rate_cb", 0.05, 0.5, 0.05
                    ),
                    "max_depth": scope.int(
                        hp.quniform("max_depth_cb", 3, 10, 1)
                    ),
                    "colsample_bylevel": hp.quniform(
                        "colsample_bylevel_cb", 0.3, 0.8, 0.1
                    ),
                },
            ],
        )

        return space

    def hyperopt_objective(self, df, metric, experiment_id):
        def train_func(params):
            model_type = params["model_type"]
            # Remove model_type parameter for model training
            del params["model_type"]

            pipe, metrics = self.mlflow_train(
                data=df,
                model_type=model_type,
                experiment_id=experiment_id,
                params=params,
                nested=True,
            )

            return {"status": STATUS_OK, "loss": metrics[metric]}

        return train_func

    def hyperopt_best(self, experiment_id, metric: str) -> None:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=experiment_id)

        # Filter to only runs with recorded metric
        runs_filtered = [run for run in runs if METRIC in run.data.metrics]

        best_run = min(runs_filtered, key=lambda run: run.data.metrics[metric])

        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])

        return best_run

    # # build up the finalized model
    # def hyperopt_final(best_run):

    #     model_type = params["model_type"]
    #         # Remove model_type parameter for model training
    #         del params["model_type"]

    #         pipe, metrics = self.mlflow_train(
    #             data=df,
    #             model_type=model_type,
    #             experiment_id=experiment_id,
    #             params=params,
    #             nested=True,
    #         )

    #     # Set up the model configuration
    #     model = xgboost.XGBClassifier(
    #         n_estimators=int(best_setting["n_estimators"]),
    #         max_depth=int(best_setting["max_depth"]),
    #         min_child_weight=best_setting["min_child_weight"],
    #         gamma=best_setting["gamma"],
    #         subsample=best_setting["subsample"],
    #         colsample_bytree=best_setting["colsample_bytree"],
    #         reg_alpha=best_setting["reg_alpha"],
    #         learning_rate=best_setting["learning_rate"],
    #     )

    #     # Fit model with given setting from the optimization space
    #     with mlflow.start_run(run_name="hyperopt tuning best result"):

    #         model.fit(X_train, y_train)

    #         y_pred = model.predict(X_test)
    #         precision = precision_score(y_test, y_pred)
    #         print("The precision score is {0}".format(precision))

    #         # Calculate the AUC for the current model
    #         probs = model.predict_proba(X_test)
    #         preds = probs[:, 1]
    #         fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    #         roc_auc = metrics.auc(fpr, tpr)

    #         plt.title("Receiver Operating Characteristic - Tuned Model")
    #         plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    #         plt.legend(loc="lower right")
    #         plt.plot([0, 1], [0, 1], "r--")
    #         plt.xlim([0, 1])
    #         plt.ylim([0, 1])
    #         plt.ylabel("True Positive Rate")
    #         plt.xlabel("False Positive Rate")

    #         roc_curve_file = "docs/plots/roc_curve_{0}.png".format(
    #             int(time.time())
    #         )
    #         plt.savefig(roc_curve_file)
    #         plt.close()

    #         # log hyperparameter setting into mlflow
    #         mlflow.log_params(best_setting)
    #         mlflow.log_metric("rocu_auc", roc_auc)
    #         mlflow.log_metric("precision", precision)
    #         mlflow.log_artifact(roc_curve_file)

    #     return model

    def get_feature_names(self, column_transformer):
        """Get feature names from all transformers.
        https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """

        def get_names(trans):
            if trans == "drop" or (
                hasattr(column, "__len__") and not len(column)
            ):
                return []
            if trans == "passthrough":
                if hasattr(column_transformer, "_df_columns"):
                    if (not isinstance(column, slice)) and all(
                        isinstance(col, str) for col in column
                    ):
                        return column
                    else:
                        return column_transformer._df_columns[column]
                else:
                    indices = np.arange(column_transformer._n_features)
                    return ["x%d" % i for i in indices[column]]
            if not hasattr(trans, "get_feature_names"):
                warnings.warn(
                    "Transformer %s (type %s) does not "
                    "provide get_feature_names. "
                    "Will return input column names if available"
                    % (str(name), type(trans).__name__)
                )
                # For transformers without a get_features_names method,
                # use the input names to the column transformer
                if column is None:
                    return []
                else:
                    return [f for f in column]

            return [f for f in trans.get_feature_names()]

        # Start of processing
        feature_names = []

        # Allow transformers to be pipelines. Pipeline steps are named
        # differently, so preprocessing is needed
        if type(column_transformer) == Pipeline:
            l_transformers = [
                (name, trans, None, None)
                for step, name, trans in column_transformer._iter()
            ]
        else:
            # For column transformers, follow the original method
            l_transformers = list(column_transformer._iter(fitted=True))

        for name, trans, column, _ in l_transformers:
            if type(trans) == Pipeline:
                # Recursive call on pipeline
                _names = self.get_feature_names(trans)
                # if pipeline has no transformer that returns names
                if len(_names) == 0:
                    _names = [f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans))

        return feature_names

    def get_intermediate_df(self, pipe: Pipeline, df: pd.DataFrame):
        """[summary]

        Args:
            pipe (Pipeline): [description]
            df (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """

        # Check if preprocessor in pipeline
        if "preprocessor" in list(pipe.named_steps):
            # Get feature names from pre-processing step
            features = self.get_feature_names(pipe.named_steps["preprocessor"])

            # Get support from feature selector
            sup = pipe.named_steps["feature_selection"].get_support()

            # Get list of prioritized column names
            selected_cols = list(np.array(features)[sup])

            # Create intermediate dataframe before prediction
            temp_df = pipe[:-1].transform(df)
            temp_df.columns = selected_cols

        else:
            # If no preprocessor, run feature selector (retains column names)
            temp_df = pipe[:-1].transform(df)

        return temp_df

    def regression_metrics(self, actual: pd.Series, pred: pd.Series) -> dict:
        """Return a collection of regression metrics as a Series.

        Args:
            actual: series of actual/true values
            pred: series of predicted values

        Returns:
            Series with the following values in a labeled index:
            MAE, RMSE
        """
        return {
            "MAE": metrics.mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(metrics.mean_squared_error(actual, pred)),
            "R2": metrics.r2_score(actual, pred),
        }


class CatBoostRegressorPipe(CatBoostRegressor):
    """Custom catboost model referencing
    https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7
    """

    def fit(self, X, y=None, **fit_params):

        X = pd.DataFrame(X)

        categorical_features = list(X.select_dtypes(object))

        return super().fit(
            X,
            y=y,
            cat_features=categorical_features,
            **fit_params,
        )


class FeatureSelector(SelectFromModel):
    """Custom feature selection referencing
    https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7
    """

    def transform(self, X):

        # Get indices of important features
        important_features_indices = list(self.get_support(indices=True))
        X = pd.DataFrame(X)

        # Select important features
        _X = X.iloc[:, important_features_indices].copy()

        return _X
