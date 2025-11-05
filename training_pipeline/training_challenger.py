import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    return df


@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient="records"))
    X_val = dv.transform(df_val[categorical + numerical].to_dict(orient="records"))

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv


@task(name="Train Two Models")
def train_models(X_train, X_val, y_train, y_val, dv):
    """Entrena dos modelos y devuelve sus métricas y run_ids"""
    results = []

    models = {
        "RandomForestRegressor": RandomForestRegressor(max_depth=10, n_estimators=80, random_state=42),
        "SVR": SVR(kernel="rbf", C=1.5, epsilon=0.2)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_run"):
            mlflow.set_tag("model_family", name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            sig = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model", signature=sig)
            run_id = mlflow.active_run().info.run_id
            results.append((name, rmse, run_id))

    return results


@task(name="Select Challenger")
def select_challenger(results):
    """Escoge el mejor modelo como challenger"""
    best = min(results, key=lambda x: x[1])
    best_name, best_rmse, best_run_id = best

    client = MlflowClient()
    model_name = "workspace.default.nyc-taxi-model-prefect"

    result = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=model_name)
    client.set_registered_model_alias(model_name, "Challenger", result.version)

    print(f"🏆 {best_name} es el nuevo @challenger con RMSE={best_rmse:.4f}")
    return best_run_id


@task(name="Compare and Promote")
def compare_and_promote(challenger_id: str, df_train: pd.DataFrame):
    """Compara el challenger con el champion actual y promueve si mejora"""
    client = MlflowClient()
    model_name = "workspace.default.nyc-taxi-model-prefect"

    # obtener version champion actual
    try:
        champion = client.get_model_version_by_alias(model_name, "Champion")
        champ_uri = f"models:/{model_name}@Champion"
        champ_model = mlflow.pyfunc.load_model(champ_uri)
        print(f"Modelo Champion actual: versión {champion.version}")
    except Exception:
        print("⚠️ No existe Champion actual, el Challenger será promovido automáticamente.")
        champion = None
        champ_model = None

    chall_model = mlflow.pyfunc.load_model(f"runs:/{challenger_id}/model")

    # revalúa sobre nuevo dataset
    df_reval = read_data("../data/green_tripdata_2025-03.parquet")
    X_train, X_reval, y_train, y_reval, dv = add_features(df_train, df_reval)

    y_chall = chall_model.predict(X_reval)
    rmse_chall = root_mean_squared_error(y_reval, y_chall)

    if champion:
        y_champ = champ_model.predict(X_reval)
        rmse_champ = root_mean_squared_error(y_reval, y_champ)
        print(f"RMSE Champion={rmse_champ:.4f}, RMSE Challenger={rmse_chall:.4f}")

        if rmse_chall < rmse_champ:
            print("🚀 El Challenger supera al Champion. Se promueve.")
            result = mlflow.register_model(f"runs:/{challenger_id}/model", model_name)
            client.set_registered_model_alias(model_name, "Champion", result.version)
        else:
            print("✅ El Champion actual sigue siendo el mejor.")
    else:
        # No hay champion previo
        result = mlflow.register_model(f"runs:/{challenger_id}/model", model_name)
        client.set_registered_model_alias(model_name, "Champion", result.version)
        print("🚀 Challenger promovido como primer Champion.")


@flow(name="nyc-taxi-experiment-prefect-v2")
def main_flow_v2(year: int = 2025, month_train: str = "01", month_val: str = "02"):
    load_dotenv(override=True)
    mlflow.set_tracking_uri("databricks")

    EXPERIMENT_NAME = "/Users/aclarapao@gmail.com/nyc-taxi-experiment-prefect"
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    results = train_models(X_train, X_val, y_train, y_val, dv)
    challenger_id = select_challenger(results)
    compare_and_promote(challenger_id, df_train)


if __name__ == "__main__":
    main_flow_v2()
