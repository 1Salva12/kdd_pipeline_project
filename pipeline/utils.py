import arff
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os

from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# =========================
# UTILIDAD PARA GRÁFICOS
# =========================
def get_base64_chart():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    return graph


# =========================
# PROCESAMIENTO PRINCIPAL
# =========================
def process_kdd_data():

    # --------- CARGA DATASET ----------
    path = os.path.join(
        settings.BASE_DIR,
        'datasets',
        'NSL-KDD',
        'KDDTrain+.arff'
    )

    with open(path, "r") as f:
        dataset = arff.load(f)
        attributes = [attr[0] for attr in dataset["attributes"]]
        df = pd.DataFrame(dataset["data"], columns=attributes)

    # --------- GRÁFICO 1: PROTOCOLOS ----------
    plt.figure(figsize=(10, 5))
    df["protocol_type"].value_counts().plot(
        kind="bar", edgecolor="black"
    )
    plt.title("Distribución por Tipo de Protocolo")
    plt.xticks(rotation=0)
    chart_protocol = get_base64_chart()

    # --------- GRÁFICO 2: CLASES ----------
    plt.figure(figsize=(10, 5))
    df["class"].value_counts().plot(
        kind="bar", edgecolor="black"
    )
    plt.title("Distribución de Clases")
    plt.xticks(rotation=0)
    chart_class = get_base64_chart()

    # --------- CODIFICAR CLASE ----------
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])

    # --------- DIVISIÓN TRAIN / VAL / TEST ----------
    train_set, temp_set = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df["protocol_type"]
    )

    val_set, test_set = train_test_split(
        temp_set,
        test_size=0.5,
        random_state=42,
        stratify=temp_set["protocol_type"]
    )

    # --------- GRÁFICO 3: SPLIT ----------
    plt.figure(figsize=(10, 5))
    split_counts = {
        'Entrenamiento': len(train_set),
        'Validación': len(val_set),
        'Prueba': len(test_set)
    }
    plt.bar(split_counts.keys(), split_counts.values(), edgecolor="black")
    plt.title("División de Datos (60/20/20)")
    chart_split = get_base64_chart()

    # --------- SEPARAR VARIABLES ----------
    X_train = train_set.drop("class", axis=1)
    y_train = train_set["class"].copy()

    # --------- SIMULAR VALORES NULOS ----------
    X_train.loc[
        (X_train["src_bytes"] > 400) & (X_train["src_bytes"] < 800),
        "src_bytes"
    ] = np.nan

    X_train.loc[
        (X_train["dst_bytes"] > 500) & (X_train["dst_bytes"] < 2000),
        "dst_bytes"
    ] = np.nan

    # --------- PIPELINES ----------
    num_attribs = X_train.select_dtypes(exclude=["object"]).columns
    cat_attribs = X_train.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
    ])

    # --------- PREPARACIÓN FINAL ----------
    X_train_prepared = full_pipeline.fit_transform(X_train)

    df_prep = pd.DataFrame(
        X_train_prepared,
        columns=full_pipeline.get_feature_names_out()
    ).head(10)

    # --------- RESPUESTA PARA LA VISTA ----------
    return {
        "chart_protocol": chart_protocol,
        "chart_class": chart_class,
        "chart_split": chart_split,
        "original_html": df.head(10).to_html(
            classes="table table-sm table-hover"
        ),
        "prepared_html": df_prep.to_html(
            classes="table table-dark table-striped table-sm"
        ),
        "stats": df.describe().to_html(
            classes="table table-bordered"
        )
    }
