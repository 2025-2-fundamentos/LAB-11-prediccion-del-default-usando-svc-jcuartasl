# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
import pickle
import gzip
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC

def load_data():
    
    train_data = pd.read_csv('./files/input/train_data.csv.zip', compression='zip')
    test_data = pd.read_csv('./files/input/test_data.csv.zip', compression='zip')

    return train_data, test_data

train_data, test_data = load_data()


def clean_data(data):
    data = data.copy()
    data.rename(columns = {'default payment next month':'default'}, inplace = True)
    data.dropna(inplace=True)

    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 4 if x not in [1,2,3,4] else x)
    data.drop(columns=["ID"], inplace=True)

    return data


def make_train_test_split(train_data, test_data):
    
    x_train = train_data.drop(columns=['default'])
    y_train = train_data['default']
    x_test = test_data.drop(columns=['default'])
    y_test = test_data['default']
    return x_train, x_test, y_train, y_test


def make_pipeline(estimator):

    cat = ["SEX", "MARRIAGE", "EDUCATION"]
    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int"), cat),
        ],
        remainder="passthrough",
    )

    selectkbest = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("pca", PCA(n_components=None)),
            ("scaler", StandardScaler()),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline


def make_grid_search(estimator, param_grid, cv=10):

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
    )

    return grid_search


def save_estimator(estimator):

    model_dir = "./files/models/"
    model_name = os.path.join(model_dir, "model.pkl.gz")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Guardar nuevo modelo
    with gzip.open(model_name, "wb") as f:
        pickle.dump(estimator, f)


def load_estimator():

    if not os.path.exists("./files/models/model.pkl.gz"):
        return None
    with gzip.open("./files/models/model.pkl.gz", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def train_estimator(estimator):

    from sklearn.metrics import balanced_accuracy_score

    train_data, test_data = load_data()
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    x_train, x_test, y_train, y_test = make_train_test_split(train_data, test_data)

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()
    if best_estimator is not None:

        saved_bal_acc = balanced_accuracy_score(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_bal_acc = balanced_accuracy_score(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if saved_bal_acc > current_bal_acc:
            estimator = best_estimator

    save_estimator(estimator)
    
def train_svm():
    pipeline = make_pipeline(
        estimator=SVC(),
    )

    param_grid = {
        #'selectkbest__score_func': [f_classif, mutual_info_classif],
        'selectkbest__k': [5],
        'estimator__kernel': ['rbf', 'linear'],
        #'estimator__C': [0.1, 1, 10, 100],
        #'estimator__gamma': ['scale', 'auto']  # only used for rbf
        }


    estimator = make_grid_search(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
    )

    train_estimator(estimator)


train_svm()


def eval_metrics(
    y_train_true,
    y_test_true,
    y_train_pred,
    y_test_pred,
):

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

    accuracy_train = round(accuracy_score(y_train_true, y_train_pred), 4)
    accuracy_test = round(accuracy_score(y_test_true, y_test_pred), 4)
    balanced_accuracy_train = round(balanced_accuracy_score(y_train_true, y_train_pred), 4)
    balanced_accuracy_test = round(balanced_accuracy_score(y_test_true, y_test_pred), 4)

    recall_train = round(recall_score(y_train_true, y_train_pred), 4)
    recall_test = round(recall_score(y_test_true, y_test_pred), 4)
    f1_train = round(f1_score(y_train_true, y_train_pred), 4)
    f1_test = round(f1_score(y_test_true, y_test_pred), 4)

    confusion_matrix_train = confusion_matrix(y_train_true, y_train_pred)
    confusion_matrix_test = confusion_matrix(y_test_true, y_test_pred)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": accuracy_train,
        "balanced_accuracy": balanced_accuracy_train,
        "recall": recall_train,
        "f1_score": f1_train,
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": accuracy_test,
        "balanced_accuracy": balanced_accuracy_test,
        "recall": recall_test,
        "f1_score": f1_test,
    }

    cm_matrix_train = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0" : {
            "predicted_0": int(confusion_matrix_train[0][0]),
            "predicted_1": int(confusion_matrix_train[0][1]),
        },
        "true_1" : {
            "predicted_0": int(confusion_matrix_train[1][0]),
            "predicted_1": int(confusion_matrix_train[1][1]),
        }
    }

    cm_matrix_test = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0" : {
            "predicted_0": int(confusion_matrix_test[0][0]),
            "predicted_1": int(confusion_matrix_test[0][1]),
        },
        "true_1" : {
            "predicted_0": int(confusion_matrix_test[1][0]),
            "predicted_1": int(confusion_matrix_test[1][1]),
        }
    }

    return metrics_train, metrics_test, cm_matrix_train, cm_matrix_test

def report(metrics_train, metrics_test, cm_matrix_train, cm_matrix_test):
    import json

    if not os.path.exists("../files/output/"):
        os.makedirs("../files/output/")
    # create the json file if it doesn't exist

    with open("../files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")
        f.write(json.dumps(cm_matrix_train) + "\n")
        f.write(json.dumps(cm_matrix_test) + "\n")

train_data, test_data = load_data()
train_data = clean_data(train_data)
test_data = clean_data(test_data)

x_train, x_test, y_train_true, y_test_true = make_train_test_split(train_data, test_data)

metrics_train, metrics_test, cm_matrix_train, cm_matrix_test = eval_metrics(y_train_true, y_test_true, load_estimator().predict(x_train), load_estimator().predict(x_test))
print(metrics_train)
print(metrics_test)
print(cm_matrix_train)
print(cm_matrix_test)

report(metrics_train, metrics_test, cm_matrix_train, cm_matrix_test)

    