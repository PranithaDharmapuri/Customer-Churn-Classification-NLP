import importlib

packages = [
    "jinja2","imblearn","imodels","joblib","keras","keras_tuner",
    "librosa","lightgbm","lmdb","matplotlib","matplotlib_inline","ngboost","nltk","numpy","cv2",
    "pandas","PIL","plotly","google.protobuf","pymysql","redis","regex","sklearn","scipy","seaborn",
    "spacy","spektral","tensorflow","textblob","tinydb","torch","tqdm","transformers","ultralytics",
    "urllib3","wordcloud","xgboost","zipp"
]

for p in packages:
    try:
        importlib.import_module(p)
        print("OK:", p)
    except Exception as e:
        print("FAIL:", p)
