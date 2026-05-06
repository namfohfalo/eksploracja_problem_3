import pandas as pd
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

df_train = pd.read_parquet(config.traindatapath)
df_val = pd.read_parquet(config.valdatapath)
df_test = pd.read_parquet(config.testdatapath)


preprocessor = ColumnTransformer(
    transformers=[

        ('code_tfidf', TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w\w+\b|[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
        ), 'code'),
        
        ('lang_enc', OneHotEncoder(handle_unknown='ignore'), ['language'])
    ]
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', max_iter=100))
])


features = ['code', 'language']
pipeline.fit(df_train[features], df_train['label'])

#### Ewaluacja ####
y_pred = pipeline.predict(df_val[features])
print(f"F1: {f1_score(df_val['label'], y_pred):.4f}")
print("\nRaport:")
print(classification_report(df_val['label'], y_pred))
###################