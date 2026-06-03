import pandas as pd
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# 1. Wczytanie danych
df_train = pd.read_parquet(config.traindatapath)
df_val = pd.read_parquet(config.valdatapath)

# 2. Preprocesor (zostawiamy bogate cechy)
preprocessor = ColumnTransformer(
    transformers=[
        ('code_word_tfidf', TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w\w+\b|[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
        ), 'code'),
        
        ('code_char_tfidf', TfidfVectorizer(
            max_features=20000,
            analyzer='char',
            ngram_range=(3, 5)
        ), 'code'),
        
        ('lang_enc', OneHotEncoder(handle_unknown='ignore'), ['language'])
    ]
)

# 3. Pipeline z poprawionym skalowaniem i balansem klas
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', MaxAbsScaler()), # KLUCZOWE: Skalowanie rzadkich macierzy TF-IDF
    ('classifier', LinearSVC(
        C=0.1,                 # Silniejsza regularyzacja, by nie overfittować
        class_weight='balanced', # KLUCZOWE: Automatyczne wyrównanie proporcji klas 0 i 1
        random_state=42, 
        max_iter=5000
    ))
])

# 4. Trenowanie
features = ['code', 'language']
print("Trenowanie zbalansowanego modelu...")
pipeline.fit(df_train[features], df_train['label'])
print("Gotowe!\n")

#### Ewaluacja ####
y_pred = pipeline.predict(df_val[features])

# Sprawdźmy F1-score dla obu klas (macro) oraz domyślne (dla klasy 1)
print(f"Domyślne F1 (klasa 1): {f1_score(df_val['label'], y_pred):.4f}")
print(f"Macro F1: {f1_score(df_val['label'], y_pred, average='macro'):.4f}")
print("\nPełny Raport:")
print(classification_report(df_val['label'], y_pred))