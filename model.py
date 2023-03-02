import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

df_train = pd.read_csv("train.csv")
X_train, X_test, y_train, y_test = train_test_split(df_train.excerpt, df_train.target, test_size=0.33, random_state=42)

text_transformer = Pipeline(steps=[
    ('count_vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer', TfidfTransformer())
])

X_train = text_transformer.fit_transform(X_train)
X_test = text_transformer.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train) 

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('text_transformer.pkl', 'wb') as f:
    pickle.dump(text_transformer, f)

rms_train = mean_squared_error(model.predict(X_train), y_train, squared=False)
rms_test = mean_squared_error(model.predict(X_test), y_test, squared=False)

print("RMSE Train: ", rms_train)
print("RMSE Test: ", rms_test)
