import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = "./train.csv"
df_train = pd.read_csv(file_path)
X_train, X_test, y_train, y_test = train_test_split(df_train.excerpt, df_train.target, test_size=0.33, random_state=42)

count_vect = CountVectorizer(ngram_range=(1, 2))
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

reg = LinearRegression()
reg.fit(X_train_tfidf, y_train)

rms_train = mean_squared_error(reg.predict(X_train_tfidf), y_train, squared=False)
rms_test = mean_squared_error(reg.predict(X_test_tfidf), y_test, squared=False)

print("RMSE Train: ", rms_train)
print("RMSE Test: ", rms_test)
