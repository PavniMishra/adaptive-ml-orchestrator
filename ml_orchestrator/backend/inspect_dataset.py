
import os
import pandas as pd
import ml_runner

print("\n========== DATASET INSPECTION ==========")

# ==========================================
# 1. BINARY CLASSIFICATION - SMS SPAM
# ==========================================

print("\n\n========================================")
print("DATASET: SMS Spam Collection")
print("CLASSIFICATION TYPE: BINARY")
print("========================================")

try:
    X, y = ml_runner._load_sms_data()

    print("\n1. DATASET SHAPE")
    print("Total messages:", len(X))
    print("Total labels:", len(y))

    print("\n2. FIRST 5 RECORDS")
    sms_df = pd.DataFrame({
        "Message": X,
        "Label": y
    })
    print(sms_df.head().to_string(index=False))

    print("\n3. UNIQUE CLASSES")
    print(pd.Series(y).unique())

    print("\n4. CLASS DISTRIBUTION")
    print(pd.Series(y).value_counts())

except Exception as e:
    print("SMS dataset could not be loaded:", e)


# ==========================================
# 2. MULTICLASS CLASSIFICATION - IRIS
# ==========================================

print("\n\n========================================")
print("DATASET: Iris Flower Dataset")
print("CLASSIFICATION TYPE: MULTICLASS")
print("========================================")

try:
    X, y, target_names, feature_names = ml_runner._load_iris_data()

    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df["Label"] = [target_names[i] for i in y]

    print("\n1. DATASET SHAPE")
    print("Total samples:", X.shape[0])
    print("Total features:", X.shape[1])

    print("\n2. FEATURE NAMES")
    print(feature_names)

    print("\n3. FIRST 5 RECORDS")
    print(iris_df.head().to_string(index=False))

    print("\n4. UNIQUE CLASSES")
    print(target_names)

    print("\n5. CLASS DISTRIBUTION")
    print(iris_df["Label"].value_counts())

except Exception as e:
    print("Iris dataset could not be loaded:", e)


# ==========================================
# 3. MULTILABEL CLASSIFICATION - MOVIES
# ==========================================

print("\n\n========================================")
print("DATASET: Movie Genre Dataset")
print("CLASSIFICATION TYPE: MULTILABEL")
print("========================================")

try:
    X, y, genre_names = ml_runner._load_movie_data()

    print("\n1. DATASET SHAPE")
    print("Total movies:", len(X))
    print("Total genre labels:", y.shape[1])

    print("\n2. AVAILABLE GENRES")
    print(genre_names)

    print("\n3. FIRST 5 MOVIE DESCRIPTIONS")

    for i in range(min(5, len(X))):
        genres = [
            genre_names[j]
            for j, value in enumerate(y[i])
            if value == 1
        ]

        print("\nMovie", i + 1)
        print("Description:", X[i][:250], "...")
        print("Genres:", genres)

    print("\n4. GENRE DISTRIBUTION")

    genre_counts = pd.Series(
        y.sum(axis=0),
        index=genre_names
    )

    print(genre_counts.sort_values(ascending=False))

    print("\n5. LABEL MATRIX")
    print(pd.DataFrame(y, columns=genre_names).head())

except Exception as e:
    print("Movie dataset could not be loaded:", e)


print("\n\n========== INSPECTION COMPLETE ==========")
