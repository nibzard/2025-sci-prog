# %%
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# %%
def create_detector(random_state=42):

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                max_features=60000,
                sublinear_tf=True,
                strip_accents="unicode"
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=30000,
                sublinear_tf=True
            )
        )
    ])

    model = Pipeline([
        ("features", features),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=random_state
            )
        )
    ])

    return model
# %%