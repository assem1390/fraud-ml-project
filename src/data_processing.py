from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

# Important: keep pandas output (prevents TargetEncoder column-name issues)
set_config(transform_output="pandas")

def build_preprocess(cat_cols, num_cols):
    """
    Returns a ColumnTransformer:
      - numerics: median impute
      - categoricals: most-freq impute + TargetEncoding (leak-safe in CV)
    """
    cat_pipe = SkPipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("te", TargetEncoder(cols=None, smoothing=0.2, min_samples_leaf=200, return_df=True)),
    ])
    num_pipe = SkPipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocess
