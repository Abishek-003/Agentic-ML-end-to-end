import pandas as pd

from sklearn.decomposition import (
    PCA,
    FastICA
)


# =====================================================
# PCA
# =====================================================

def apply_pca(
        X,
        n_components=5
):

    n_components = min(
        n_components,
        X.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X
    )

    column_names = [

        f"PCA_{i+1}"

        for i in range(
            n_components
        )
    ]

    X_pca = pd.DataFrame(
        X_pca,
        columns=column_names,
        index=X.index
    )

    return (
        X_pca,
        pca
    )


# =====================================================
# PCA VARIANCE REPORT
# =====================================================

def get_pca_variance(
        pca_model
):

    variance_df = pd.DataFrame(
        {

            "Component":

            [
                f"PC{i+1}"

                for i in range(
                    len(
                        pca_model.explained_variance_ratio_
                    )
                )
            ],

            "Explained Variance":

            pca_model.explained_variance_ratio_,

            "Cumulative Variance":

            pca_model
            .explained_variance_ratio_
            .cumsum()
        }
    )

    return variance_df


# =====================================================
# AUTO PCA COMPONENTS
# =====================================================

def apply_auto_pca(
        X,
        variance_threshold=0.95
):

    pca = PCA(
        n_components=variance_threshold,
        random_state=42
    )

    X_pca = pca.fit_transform(
        X
    )

    column_names = [

        f"PCA_{i+1}"

        for i in range(
            X_pca.shape[1]
        )
    ]

    X_pca = pd.DataFrame(
        X_pca,
        columns=column_names,
        index=X.index
    )

    return (
        X_pca,
        pca
    )


# =====================================================
# ICA
# =====================================================

def apply_ica(
        X,
        n_components=5
):

    n_components = min(
        n_components,
        X.shape[1]
    )

    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=1000
    )

    X_ica = ica.fit_transform(
        X
    )

    column_names = [

        f"ICA_{i+1}"

        for i in range(
            n_components
        )
    ]

    X_ica = pd.DataFrame(
        X_ica,
        columns=column_names,
        index=X.index
    )

    return (
        X_ica,
        ica
    )


# =====================================================
# COMPARISON REPORT
# =====================================================

def dimensionality_reduction_report(
        X
):

    report = {

        "original_features":
        X.shape[1],

        "recommended_pca_components":

        max(
            2,
            int(
                X.shape[1] * 0.5
            )
        ),

        "recommended_ica_components":

        max(
            2,
            int(
                X.shape[1] * 0.5
            )
        )
    }

    return report