
import pandas as pd
import numpy as np
import joblib
import gradio as gr

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "spotify_churn_dataset.csv"
MODEL_PATH = BASE_DIR / "spotify_churn_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
N_SPLITS = 5

CATEGORICAL_FEATURES = [
    "subscription_type",
    "country",
    "top_genre"
]

NUMERICAL_FEATURES = [
    "avg_daily_minutes",
    "number_of_playlists",
    "skips_per_day",
    "support_tickets",
    "days_since_last_login"
]

TARGET = "churned"

REQUIRED_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET]
)


# 2. DATA LOADING AND CLEANING

def load_and_clean_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    print("\n========================================")
    print("    SPOTIFY USER CHURN PREDICTION")
    print("========================================")

    print("\nDataset Loaded:", df.shape)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])

    print("\n========================================")
    print("             DATA CLEANING")
    print("========================================")

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    before_target_cleaning = len(df)

    df = df.dropna(subset=[TARGET])

    print(
        "Rows removed due to missing target:",
        before_target_cleaning - len(df)
    )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "The churned column must contain only 0 and 1."
        )

    df[TARGET] = df[TARGET].astype(int)

    before_numeric_cleaning = len(df)

    invalid_rows = (
        df[NUMERICAL_FEATURES]
        .lt(0)
        .any(axis=1)
    )

    df = df.loc[~invalid_rows].copy()

    print(
        "Invalid numerical rows removed:",
        before_numeric_cleaning - len(df)
    )

    print("Dataset after cleaning:", df.shape)

    if df.empty:
        raise ValueError(
            "No valid records remain after data cleaning."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Both churn classes must be present."
        )

    print("\nChurn Distribution:")
    print(df[TARGET].value_counts().sort_index())

    print(
        "\nChurn Rate:",
        round(df[TARGET].mean() * 100, 2),
        "%"
    )

    return df


# 3. TRAIN-TEST SPLIT

def split_data(df):

    X = df[
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    ].copy()

    y = df[TARGET].copy()

    if y.value_counts().min() < N_SPLITS:
        raise ValueError(
            "Not enough samples in the minority class "
            "for the selected cross-validation."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========================================")
    print("             DATA SPLITTING")
    print("========================================")

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    return X_train, X_test, y_train, y_test


# 4. APPLICATION INPUT METADATA

def get_input_metadata(X_train):

    category_options = {}
    feature_ranges = {}

    for col in CATEGORICAL_FEATURES:

        categories = sorted(
            X_train[col].dropna().unique().tolist(),
            key=str
        )

        if not categories:
            raise ValueError(
                f"No valid training categories found for {col}."
            )

        category_options[col] = categories

    for col in NUMERICAL_FEATURES:

        if X_train[col].notna().sum() == 0:
            raise ValueError(
                f"No valid numerical training values found for {col}."
            )

        feature_ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max())
        }

    print("\nTraining Feature Ranges:")

    for col, limits in feature_ranges.items():

        print(
            f"{col}: {limits['min']} to {limits['max']}"
        )

    return category_options, feature_ranges


# 5. PREPROCESSING PIPELINE

def create_preprocessor():

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    numerical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    return ColumnTransformer([
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            numerical_transformer,
            NUMERICAL_FEATURES
        )
    ])


# 6. MODEL DEFINITIONS

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# 7. CALIBRATION

def create_calibrated_model(estimator):

    calibration_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=calibration_cv
        )

    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=calibration_cv
        )


# 8. MODEL COMPARISON

def compare_models(X_train, y_train):

    print("\n========================================")
    print("             MODEL COMPARISON")
    print("========================================")

    models = get_models()

    preprocessor = create_preprocessor()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model_scores = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )

        mean_score = scores.mean()

        model_scores[name] = float(mean_score)

        print(
            f"{name}: Mean ROC-AUC = {mean_score:.4f}"
        )

    best_model_name = max(
        model_scores,
        key=model_scores.get
    )

    print("\nSelected Model:", best_model_name)

    best_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", models[best_model_name])
    ])

    return best_model_name, best_pipeline, model_scores


# 9. THRESHOLD OPTIMIZATION

def optimize_threshold(
    best_pipeline,
    X_train,
    y_train
):

    print("\n========================================")
    print("        THRESHOLD OPTIMIZATION")
    print("========================================")

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    threshold_model = create_calibrated_model(
        best_pipeline
    )

    threshold_model.fit(
        X_fit,
        y_fit
    )

    validation_probabilities = (
        threshold_model.predict_proba(X_validation)[:, 1]
    )

    threshold_results = []

    for threshold in np.linspace(0.05, 0.95, 91):

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        current_precision = precision_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )

        threshold_results.append({
            "threshold": float(threshold),
            "f1": float(current_f1),
            "precision": float(current_precision)
        })

    best_result = max(
        threshold_results,
        key=lambda item: (
            item["f1"],
            item["precision"]
        )
    )

    best_threshold = best_result["threshold"]

    print(
        "Selected Classification Threshold:",
        round(best_threshold, 2)
    )

    print(
        "Validation F1 Score:",
        round(best_result["f1"], 4)
    )

    print(
        "Validation Precision:",
        round(best_result["precision"], 4)
    )

    return best_threshold


# 10. FINAL MODEL EVALUATION

def evaluate_model(
    calibrated_model,
    X_test,
    y_test,
    best_model_name,
    best_threshold,
    model_scores,
    y_train
):

    print("\n========================================")
    print("        FINAL MODEL PERFORMANCE")
    print("========================================")

    y_prob = calibrated_model.predict_proba(
        X_test
    )[:, 1]

    default_predictions = (
        y_prob >= 0.50
    ).astype(int)

    optimized_predictions = (
        y_prob >= best_threshold
    ).astype(int)

    default_accuracy = accuracy_score(
        y_test,
        default_predictions
    )

    default_recall = recall_score(
        y_test,
        default_predictions,
        zero_division=0
    )

    default_f1 = f1_score(
        y_test,
        default_predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        optimized_predictions
    )

    precision = precision_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        optimized_predictions
    )

    brier = brier_score_loss(
        y_test,
        y_prob
    )

    majority_class = y_train.mode()[0]

    baseline_predictions = np.full(
        len(y_test),
        majority_class
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    print("\nSelected Model:", best_model_name)

    print(
        f"Selected Threshold: {best_threshold:.2f}"
    )

    print("\n--- Default Threshold (0.50) ---")

    print(
        f"Accuracy: {default_accuracy * 100:.2f}%"
    )

    print(
        f"Recall: {default_recall * 100:.2f}%"
    )

    print(
        f"F1 Score: {default_f1:.4f}"
    )

    print("\n--- Optimized Threshold ---")

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"Brier Score: {brier:.4f}"
    )

    print(
        f"Majority Baseline Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            optimized_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            optimized_predictions
        )
    )

    print("\nCross-Validation Results:")

    for name, score in model_scores.items():

        print(
            f"{name}: ROC-AUC = {score:.4f}"
        )


# 11. MODEL TRAINING AND SAVING

def train_model():

    df = load_and_clean_data()

    X_train, X_test, y_train, y_test = split_data(df)

    category_options, feature_ranges = get_input_metadata(
        X_train
    )

    best_model_name, best_pipeline, model_scores = compare_models(
        X_train,
        y_train
    )

    best_threshold = optimize_threshold(
        best_pipeline,
        X_train,
        y_train
    )

    print("\n========================================")
    print("        FINAL MODEL TRAINING")
    print("========================================")

    calibrated_model = create_calibrated_model(
        best_pipeline
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    print("Final model training completed!")

    evaluate_model(
        calibrated_model,
        X_test,
        y_test,
        best_model_name,
        best_threshold,
        model_scores,
        y_train
    )

    artifacts = {
        "model": calibrated_model,
        "best_model_name": best_model_name,
        "model_scores": model_scores,
        "threshold": best_threshold,
        "feature_ranges": feature_ranges,
        "category_options": category_options,
        "churn_rate": float(y_train.mean()),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }

    joblib.dump(
        artifacts,
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print("Saved at:", MODEL_PATH)

    return artifacts


# 12. GRADIO APPLICATION

def slider_settings(feature, feature_ranges, default_value):

    minimum = feature_ranges[feature]["min"]
    maximum = feature_ranges[feature]["max"]

    if maximum <= minimum:
        maximum = minimum + 1

    value = min(
        max(default_value, minimum),
        maximum
    )

    return minimum, maximum, value


def create_gradio_app(artifacts):

    loaded_model = artifacts["model"]

    best_model_name = artifacts["best_model_name"]

    loaded_threshold = artifacts["threshold"]

    feature_ranges = artifacts["feature_ranges"]

    category_options = artifacts["category_options"]

    baseline_churn = artifacts["churn_rate"]

    spotify_green = "#1DB954"
    spotify_black = "#191414"

    def predict(
        subscription_type,
        country,
        avg_daily_minutes,
        number_of_playlists,
        top_genre,
        skips_per_day,
        support_tickets,
        days_since_last_login
    ):

        new_row = pd.DataFrame([{
            "subscription_type": subscription_type,
            "country": country,
            "top_genre": top_genre,
            "avg_daily_minutes": avg_daily_minutes,
            "number_of_playlists": number_of_playlists,
            "skips_per_day": skips_per_day,
            "support_tickets": support_tickets,
            "days_since_last_login": days_since_last_login
        }])

        probability = loaded_model.predict_proba(
            new_row
        )[0][1]

        churn_probability = probability * 100

        retention_probability = (
            1 - probability
        ) * 100

        if probability >= loaded_threshold:

            heading = "HIGHER CHURN RISK"
            color = "#FF5252"

            description = (
                "The model classifies this user as "
                "having a higher churn risk."
            )

        else:

            heading = "LOWER CHURN RISK"
            color = spotify_green

            description = (
                "The model classifies this user as "
                "having a lower churn risk."
            )

        out_of_range = []

        for col in NUMERICAL_FEATURES:

            value = float(new_row[col].iloc[0])

            minimum = feature_ranges[col]["min"]
            maximum = feature_ranges[col]["max"]

            if value < minimum or value > maximum:
                out_of_range.append(col)

        result = f"""
        <div style="
            background-color:#191414;
            padding:25px;
            border-radius:15px;
            text-align:center;
            color:white;
        ">

            <h2 style="color:{color};">
                {heading}
            </h2>

            <p style="font-size:17px;">
                {description}
            </p>

            <h1 style="
                color:{color};
                font-size:42px;
            ">
                {churn_probability:.2f}%
            </h1>

            <p>Estimated Churn Probability</p>

            <hr style="border-color:#444;">

            <h3 style="color:#1DB954;">
                {retention_probability:.2f}%
            </h3>

            <p>Estimated Retention Probability</p>

            <p style="color:#AAAAAA;font-size:13px;">
                Classification Threshold: {loaded_threshold:.2f}
            </p>

            <p style="color:#AAAAAA;font-size:13px;">
                Training Dataset Baseline Churn Rate:
                {baseline_churn * 100:.2f}%
            </p>
        """

        if out_of_range:

            result += """
            <div style="
                background-color:#45351B;
                padding:12px;
                border-radius:10px;
                margin-top:15px;
            ">
                <p style="color:#FFD580;">
                    <b>Input Range Warning</b>
                </p>

                <p style="font-size:14px;">
                    One or more values are outside the range
                    observed in the training dataset.
                    This prediction may be unreliable.
                </p>
            </div>
            """

        result += """
            <p style="
                color:#AAAAAA;
                font-size:12px;
                margin-top:20px;
            ">
                Predictions are estimates based on historical
                dataset patterns, not guaranteed outcomes.
            </p>

        </div>
        """

        return result

    minutes_min, minutes_max, minutes_default = slider_settings(
        "avg_daily_minutes",
        feature_ranges,
        45
    )

    playlists_min, playlists_max, playlists_default = slider_settings(
        "number_of_playlists",
        feature_ranges,
        3
    )

    skips_min, skips_max, skips_default = slider_settings(
        "skips_per_day",
        feature_ranges,
        2
    )

    tickets_min, tickets_max, tickets_default = slider_settings(
        "support_tickets",
        feature_ranges,
        0
    )

    login_min, login_max, login_default = slider_settings(
        "days_since_last_login",
        feature_ranges,
        5
    )

    spotify_css = f"""
    body {{
        background-color: {spotify_black};
        color: white;
        font-family: 'Poppins', sans-serif;
    }}

    .gr-button {{
        background-color: {spotify_green} !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }}

    .gr-textbox,
    .gr-number,
    .gr-dropdown,
    .gr-slider {{
        border-radius: 10px !important;
    }}

    .gr-markdown {{
        text-align: center;
    }}

    h1 {{
        color: {spotify_green};
        font-weight: 900;
        font-size: 36px;
        text-align: center;
    }}
    """

    with gr.Blocks() as demo:

        gr.Markdown(
            "# 🎧 SPOTIFY USER CHURN PREDICTION 💚"
        )

        gr.Markdown(
            "Predict whether a Spotify user will "
            "**stay or leave** based on their listening habits."
        )

        gr.Markdown(
            f"Selected Machine Learning Model: **{best_model_name}**"
        )

        with gr.Row():

            with gr.Column():

                sub_type = gr.Dropdown(
                    choices=category_options["subscription_type"],
                    label="Subscription Type",
                    value=category_options["subscription_type"][0]
                )

                country = gr.Dropdown(
                    choices=category_options["country"],
                    label="Country",
                    value=(
                        "IN"
                        if "IN" in category_options["country"]
                        else category_options["country"][0]
                    )
                )

                genre = gr.Dropdown(
                    choices=category_options["top_genre"],
                    label="Top Genre",
                    value=(
                        "Pop"
                        if "Pop" in category_options["top_genre"]
                        else category_options["top_genre"][0]
                    )
                )

            with gr.Column():

                avg_minutes = gr.Slider(
                    minutes_min,
                    minutes_max,
                    value=minutes_default,
                    step=1,
                    label="Average Daily Minutes"
                )

                playlists = gr.Slider(
                    playlists_min,
                    playlists_max,
                    value=playlists_default,
                    step=1,
                    label="Number of Playlists"
                )

                skips = gr.Slider(
                    skips_min,
                    skips_max,
                    value=skips_default,
                    step=1,
                    label="Skips per Day"
                )

                tickets = gr.Slider(
                    tickets_min,
                    tickets_max,
                    value=tickets_default,
                    step=1,
                    label="Support Tickets"
                )

                last_login = gr.Slider(
                    login_min,
                    login_max,
                    value=login_default,
                    step=1,
                    label="Days Since Last Login"
                )

        btn = gr.Button("🔍 Predict")

        output = gr.HTML(
            label="Prediction Result"
        )

        btn.click(
            fn=predict,
            inputs=[
                sub_type,
                country,
                avg_minutes,
                playlists,
                genre,
                skips,
                tickets,
                last_login
            ],
            outputs=output
        )

        gr.Markdown("---")

        gr.Markdown(
            "💚 **Powered by Machine Learning & Gradio | Inspired by Spotify**"
        )

    return demo, spotify_css


# 13. APPLICATION ENTRY POINT

def main():

    artifacts = train_model()

    demo, spotify_css = create_gradio_app(
        artifacts
    )

    demo.launch(
        share=False,
        css=spotify_css
    )


if __name__ == "__main__":
    main()
