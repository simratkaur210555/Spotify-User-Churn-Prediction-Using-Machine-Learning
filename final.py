
# ============================================================
# SPOTIFY USER CHURN PREDICTION
# MODEL COMPARISON + DATA CLEANING + CALIBRATION
# THRESHOLD OPTIMIZATION + GRADIO
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

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


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "spotify_churn_dataset.csv"

MODEL_PATH = BASE_DIR / "spotify_churn_model.joblib"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\n========================================")
print("       SPOTIFY CHURN PREDICTION")
print("========================================")

print("\nDataset Loaded:", df.shape)


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

categorical_features = [
    "subscription_type",
    "country",
    "top_genre"
]

numerical_features = [
    "avg_daily_minutes",
    "number_of_playlists",
    "skips_per_day",
    "support_tickets",
    "days_since_last_login"
]

target = "churned"

required_columns = (
    categorical_features
    + numerical_features
    + [target]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )


# Remove identifier column

if "user_id" in df.columns:
    df = df.drop(columns=["user_id"])


# ============================================================
# 5. DATA CLEANING
# ============================================================

print("\n========================================")
print("             DATA CLEANING")
print("========================================")

# Convert numerical columns

for col in numerical_features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

# Remove rows with missing target

before_target_cleaning = len(df)

df = df.dropna(subset=[target])

print(
    "Rows removed due to missing target:",
    before_target_cleaning - len(df)
)

# Validate target values

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError(
        "The churned column must contain only 0 and 1."
    )

df[target] = df[target].astype(int)

# Remove rows containing negative numerical values

before_numeric_cleaning = len(df)

valid_rows = (
    df[numerical_features]
    .ge(0)
    .all(axis=1)
)

df = df.loc[valid_rows].copy()

removed_invalid_rows = (
    before_numeric_cleaning - len(df)
)

print(
    "Invalid numerical rows removed:",
    removed_invalid_rows
)

print("Dataset after cleaning:", df.shape)

if df.empty:
    raise ValueError(
        "No valid records remain after data cleaning."
    )

if df[target].nunique() < 2:
    raise ValueError(
        "Both churn classes must be present after cleaning."
    )


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[
    categorical_features + numerical_features
].copy()

y = df[target].copy()


# Actual categories for Gradio

category_options = {}

for col in categorical_features:

    category_options[col] = sorted(
        X[col].dropna().unique().tolist(),
        key=str
    )

    if not category_options[col]:
        raise ValueError(
            f"No valid categories found for {col}."
        )


# Actual numerical ranges after cleaning

feature_ranges = {}

for col in numerical_features:

    feature_ranges[col] = {

        "min": float(X[col].min()),

        "max": float(X[col].max())

    }


print("\nChurn Distribution:")

print(y.value_counts().sort_index())

print(
    "\nChurn Rate:",
    round(y.mean() * 100, 2),
    "%"
)


print("\nCleaned Feature Ranges:")

for col, limits in feature_ranges.items():

    print(
        f"{col}: "
        f"{limits['min']} to {limits['max']}"
    )


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples:", len(X_train))

print("Testing Samples:", len(X_test))


# ============================================================
# 8. PREPROCESSING
# ============================================================

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


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),

        (
            "numerical",
            numerical_transformer,
            numerical_features
        )

    ]

)


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(

        max_iter=2000,

        random_state=42

    ),


    "Random Forest": RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=3,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1

    ),


    "Extra Trees": ExtraTreesClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=3,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1

    )

}


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print("\n========================================")
print("             MODEL COMPARISON")
print("========================================")

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

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


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = max(

    model_scores,

    key=model_scores.get

)

print("\nSelected Model:", best_model_name)

best_classifier = models[best_model_name]

best_pipeline = Pipeline([

    ("preprocessor", preprocessor),

    ("classifier", best_classifier)

])


# ============================================================
# 12. CALIBRATION FUNCTION
# ============================================================

def create_calibrated_model(estimator):

    try:

        return CalibratedClassifierCV(

            estimator=estimator,

            method="sigmoid",

            cv=5

        )

    except TypeError:

        return CalibratedClassifierCV(

            base_estimator=estimator,

            method="sigmoid",

            cv=5

        )


# ============================================================
# 13. THRESHOLD OPTIMIZATION
# ============================================================

print("\n========================================")
print("        THRESHOLD OPTIMIZATION")
print("========================================")

# Split the training data again.
# This validation set is used only for threshold selection.

X_fit, X_validation, y_fit, y_validation = train_test_split(

    X_train,

    y_train,

    test_size=0.20,

    random_state=42,

    stratify=y_train

)

# Train a calibrated model on the fitting portion

threshold_model = create_calibrated_model(
    best_pipeline
)

threshold_model.fit(
    X_fit,
    y_fit
)

# Get validation probabilities

validation_probabilities = (
    threshold_model.predict_proba(X_validation)[:, 1]
)

# Test multiple classification thresholds

threshold_results = []

for threshold in np.arange(0.05, 0.96, 0.01):

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


# Select threshold with the highest validation F1.
# If tied, prefer the threshold with higher precision.

best_threshold_result = max(

    threshold_results,

    key=lambda item: (
        item["f1"],
        item["precision"]
    )

)

best_threshold = best_threshold_result["threshold"]

print(
    "Selected Classification Threshold:",
    round(best_threshold, 2)
)

print(
    "Validation F1 Score:",
    round(best_threshold_result["f1"], 4)
)

print(
    "Validation Precision:",
    round(best_threshold_result["precision"], 4)
)


# ============================================================
# 14. TRAIN FINAL CALIBRATED MODEL
# ============================================================

print("\n========================================")
print("        FINAL MODEL TRAINING")
print("========================================")

# Refit using the complete training dataset

calibrated_model = create_calibrated_model(
    best_pipeline
)

calibrated_model.fit(

    X_train,

    y_train

)

print("Final model training completed!")


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

print("\n========================================")
print("        FINAL MODEL PERFORMANCE")
print("========================================")

# Predict probabilities on untouched test data

y_prob = calibrated_model.predict_proba(
    X_test
)[:, 1]


# Predictions using the default threshold

default_predictions = (
    y_prob >= 0.50
).astype(int)


# Predictions using the optimized threshold

y_pred = (
    y_prob >= best_threshold
).astype(int)


# Default threshold metrics

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


# Optimized threshold metrics

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

brier = brier_score_loss(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# Majority-class baseline

baseline_predictions = np.zeros(
    len(y_test),
    dtype=int
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# Display results

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

        y_pred,

        zero_division=0

    )

)


print("\nConfusion Matrix:")

print(
    confusion_matrix(y_test, y_pred)
)


print("\nCross-Validation Results:")

for name, score in model_scores.items():

    print(
        f"{name}: ROC-AUC = {score:.4f}"
    )


# ============================================================
# 16. SAVE COMPLETE MODEL
# ============================================================

artifacts = {

    "model": calibrated_model,

    "best_model_name": best_model_name,

    "model_scores": model_scores,

    "threshold": best_threshold,

    "feature_ranges": feature_ranges,

    "category_options": category_options,

    "churn_rate": float(y_train.mean()),

    "numerical_features": numerical_features,

    "categorical_features": categorical_features

}


joblib.dump(

    artifacts,

    MODEL_PATH

)

print("\nModel saved successfully!")

print("Saved at:", MODEL_PATH)


# ============================================================
# 17. LOAD SAVED MODEL
# ============================================================

saved_artifacts = joblib.load(MODEL_PATH)

loaded_model = saved_artifacts["model"]

baseline_churn = saved_artifacts["churn_rate"]

loaded_threshold = saved_artifacts["threshold"]


# ============================================================
# 18. GRADIO SLIDER RANGE FUNCTION
# ============================================================

def slider_settings(feature, default_value):

    minimum = feature_ranges[feature]["min"]

    maximum = feature_ranges[feature]["max"]

    # Gradio needs a non-zero slider range

    if maximum <= minimum:

        maximum = minimum + 1

    value = min(
        max(default_value, minimum),
        maximum
    )

    return minimum, maximum, value


# ============================================================
# 19. PREDICTION FUNCTION
# ============================================================

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

        "avg_daily_minutes": avg_daily_minutes,

        "number_of_playlists": number_of_playlists,

        "top_genre": top_genre,

        "skips_per_day": skips_per_day,

        "support_tickets": support_tickets,

        "days_since_last_login": days_since_last_login

    }])


    # Predict churn probability

    probability = loaded_model.predict_proba(
        new_row
    )[0][1]

    churn_probability = probability * 100

    retention_probability = (
        1 - probability
    ) * 100


    # Apply optimized threshold

    if probability >= loaded_threshold:

        heading = "HIGHER CHURN RISK"

        color = "#FF5252"

        description = (
            "The model classifies this user as "
            "having a higher churn risk."
        )

    else:

        heading = "LOWER CHURN RISK"

        color = "#1DB954"

        description = (
            "The model classifies this user as "
            "having a lower churn risk."
        )


    # Check numerical inputs against observed ranges

    out_of_range = []

    for col in numerical_features:

        value = float(new_row[col].iloc[0])

        minimum = feature_ranges[col]["min"]

        maximum = feature_ranges[col]["max"]

        if value < minimum or value > maximum:

            out_of_range.append(col)


    # Build prediction result

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

            Classification Threshold:

            {loaded_threshold:.2f}

        </p>

        <p style="color:#AAAAAA;font-size:13px;">

            Dataset Baseline Churn Rate:

            {baseline_churn * 100:.2f}%

        </p>

    """


    # Show warning for unusual inputs

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

                One or more values are outside the

                range observed in the training dataset.

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


# ============================================================
# 20. GRADIO INTERFACE
# ============================================================

spotify_green = "#1DB954"

spotify_black = "#191414"


# Calculate slider ranges from cleaned data

minutes_min, minutes_max, minutes_default = slider_settings(
    "avg_daily_minutes", 45
)

playlists_min, playlists_max, playlists_default = slider_settings(
    "number_of_playlists", 3
)

skips_min, skips_max, skips_default = slider_settings(
    "skips_per_day", 2
)

tickets_min, tickets_max, tickets_default = slider_settings(
    "support_tickets", 0
)

login_min, login_max, login_default = slider_settings(
    "days_since_last_login", 5
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


# ============================================================
# 21. LAUNCH APPLICATION
# ============================================================

demo.launch(

    share=False,

    css=spotify_css

)
