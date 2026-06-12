import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["attendance", "study", "assign", "lms"]
TARGET = "grade"
# L2 strength: larger alpha = smaller coefficients (less single-feature dominance).
RIDGE_ALPHA = 1.0


def build_frontend_features(df):
    yes_no = {"yes": 1.0, "no": 0.0}

    engineered = pd.DataFrame()
    max_absences = max(float(df["absences"].max()), 1.0)

    # Attendance proxy from absences (higher absences -> lower attendance)
    engineered["attendance"] = (1 - (df["absences"] / max_absences)) * 100
    engineered["attendance"] = engineered["attendance"].clip(0, 100)

    # studytime is 1..4 in this dataset; scale to 0..10 for UI parity.
    engineered["study"] = (df["studytime"] * 2.5).clip(0, 10)

    # Continuous assessment proxy from G1 and G2 (0..20 => 0..100).
    engineered["assign"] = (((df["G1"] + df["G2"]) / 2) * 5).clip(0, 100)

    # LMS/engagement proxy scaled to 0..20 for UI parity.
    internet = df["internet"].map(yes_no).fillna(0)
    higher = df["higher"].map(yes_no).fillna(0)
    activities = df["activities"].map(yes_no).fillna(0)
    schoolsup = df["schoolsup"].map(yes_no).fillna(0)
    famsup = df["famsup"].map(yes_no).fillna(0)
    attendance_signal = engineered["attendance"] / 100.0

    engineered["lms"] = (
        (internet * 5)
        + (higher * 4)
        + (activities * 4)
        + (schoolsup * 3)
        + (famsup * 2)
        + (attendance_signal * 2)
    ).clip(0, 20)

    # Final grade target in percentage (G3 is 0..20).
    engineered["grade"] = (df["G3"] * 5).clip(0, 100)
    return engineered


def print_metrics(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(
        f"{name}: "
        f"MAE={mean_absolute_error(y_true, y_pred):.3f}, "
        f"RMSE={rmse:.3f}, "
        f"R2={r2_score(y_true, y_pred):.3f}"
    )


def main():
    raw_df = pd.read_csv("student-por.csv")
    processed_df = build_frontend_features(raw_df).dropna()
    processed_df.to_csv("processed_student_features.csv", index=False)

    X = processed_df[FEATURES]
    y = processed_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    print_metrics("LinearRegression (baseline, unregularized)", y_test, baseline_pred)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    model.fit(X_train, y_train)
    ridge_pred = model.predict(X_test)
    print_metrics(
        f"Ridge (alpha={RIDGE_ALPHA}, scaled features)",
        y_test,
        ridge_pred,
    )

    joblib.dump(model, "model.pkl")
    print("Saved model.pkl using Pipeline(StandardScaler, Ridge)")
    print("Saved engineered dataset: processed_student_features.csv")


if __name__ == "__main__":
    main()