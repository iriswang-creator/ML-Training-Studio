import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Training Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
    }
    .main-header {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #00ff9f;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border-left: 4px solid #00ff9f;
    }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #00ff9f44;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: #00ff9f;
        color: #0f0f0f;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #00cc7a;
    }
    .step-badge {
        background: #00ff9f;
        color: #0f0f0f;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🧪 ML Training Studio</h1>
    <p style="margin:0.5rem 0 0 0; color:#aaa; font-family:'IBM Plex Mono',monospace; font-size:0.9rem;">
        Upload → Explore → Preprocess → Train → Evaluate
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "model_log" not in st.session_state:
    st.session_state.model_log = []

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload Data
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### <span class='step-badge'>01</span> Upload Data", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    st.session_state.df = pd.read_csv(uploaded)

df = st.session_state.df

if df is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("📊 Data Types & Missing Values"):
        info_df = pd.DataFrame({
            "Type": df.dtypes,
            "Missing": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(1)
        })
        st.dataframe(info_df, use_container_width=True)

    with st.expander("📈 Descriptive Statistics"):
        st.dataframe(df.describe(), use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Visualize
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### <span class='step-badge'>02</span> Visualize Data", unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Categorical"])

    with tab1:
        if numeric_cols:
            sel_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[sel_col].dropna(), bins=30, color="#00ff9f", edgecolor="#0f0f0f", alpha=0.85)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title(f"Distribution of {sel_col}", color="white")
            ax.set_xlabel(sel_col, color="white")
            ax.set_ylabel("Count", color="white")
            st.pyplot(fig)

    with tab2:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGn", ax=ax,
                        linewidths=0.5, linecolor="#0f0f0f")
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            plt.xticks(rotation=45, ha="right", color="white")
            plt.yticks(color="white")
            ax.set_title("Correlation Matrix", color="white")
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for correlation.")

    with tab3:
        if cat_cols:
            sel_cat = st.selectbox("Select categorical column", cat_cols, key="cat_col")
            counts = df[sel_cat].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(counts.index.astype(str), counts.values, color="#00ff9f", alpha=0.85)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title(f"Value Counts: {sel_cat}", color="white")
            st.pyplot(fig)
        else:
            st.info("No categorical columns found.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Configure Model
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### <span class='step-badge'>03</span> Configure & Train Model", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        target = st.selectbox("🎯 Target Column", df.columns.tolist())
        features = st.multiselect(
            "📦 Feature Columns",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target]
        )

        task_type = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)

    with col_right:
        if task_type == "Classification":
            model_choice = st.selectbox("Model", [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ])
        else:
            model_choice = st.selectbox("Model", [
                "Linear Regression",
                "Decision Tree",
                "Random Forest"
            ])

        val_method = st.radio("Validation", ["Train/Test Split", "5-Fold Cross-Validation"], horizontal=True)
        if val_method == "Train/Test Split":
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_seed = st.number_input("Random Seed", value=42, step=1)

    # ── Model name for logging
    model_name = st.text_input("Experiment Name (for tracking)", value=f"{model_choice} - Exp {len(st.session_state.model_log)+1}")

    # ── Train button
    train_btn = st.button("🚀 Train Model")

    if train_btn and features:
        # Prepare data
        X = df[features].copy()
        y = df[target].copy()

        # Simple preprocessing: encode categoricals, fill missing
        for col in X.select_dtypes(include="object").columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        if task_type == "Classification":
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

        # Build model
        models_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=int(random_seed)),
            "Decision Tree": DecisionTreeClassifier(random_state=int(random_seed)) if task_type == "Classification" else DecisionTreeRegressor(random_state=int(random_seed)),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=int(random_seed)) if task_type == "Classification" else RandomForestRegressor(n_estimators=100, random_state=int(random_seed)),
            "Linear Regression": LinearRegression(),
        }
        model = models_map[model_choice]

        st.divider()
        st.markdown("### <span class='step-badge'>04</span> Results", unsafe_allow_html=True)

        if val_method == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_seed)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", ax=ax,
                            linewidths=0.5, linecolor="#0f0f0f")
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#0f0f0f")
                ax.tick_params(colors="white")
                ax.set_title("Confusion Matrix", color="white")
                ax.set_xlabel("Predicted", color="white")
                ax.set_ylabel("Actual", color="white")
                st.pyplot(fig)

                with st.expander("Classification Report"):
                    st.text(classification_report(y_test, y_pred))

                log_entry = {"Name": model_name, "Model": model_choice, "Accuracy": round(acc, 4)}

            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{rmse:.4f}")
                m2.metric("MAE", f"{mae:.4f}")
                m3.metric("R²", f"{r2:.4f}")

                # Actual vs Predicted
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, y_pred, alpha=0.5, color="#00ff9f", s=20)
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                ax.plot(lims, lims, "r--", linewidth=1)
                ax.set_facecolor("#1a1a2e")
                fig.patch.set_facecolor("#0f0f0f")
                ax.tick_params(colors="white")
                ax.set_title("Actual vs Predicted", color="white")
                ax.set_xlabel("Actual", color="white")
                ax.set_ylabel("Predicted", color="white")
                st.pyplot(fig)

                log_entry = {"Name": model_name, "Model": model_choice, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R²": round(r2, 4)}

        else:  # Cross-validation
            scoring = "accuracy" if task_type == "Classification" else "r2"
            scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            st.metric(f"5-Fold CV Mean ({scoring})", f"{scores.mean():.4f}")
            st.metric("Std Dev", f"{scores.std():.4f}")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(1, 6), scores, color="#00ff9f", alpha=0.85)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title("CV Scores per Fold", color="white")
            ax.set_xlabel("Fold", color="white")
            ax.set_ylabel(scoring, color="white")
            st.pyplot(fig)

            log_entry = {"Name": model_name, "Model": model_choice, f"CV {scoring} Mean": round(scores.mean(), 4)}

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.4)))
            ax.barh(fi.index, fi.values, color="#00ff9f", alpha=0.85)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title("Feature Importance", color="white")
            st.pyplot(fig)

        # Log experiment
        st.session_state.model_log.append(log_entry)
        st.success(f"✅ Experiment '{model_name}' logged!")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5 — Experiment Tracking
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### <span class='step-badge'>05</span> Experiment Tracking", unsafe_allow_html=True)

    if st.session_state.model_log:
        log_df = pd.DataFrame(st.session_state.model_log)
        st.dataframe(log_df, use_container_width=True)

        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Experiment Log", csv, "experiment_log.csv", "text/csv")

        if st.button("🗑️ Clear Log"):
            st.session_state.model_log = []
            st.rerun()
    else:
        st.info("No experiments logged yet. Train a model to start tracking!")

else:
    st.info("👆 Upload a CSV file to get started.")