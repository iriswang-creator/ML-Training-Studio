import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Training Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .main-header {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #00ff9f; padding: 1.5rem 2rem; border-radius: 8px;
        margin-bottom: 1rem; border-left: 4px solid #00ff9f;
    }
    .nav-bar {
        display: flex; gap: 0.5rem; margin-bottom: 1.5rem;
        padding: 0.75rem 1rem; background: #1a1a2e;
        border-radius: 8px; border: 1px solid #333;
        flex-wrap: wrap;
    }
    .nav-item {
        color: #aaa; font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem; padding: 0.3rem 0.7rem;
        border-radius: 4px; border: 1px solid #333;
        text-decoration: none; white-space: nowrap;
    }
    .nav-item:hover { color: #00ff9f; border-color: #00ff9f; }
    .section-intro {
        background: #1a1a2e; border-left: 3px solid #00ff9f44;
        padding: 0.75rem 1rem; border-radius: 0 6px 6px 0;
        margin-bottom: 1rem; color: #aaa; font-size: 0.9rem;
    }
    .stButton > button {
        background: #00ff9f; color: #0f0f0f;
        font-family: 'IBM Plex Mono', monospace; font-weight: 600;
        border: none; border-radius: 4px; padding: 0.5rem 2rem; width: 100%;
    }
    .stButton > button:hover { background: #00cc7a; }
    .step-badge {
        background: #00ff9f; color: #0f0f0f;
        font-family: 'IBM Plex Mono', monospace; font-weight: 600;
        padding: 0.2rem 0.6rem; border-radius: 4px;
        font-size: 0.8rem; margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🧪 ML Training Studio</h1>
    <p style="margin:0.4rem 0 0 0; color:#aaa; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;">
        Upload → Explore → Configure → Train → Evaluate → Track
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top Navigation Bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <span style="color:#00ff9f; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; padding:0.3rem 0.4rem; font-weight:600;">GO TO →</span>
    <a class="nav-item" href="#section-01">01 Upload</a>
    <a class="nav-item" href="#section-02">02 Visualize</a>
    <a class="nav-item" href="#section-03">03 Configure</a>
    <a class="nav-item" href="#section-04">04 Results</a>
    <a class="nav-item" href="#section-05">05 Experiments</a>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("df", None), ("model_log", []), ("trained_model", None), ("trained_model_name", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 01 — Upload Data
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<div id='section-01'></div>", unsafe_allow_html=True)
st.markdown("### <span class='step-badge'>01</span> Upload Data", unsafe_allow_html=True)
st.markdown("""
<div class="section-intro">
Upload a CSV file to get started. The app will automatically detect column types,
show a preview, and summarize missing values. All subsequent steps depend on the data loaded here.
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded:
    st.session_state.df = pd.read_csv(uploaded)

df = st.session_state.df

if df is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

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
    # SECTION 02 — Visualize
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div id='section-02'></div>", unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>02</span> Visualize Data", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-intro">
    Explore your data visually before modeling. Use the <b>Distribution</b> tab to check for skewness
    or outliers, <b>Correlation</b> to find related features, and <b>Categorical</b> to see class balance.
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Categorical"])

    with tab1:
        if numeric_cols:
            sel_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            fig, ax = plt.subplots(figsize=(7, 3))          # smaller plot
            ax.hist(df[sel_col].dropna(), bins=30, color="#00ff9f", edgecolor="#0f0f0f", alpha=0.85)
            ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title(f"Distribution of {sel_col}", color="white")
            ax.set_xlabel(sel_col, color="white"); ax.set_ylabel("Count", color="white")
            st.pyplot(fig, use_container_width=False)

    with tab2:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))           # smaller plot
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGn", ax=ax,
                        linewidths=0.5, linecolor="#0f0f0f")
            ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            plt.xticks(rotation=45, ha="right", color="white")
            plt.yticks(color="white")
            ax.set_title("Correlation Matrix", color="white")
            st.pyplot(fig, use_container_width=False)
        else:
            st.info("Need at least 2 numeric columns for correlation.")

    with tab3:
        if cat_cols:
            sel_cat = st.selectbox("Select categorical column", cat_cols, key="cat_col")
            counts = df[sel_cat].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(7, 3))           # smaller plot
            ax.barh(counts.index.astype(str), counts.values, color="#00ff9f", alpha=0.85)
            ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title(f"Value Counts: {sel_cat}", color="white")
            st.pyplot(fig, use_container_width=False)
        else:
            st.info("No categorical columns found.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 03 — Configure & Train
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div id='section-03'></div>", unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>03</span> Configure & Train Model", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-intro">
    Set the <b>random seed first</b> to ensure reproducibility, then select your target column,
    features, task type, model, and hyperparameters. Adjust the validation method and hit Train.
    </div>
    """, unsafe_allow_html=True)

    # ── Seed at the very top of section 03 ───────────────────────────────
    random_seed = st.number_input("🎲 Random Seed (set this first for reproducibility)", value=42, step=1)
    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        target = st.selectbox("🎯 Target Column", df.columns.tolist())
        features = st.multiselect(
            "📦 Feature Columns",
            [c for c in df.columns if c != target],
            default=[c for c in df.columns if c != target]
        )
        task_type = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)
        val_method = st.radio("Validation Method", ["Train/Test Split", "5-Fold Cross-Validation"], horizontal=True)
        if val_method == "Train/Test Split":
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    with col_right:
        # ── Model selection ───────────────────────────────────────────────
        if task_type == "Classification":
            model_choice = st.selectbox("Model", [
                "Logistic Regression", "Decision Tree", "Random Forest",
                "Gradient Boosting", "K-Nearest Neighbors", "SVM"
            ])
        else:
            model_choice = st.selectbox("Model", [
                "Linear Regression", "Ridge Regression", "Lasso Regression",
                "Decision Tree", "Random Forest", "Gradient Boosting",
                "K-Nearest Neighbors", "SVM"
            ])

        # ── Hyperparameters per model ─────────────────────────────────────
        st.markdown("**⚙️ Hyperparameters**")

        hp = {}
        if model_choice in ("Decision Tree", "Random Forest", "Gradient Boosting"):
            hp["max_depth"] = st.slider("Max Depth", 1, 20, 5)

        if model_choice in ("Random Forest", "Gradient Boosting"):
            hp["n_estimators"] = st.slider("N Estimators", 10, 500, 100, step=10)

        if model_choice == "Gradient Boosting":
            hp["learning_rate"] = st.select_slider("Learning Rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)

        if model_choice in ("Ridge Regression", "Lasso Regression", "Logistic Regression", "SVM"):
            hp["C_alpha"] = st.select_slider(
                "Regularization (C / alpha)",
                options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0
            )

        if model_choice == "K-Nearest Neighbors":
            hp["n_neighbors"] = st.slider("N Neighbors", 1, 30, 5)

    # ── Experiment name ───────────────────────────────────────────────────
    model_name = st.text_input(
        "🏷️ Experiment Name",
        value=f"{model_choice} - Exp {len(st.session_state.model_log)+1}"
    )

    train_btn = st.button("🚀 Train Model")

    if train_btn and features:
        # ── Preprocess ────────────────────────────────────────────────────
        X = df[features].copy()
        y = df[target].copy()

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        if task_type == "Classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y.astype(str))

        # ── Build model with hyperparameters ──────────────────────────────
        def build_model():
            rs = int(random_seed)
            md = hp.get("max_depth", None)
            ne = hp.get("n_estimators", 100)
            lr = hp.get("learning_rate", 0.1)
            ca = hp.get("C_alpha", 1.0)
            kn = hp.get("n_neighbors", 5)

            if task_type == "Classification":
                return {
                    "Logistic Regression":   LogisticRegression(C=ca, max_iter=1000, random_state=rs),
                    "Decision Tree":         DecisionTreeClassifier(max_depth=md, random_state=rs),
                    "Random Forest":         RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=rs),
                    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=ne, max_depth=md, learning_rate=lr, random_state=rs),
                    "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=kn),
                    "SVM":                   SVC(C=ca, random_state=rs),
                }[model_choice]
            else:
                return {
                    "Linear Regression":     LinearRegression(),
                    "Ridge Regression":      Ridge(alpha=ca),
                    "Lasso Regression":      Lasso(alpha=ca),
                    "Decision Tree":         DecisionTreeRegressor(max_depth=md, random_state=rs),
                    "Random Forest":         RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=rs),
                    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr, random_state=rs),
                    "K-Nearest Neighbors":   KNeighborsRegressor(n_neighbors=kn),
                    "SVM":                   SVR(C=ca),
                }[model_choice]

        model = build_model()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 04 — Results
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown("<div id='section-04'></div>", unsafe_allow_html=True)
        st.markdown("### <span class='step-badge'>04</span> Results", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-intro">
        Model performance metrics, visualizations, and feature importance.
        The test set predictions are shown at the bottom after the feature analysis.
        </div>
        """, unsafe_allow_html=True)

        log_entry = {"Name": model_name, "Model": model_choice,
                     "Seed": int(random_seed), **{k: v for k, v in hp.items()}}

        if val_method == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_seed)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.session_state.trained_model = model
            st.session_state.trained_model_name = model_name

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")
                log_entry["Accuracy"] = round(acc, 4)

                # Confusion matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                            cmap="YlGn", ax=ax, linewidths=0.5, linecolor="#0f0f0f")
                ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
                ax.tick_params(colors="white")
                ax.set_title("Confusion Matrix", color="white")
                ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
                st.pyplot(fig, use_container_width=False)

            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{rmse:.4f}")
                m2.metric("MAE",  f"{mae:.4f}")
                m3.metric("R²",   f"{r2:.4f}")
                log_entry.update({"RMSE": round(rmse,4), "MAE": round(mae,4), "R²": round(r2,4)})

            # ── Feature importance ────────────────────────────────────────
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, max(2.5, len(features) * 0.35)))
                ax.barh(fi.index, fi.values, color="#00ff9f", alpha=0.85)
                ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
                ax.tick_params(colors="white")
                ax.set_title("Feature Importance", color="white")
                st.pyplot(fig, use_container_width=False)

            # ── Decision Tree visualisation ───────────────────────────────
            if "Decision Tree" in model_choice:
                with st.expander("🌳 View Decision Tree Rules"):
                    tree_rules = export_text(model, feature_names=list(features), max_depth=4)
                    st.code(tree_rules)

            # ── Test set predictions — at the bottom ──────────────────────
            st.markdown("#### 🔍 Test Set Predictions")
            pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True)
            if task_type == "Regression":
                pred_df["Error"] = (pred_df["Predicted"] - pred_df["Actual"]).round(4)
                # Actual vs Predicted scatter
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred, alpha=0.5, color="#00ff9f", s=15)
                lims = [min(float(y_test.min()), float(y_pred.min())),
                        max(float(y_test.max()), float(y_pred.max()))]
                ax.plot(lims, lims, "r--", linewidth=1)
                ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
                ax.tick_params(colors="white")
                ax.set_title("Actual vs Predicted", color="white")
                ax.set_xlabel("Actual", color="white"); ax.set_ylabel("Predicted", color="white")
                st.pyplot(fig, use_container_width=False)
            else:
                with st.expander("Classification Report"):
                    st.text(classification_report(y_test, y_pred))
            st.dataframe(pred_df.head(50), use_container_width=True)

        else:  # Cross-validation
            scoring = "accuracy" if task_type == "Classification" else "r2"
            scores  = cross_val_score(model, X, y, cv=5, scoring=scoring)
            st.metric(f"5-Fold CV Mean ({scoring})", f"{scores.mean():.4f}")
            st.metric("Std Dev", f"{scores.std():.4f}")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(1, 6), scores, color="#00ff9f", alpha=0.85)
            ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white")
            ax.set_title("CV Scores per Fold", color="white")
            ax.set_xlabel("Fold", color="white"); ax.set_ylabel(scoring, color="white")
            st.pyplot(fig, use_container_width=False)
            log_entry[f"CV {scoring} Mean"] = round(scores.mean(), 4)

        # Log experiment
        st.session_state.model_log.append(log_entry)
        st.success(f"✅ Experiment '{model_name}' logged!")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 05 — Experiment Tracking
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div id='section-05'></div>", unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>05</span> Experiment Tracking", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-intro">
    Every trained model is logged here with its name, hyperparameters, and metrics.
    Compare runs side by side, download the log as CSV, or download the last trained model as a .pkl file.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.model_log:
        log_df = pd.DataFrame(st.session_state.model_log)
        st.dataframe(log_df, use_container_width=True)

        dl1, dl2, dl3 = st.columns(3)

        # Download experiment log
        with dl1:
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Experiment Log", csv,
                               "experiment_log.csv", "text/csv")

        # Download trained model
        with dl2:
            if st.session_state.trained_model is not None:
                model_bytes = pickle.dumps(st.session_state.trained_model)
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    model_bytes,
                    f"{st.session_state.trained_model_name}.pkl",
                    "application/octet-stream"
                )

        with dl3:
            if st.button("🗑️ Clear Log"):
                st.session_state.model_log = []
                st.session_state.trained_model = None
                st.rerun()
    else:
        st.info("No experiments logged yet. Train a model above to start tracking!")

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color:#aaa;">
        <p style="font-size:3rem;">📂</p>
        <p style="font-family:'IBM Plex Mono',monospace;">Upload a CSV file in Section 01 to get started.</p>
    </div>
    """, unsafe_allow_html=True)