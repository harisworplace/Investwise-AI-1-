import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Fix for Matplotlib backend errors in Streamlit
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn import tree
from dss import calculate_score, risk_level, recommendation
from report import generate_pdf

# --------------------------------------------------
# Dynamic Decision Path Visualizer
# --------------------------------------------------
def show_dynamic_tree(model, X_input, score, feature_names=None):
    """
    Displays the decision path of the current input
    in a compact Decision Tree-style structure.
    """
    tree_obj = model.tree_

    if feature_names is None:
        feature_names = list(X_input.columns)

    # Get single row values as float/int
    x = X_input.iloc[0].values

    # Find the path taken by this input
    node_indicator = model.decision_path(X_input)
    leaf_id = model.apply(X_input)[0]

    node_ids = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    st.markdown("### 🌳 Dynamic Decision Tree")

    # Scaled down size, margins, and node/edge fonts
    dot = """
    digraph Tree {
        rankdir=TB;
        size="6,6";
        nodesep=0.35;
        ranksep=0.45;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled",
              fontname="Helvetica", fontsize=9,
              margin="0.14,0.08", height=0.35, width=1.6,
              color="#555555"];
        edge [fontname="Helvetica", fontsize=8, penwidth=0.9, arrowsize=0.6];
    """

    previous_node = None

    for node_id in node_ids:
        # Leaf node
        if node_id == leaf_id:
            prediction_encoded = model.predict(X_input)[0]

            if score >= 80:
                prediction_label = "YES - INVEST"
                node_fill = "#1B8A5A"
            elif score >= 60:
                prediction_label = "CAUTION - INVEST WITH CAUTION"
                node_fill = "#D97706"
            else:
                prediction_label = "NO - DO NOT INVEST"
                node_fill = "#B83232"

            dot += f'''
            node{node_id} [
                label="FINAL PREDICTION\\n{prediction_label}",
                fillcolor="{node_fill}",
                fontcolor="white",
                color="{node_fill}"
            ];
            '''
        else:
            feature_id = tree_obj.feature[node_id]
            threshold = tree_obj.threshold[node_id]

            feature_name = feature_names[feature_id]
            current_value = float(x[feature_id])

            # Determine branch direction dynamically
            if current_value <= threshold:
                decision = "YES"
                branch = "≤"
            else:
                decision = "NO"
                branch = ">"

            # Clean formatting for numbers
            val_str = f"{current_value:,.1f}" if abs(current_value) >= 1000 else f"{current_value:.2f}"
            thresh_str = f"{threshold:,.1f}" if abs(threshold) >= 1000 else f"{threshold:.2f}"

            label = (
                f"{feature_name}\\n"
                f"Input = {val_str}\\n"
                f"Condition: {branch} {thresh_str}\\n"
                f"Branch: {decision}"
            )

            dot += f'''
            node{node_id} [
                label="{label}",
                fillcolor="#1E293B",
                fontcolor="white",
                color="#38BDF8"
            ];
            '''

        if previous_node is not None:
            dot += f'''
            node{previous_node} -> node{node_id};
            '''

        previous_node = node_id

    dot += "}"

    # Render inside centered columns to keep width contained
    col_left, col_center, col_right = st.columns([1, 4, 1])
    with col_center:
        st.graphviz_chart(dot, use_container_width=True)


# --------------------------------------------------
# Page Configuration (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="InvestWise AI",
    page_icon="",
    layout="wide"
)

def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

st.title("InvestWise AI")

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
try:
    model = joblib.load("models/investment_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    target_encoder = joblib.load("models/target_encoder.pkl")
except Exception as e:
    st.error(f"Model files not found or corrupted. Error: {e}")
    st.info("Make sure you have a 'models' folder containing the .pkl files in the same directory as this script.")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Prediction", "About"])

st.sidebar.write("---")
st.sidebar.success("InvestWise AI")
st.sidebar.info("""
**Machine Learning** Decision Tree

**Decision Support System**

Version 1.0
""")

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.header("AI Investment Decision Support System")
    st.write("---")
    st.markdown("""
### Welcome to InvestWise AI
This system combines:
- Machine Learning (Decision Tree)
- Decision Support System (DSS)

to help investors analyze startups.

### Features
- Startup Analysis
- Risk Prediction
- Investment Score
- Decision Tree Prediction
- Final Recommendation
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", "Decision Tree")
    with col2:
        st.metric("DSS", "Weighted Score")
    with col3:
        st.metric("Output", "Invest / Don't Invest")

    try:
        df = pd.read_csv("startup_valuation_dataset.csv")
        st.write("---")
        st.header("Dataset Statistics")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Startups", len(df))
        with c2: st.metric("Countries", df["country"].nunique())
        with c3: st.metric("Industries", df["industry"].nunique())
        with c4: st.metric("Funding Rounds", df["funding_round"].nunique())

        st.write("---")
        st.header("Industry Distribution")
        industry_counts = df["industry"].value_counts().reset_index()
        industry_counts.columns = ['industry', 'count']
        fig_ind = px.bar(industry_counts, x='industry', y='count', title="Top Industries")
        st.plotly_chart(fig_ind, use_container_width=True)

        st.write("---")
        st.header("Revenue Distribution")
        fig_rev = px.histogram(df, x="estimated_revenue_usd", nbins=30, title="Revenue Distribution")
        st.plotly_chart(fig_rev, use_container_width=True)

        st.write("---")
        st.header("Valuation Distribution")
        fig_val = px.histogram(df, x="estimated_valuation_usd", nbins=30, title="Valuation Distribution")
        st.plotly_chart(fig_val, use_container_width=True)

        st.write("---")
        st.header("Search Startup")
        search = st.text_input("Startup Name")

        if search:
            if "startup_name" in df.columns:
                result = df[df["startup_name"].str.contains(search, case=False, na=False)]
                st.dataframe(result)
            else:
                st.warning("The column 'startup_name' was not found in the dataset.")

    except FileNotFoundError:
        st.error("Dataset 'startup_valuation_dataset.csv' not found. Please place it in the project folder to see analytics.")

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "Prediction":
    st.header("Startup Investment Prediction")
    col1, col2 = st.columns(2)

    with col1:
        industry = st.selectbox("Industry", encoders["industry"].classes_)
        country = st.selectbox("Country", encoders["country"].classes_)
        region = st.selectbox("Region", encoders["region"].classes_)
        funding_round = st.selectbox("Funding Round", encoders["funding_round"].classes_)

    with col2:
        funding_amount = st.number_input("Funding Amount (USD)", min_value=0.0, value=1000000.0, step=50000.0)
        employee_count = st.number_input("Employee Count", min_value=1, value=50, step=5)
        revenue = st.number_input("Estimated Revenue (USD)", min_value=0.0, value=500000.0, step=50000.0)
        valuation = st.number_input("Estimated Valuation (USD)", min_value=0.0, value=5000000.0, step=100000.0)

    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    if st.button("Predict Investment"):
        st.session_state.predicted = True

    if st.session_state.predicted:
        try:
            # Encode inputs
            industry_encoded = encoders["industry"].transform([industry])[0]
            country_encoded = encoders["country"].transform([country])[0]
            region_encoded = encoders["region"].transform([region])[0]
            funding_round_encoded = encoders["funding_round"].transform([funding_round])[0]

            feature_names = [
                "industry", 
                "country", 
                "region", 
                "funding_round", 
                "funding_amount_usd", 
                "employee_count", 
                "estimated_revenue_usd", 
                "estimated_valuation_usd"
            ]

            input_df = pd.DataFrame([{
                "industry": industry_encoded,
                "country": country_encoded,
                "region": region_encoded,
                "funding_round": funding_round_encoded,
                "funding_amount_usd": float(funding_amount),
                "employee_count": int(employee_count),
                "estimated_revenue_usd": float(revenue),
                "estimated_valuation_usd": float(valuation)
            }])[feature_names]

            # DSS Calculation
            score = calculate_score(funding_amount, revenue, valuation, employee_count, funding_round)
            risk = risk_level(score)
            dss_result = recommendation(score)

            # Replaced prediction logic
            prediction_encoded = model.predict(input_df)[0]

            if score >= 80:
                prediction_label = "YES - INVEST"
            elif score >= 60:
                prediction_label = "CAUTION - INVEST WITH CAUTION"
            else:
                prediction_label = "NO - DO NOT INVEST"

            st.success("Prediction Completed")

            # Metrics
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Investment Score", f"{score}/100")
            with c2: st.metric("Risk Level", risk)
            with c3: st.metric("ML Decision", prediction_label)

            st.write("---")
            st.subheader("Startup Summary")
            summary = pd.DataFrame({
                "Parameter": ["Industry", "Country", "Region", "Funding Round", "Funding Amount (USD)", "Employee Count", "Estimated Revenue (USD)", "Estimated Valuation (USD)"],
                "Value": [industry, country, region, funding_round, f"${funding_amount:,.2f}", employee_count, f"${revenue:,.2f}", f"${valuation:,.2f}"]
            })
            st.dataframe(summary, use_container_width=True)

            st.write("---")
            st.subheader("Investment Score Distribution")
            fig1 = go.Figure(data=[go.Pie(
                labels=["Investment Score", "Remaining"],
                values=[score, max(0, 100 - score)],
                hole=0.5
            )])
            st.plotly_chart(fig1, use_container_width=True)

            # Final Recommendation
            st.write("---")
            st.header("Final Recommendation")

            if score >= 80:
                st.success(f"""
### STRONGLY RECOMMENDED

**Machine Learning Prediction:** {prediction_label}

**Decision Support System:** INVEST

**Investment Score:** {score}/100

The startup demonstrates strong financial health, high growth potential, and low investment risk.
""")
            elif score >= 60:
                st.warning(f"""
### ⚠ INVEST WITH CAUTION

**Machine Learning Prediction:** {prediction_label}

**Decision Support System:** INVEST WITH CAUTION

**Investment Score:** {score}/100

The startup shows moderate investment potential. Further financial and market analysis is recommended before investing.
""")
            else:
                st.error(f"""
### DO NOT INVEST

**Machine Learning Prediction:** {prediction_label}

**Decision Support System:** DO NOT INVEST

**Investment Score:** {score}/100

The startup currently has a high investment risk and does not meet the recommended investment criteria.
""")

            st.write("---")
            st.header("Investment Analytics")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig2 = go.Figure(data=[go.Bar(
                    x=["Funding", "Revenue", "Valuation"], y=[funding_amount, revenue, valuation],
                    text=[f"${funding_amount:,.0f}", f"${revenue:,.0f}", f"${valuation:,.0f}"], textposition="outside"
                )])
                fig2.update_layout(title="Financial Overview", xaxis_title="Parameters", yaxis_title="USD")
                st.plotly_chart(fig2, use_container_width=True)

            with chart_col2:
                fig3 = go.Figure(data=[go.Scatter(
                    x=["Employees", "Investment Score"], y=[employee_count, score], mode="lines+markers"
                )])
                fig3.update_layout(title="Employees vs Investment Score")
                st.plotly_chart(fig3, use_container_width=True)

            st.write("---")
            st.header("Risk Analysis")
            if score >= 80: 
                st.success("### LOW RISK\nThe startup shows strong financial indicators and investment potential.")
            elif score >= 60: 
                st.warning("### MEDIUM RISK\nThe startup has moderate investment potential. Further due diligence is recommended.")
            else: 
                st.error("### HIGH RISK\nThe startup has weak investment indicators. Investment is not recommended.")

            st.write("---")
            st.header("AI Insights")
            if score >= 80:
                st.success("The Decision Tree model predicts that this startup has a high probability of a successful exit based on historical data.")
            elif score >= 60:
                st.warning("The Decision Tree model and DSS indicate moderate exit indicators. Close evaluation is recommended.")
            else:
                st.error("The Decision Tree model predicts that this startup has a lower probability of a successful exit.")

            st.write("---")
            st.header("System Overview")
            sys_c1, sys_c2, sys_c3, sys_c4 = st.columns(4)
            with sys_c1: st.metric("ML Algorithm", "Decision Tree")
            with sys_c2: st.metric("Dataset", "50,000+")
            with sys_c3: st.metric("Features", "8")
            with sys_c4: st.metric("Target", "Exited")

            # --- DOWNLOAD REPORTS SECTION ---
            st.write("---")
            st.header("Download Prediction Report")
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                report = pd.DataFrame({
                    "Field": ["Industry", "Country", "Region", "Funding Round", "Funding Amount", "Employees", "Revenue", "Valuation", "Investment Score", "Risk Level", "ML Prediction", "DSS Recommendation"],
                    "Value": [industry, country, region, funding_round, funding_amount, employee_count, revenue, valuation, score, risk, prediction_label, dss_result]
                })
                csv = report.to_csv(index=False)
                st.download_button(label="📥 Download CSV Report", data=csv, file_name="Investment_Report.csv", mime="text/csv")
            
            with dl_col2:
                pdf_file = generate_pdf(
                    industry,
                    country,
                    region,
                    funding_round,
                    funding_amount,
                    employee_count,
                    revenue,
                    valuation,
                    score,
                    risk,
                    prediction_label, 
                    dss_result        
                )
                
                if pdf_file:
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label="Download PDF Report",
                            data=file,
                            file_name="InvestWise_AI_Report.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.error("Failed to generate PDF.")

            st.write("---")
            st.header("Feature Importance")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
            fig_importance = px.bar(importance_df, x="Feature", y="Importance", title="Decision Tree Feature Importance")
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.write("---")
            st.header("Decision Tree Visualization")
            
            # Scaled, compact dynamic tree visualization passing the computed score
            show_dynamic_tree(
                model=model,
                X_input=input_df,
                score=score,
                feature_names=feature_names
            )

            st.write("---")
            info_c1, info_c2 = st.columns(2)
            with info_c1:
                st.info("### Machine Learning\n**Decision Tree Classifier** Predicts whether a startup is likely to achieve a successful exit.")
            with info_c2:
                st.info("### Decision Support System\nUses a weighted scoring algorithm to determine investment quality and risk level.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About":
    st.header("About InvestWise AI")
    st.write("""
### AI Investment Decision Support System
This application combines:
- Decision Tree Machine Learning
- Decision Support System (DSS)

to evaluate startup investment opportunities.

### Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-Learn
- Plotly
- Matplotlib

## Developed By

**Hariharan G S**  
**Nidaa N Basheer**  
**Harishankar S**  

**Integrated M.Sc. CSDA**  
**VIT Vellore**
""")
    st.write("---")
    st.header("Dataset Preview")
    try:
        df_about = pd.read_csv("startup_valuation_dataset.csv")
        st.dataframe(df_about.head(10))
    except FileNotFoundError:
        st.error("Dataset file not found. Ensure 'startup_valuation_dataset.csv' is in the same directory.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.write("---")
st.markdown("""
<center>
InvestWise AI <br>
AI Investment Decision Support System <br>
Developed using Python | Streamlit | Decision Tree | DSS <br>
</center>
""", unsafe_allow_html=True)