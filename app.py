import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF4B4B;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)


# Load models and scaler
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        dt_model = joblib.load('decision_tree_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return lr_model, dt_model, rf_model, scaler, feature_names, None
    except FileNotFoundError as e:
        return None, None, None, None, None, str(e)


# Prediction function
def predict_house_price(house_details, model, scaler, feature_names):
    # Add engineered features
    house_details['house_age'] = 2024 - house_details['built_year']
    house_details['is_renovated'] = 1 if house_details.get('renovation_year', 0) > 0 else 0
    house_details['years_since_renovation'] = (2024 - house_details.get('renovation_year', 0)
                                               if house_details.get('renovation_year', 0) > 0
                                               else house_details['house_age'])
    house_details['total_rooms'] = house_details['bedrooms'] + house_details['bathrooms']

    # Create dataframe
    df = pd.DataFrame([house_details])

    # One-hot encode
    for col in ['waterfront present', 'condition of the house', 'grade of the house']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    # Align with training features
    aligned = pd.DataFrame(columns=feature_names)
    aligned.loc[0] = 0
    for col in df.columns:
        if col in aligned.columns:
            aligned[col] = df[col].values[0]
    aligned = aligned.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale features
    scaled_features = [f for f in feature_names if not any(
        p in f for p in ['waterfront present_', 'condition of the house_', 'grade of the house_'])]
    aligned[scaled_features] = scaler.transform(aligned[scaled_features])

    # Predict
    return model.predict(aligned)[0]


# Load models
lr_model, dt_model, rf_model, scaler, feature_names, error = load_models()

# Header
st.title("  House Price Prediction System")
st.markdown("### Predict house prices based on various property features")

if error:
    st.error(f"  Error loading models: {error}")
    st.info("Please make sure you have run the training script and all model files (.pkl) are in the same directory.")
    st.stop()

st.success("  Models loaded successfully!")

# Sidebar for inputs
st.sidebar.header("   Property Details")
st.sidebar.markdown("---")

# Basic Information
st.sidebar.subheader(" Basic Information")
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1.0, 8.0, 2.0, 0.5)
floors = st.sidebar.slider("Number of Floors", 1.0, 3.5, 2.0, 0.5)

st.sidebar.markdown("---")

# Area Details
st.sidebar.subheader(" Area Details")
living_area = st.sidebar.number_input("Living Area (sq ft)", 500, 10000, 2000, 100)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 1000, 50000, 5000, 500)
basement_area = st.sidebar.number_input("Basement Area (sq ft)", 0, 5000, 500, 100)
area_no_basement = living_area - basement_area

st.sidebar.markdown("---")

# Property Characteristics
st.sidebar.subheader(" Property Characteristics")
built_year = st.sidebar.number_input("Built Year", 1900, 2024, 2010, 1)
renovation_year = st.sidebar.number_input("Renovation Year (0 if not renovated)", 0, 2024, 0, 1)
condition = st.sidebar.select_slider("Condition of House", options=[1, 2, 3, 4, 5], value=3,
                                     help="1=Poor, 5=Excellent")
grade = st.sidebar.select_slider("Grade of House", options=list(range(1, 14)), value=7,
                                 help="1=Poor, 13=Luxury")

st.sidebar.markdown("---")

# Location & Amenities
st.sidebar.subheader(" Location & Amenities")
waterfront = st.sidebar.checkbox("Waterfront Property")
views = st.sidebar.slider("Number of Views", 0, 4, 2)
schools_nearby = st.sidebar.slider("Schools Nearby", 0, 5, 2)
distance_airport = st.sidebar.slider("Distance from Airport (km)", 5, 100, 30, 5)

st.sidebar.markdown("---")

# Model Selection
st.sidebar.subheader("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Random Forest (Recommended)", "Linear Regression", "Decision Tree", "All Models (Average)"]
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Property Summary")

    # Display property details in a nice format
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.markdown("""
        <div class="metric-card">
            <h4> Rooms</h4>
            <p><strong>Bedrooms:</strong> {}</p>
            <p><strong>Bathrooms:</strong> {}</p>
            <p><strong>Floors:</strong> {}</p>
        </div>
        """.format(bedrooms, bathrooms, floors), unsafe_allow_html=True)

    with summary_col2:
        st.markdown("""
        <div class="metric-card">
            <h4> Areas</h4>
            <p><strong>Living:</strong> {} sq ft</p>
            <p><strong>Lot:</strong> {} sq ft</p>
            <p><strong>Basement:</strong> {} sq ft</p>
        </div>
        """.format(living_area, lot_area, basement_area), unsafe_allow_html=True)

    with summary_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Details</h4>
            <p><strong>Built:</strong> {}</p>
            <p><strong>Condition:</strong> {}/5</p>
            <p><strong>Grade:</strong> {}/13</p>
        </div>
        """.format(built_year, condition, grade), unsafe_allow_html=True)

with col2:
    st.subheader(" Quick Stats")
    house_age = 2024 - built_year
    total_rooms = bedrooms + bathrooms

    st.metric("House Age", f"{house_age} years")
    st.metric("Total Rooms", f"{total_rooms}")
    st.metric("Waterfront", "Yes" if waterfront else "No")
    st.metric("Schools Nearby", schools_nearby)

# Predict button
st.markdown("---")
if st.button(" Predict House Price", use_container_width=True):

    # Prepare input
    house_details = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'living_area': living_area,
        'lot_area': lot_area,
        'number of floors': floors,
        'number of views': views,
        'area_no_basement': area_no_basement,
        'basement_area': basement_area,
        'built_year': built_year,
        'renovation_year': renovation_year,
        'living_area_renov': living_area,
        'lot_area_renov': lot_area,
        'schools_nearby': schools_nearby,
        'distance_airport': distance_airport,
        'waterfront present': 1 if waterfront else 0,
        'condition of the house': condition,
        'grade of the house': grade,
    }

    # Make predictions
    with st.spinner(" Calculating predictions..."):
        if model_choice == "All Models (Average)":
            lr_pred = predict_house_price(house_details, lr_model, scaler, feature_names)
            dt_pred = predict_house_price(house_details, dt_model, scaler, feature_names)
            rf_pred = predict_house_price(house_details, rf_model, scaler, feature_names)
            final_prediction = (lr_pred + dt_pred + rf_pred) / 3

            # Show all predictions
            st.markdown("---")
            st.subheader(" Model Predictions Comparison")

            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

            with pred_col1:
                st.metric("Linear Regression", f"₹{lr_pred:,.0f}")
            with pred_col2:
                st.metric("Decision Tree", f"₹{dt_pred:,.0f}")
            with pred_col3:
                st.metric("Random Forest", f"₹{rf_pred:,.0f}")
            with pred_col4:
                st.metric("Average", f"₹{final_prediction:,.0f}", delta="Final")

            # Create comparison chart
            fig = go.Figure(data=[
                go.Bar(name='Model Predictions',
                       x=['Linear Regression', 'Decision Tree', 'Random Forest', 'Average'],
                       y=[lr_pred, dt_pred, rf_pred, final_prediction],
                       marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ])
            fig.update_layout(
                title="Prediction Comparison Across Models",
                yaxis_title="Predicted Price (₹)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Single model prediction
            if model_choice == "Random Forest (Recommended)":
                model = rf_model
            elif model_choice == "Linear Regression":
                model = lr_model
            else:
                model = dt_model

            final_prediction = predict_house_price(house_details, model, scaler, feature_names)

        # Display final prediction
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2> Predicted House Price</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">₹{final_prediction:,.2f}</h1>
            <p style="font-size: 1.2rem;">Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)

        # Price breakdown
        st.subheader(" Price Breakdown")
        breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)

        price_per_sqft = final_prediction / living_area

        with breakdown_col1:
            st.metric("Price per Sq Ft", f"₹{price_per_sqft:,.2f}")
        with breakdown_col2:
            st.metric("Total Living Area", f"{living_area:,} sq ft")
        with breakdown_col3:
            st.metric("Price per Room", f"₹{final_prediction / total_rooms:,.2f}")

        # Price factors visualization
        st.subheader(" Key Price Factors")
        factors_data = {
            'Factor': ['Location', 'Size', 'Condition', 'Age', 'Amenities'],
            'Impact': [
                (100 - distance_airport) / 100 * 20,  # Location impact
                min(living_area / 1000, 10) * 2,  # Size impact
                condition * 4,  # Condition impact
                max(20 - house_age / 5, 5),  # Age impact
                (views * 2 + schools_nearby * 1.5 + (10 if waterfront else 0)) / 2  # Amenities
            ]
        }

        fig_factors = px.bar(factors_data, x='Factor', y='Impact',
                             title='Estimated Impact on Price (%)',
                             color='Impact',
                             color_continuous_scale='Viridis')
        fig_factors.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_factors, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p> <strong>Tip:</strong> Adjust the parameters in the sidebar to see how different factors affect the house price!</p>
        <p>Built with  using Streamlit |  House Price Prediction Model</p>
    </div>
    """, unsafe_allow_html=True)