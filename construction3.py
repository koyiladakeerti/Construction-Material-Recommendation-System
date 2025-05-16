#Importing required libraries
import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import joblib
import os
import bcrypt
import uuid
import base64
import google.generativeai as genai
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components


GEMINI_API_KEY = "AIzaSyAiWRuatBc3GlDdBkiRk7DwSZ6nPjbX8Js"  # Add gemini api key

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini API key error: {e}")
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

def floating_chatbot():
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""

    # Floating button and chatbot styling
    st.markdown("""
        <style>
        .floating-button {
            position: fixed;
            bottom: 80px;
            right: 20px;
            z-index: 10000;
        }

        .chat-container {
            position: fixed;
            bottom: 140px;
            right: 20px;
            width: 320px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            padding: 0px;
            z-index: 10000;
        }
        </style>
    """, unsafe_allow_html=True)

    # Floating button to open/close chatbot
    st.markdown('<div class="floating-button">', unsafe_allow_html=True)
    if st.button("💬", key="chatbot_toggle", help="Toggle Chatbot"):
        st.session_state.chatbot_open = not st.session_state.chatbot_open
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat container UI
    if st.session_state.chatbot_open:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("####Chatbot")
        user_input = st.text_area("Ask something:", value=st.session_state.chat_input, key="chat_input_text")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send", key="send_chat"):
                if user_input.strip():
                    try:
                        response = model.generate_content(user_input)
                        st.session_state.chat_response = response.text
                        st.session_state.chat_input = user_input
                    except Exception as e:
                        st.session_state.chat_response = f"Error: {e}"

        with col2:
            if st.button("Clear", key="clear_chat"):
                st.session_state.chat_input = ""
                st.session_state.chat_response = ""
                st.experimental_rerun()

        if st.session_state.chat_response:
            st.markdown("**Response:**")
            st.write(st.session_state.chat_response)

        st.markdown('</div>', unsafe_allow_html=True)


# Add this function near the top, after imports and before other functions
def filter_nearby_suppliers(df, user_location):
    if not user_location:
        return pd.DataFrame()
    # Case-insensitive partial match for city/region
    nearby_df = df[df['location'].str.contains(user_location, case=False, na=False)]
    return nearby_df

# Corrected get_live_price() function
def get_live_price(material_name="Steel"):
    import yfinance as yf

    # Map construction materials to reliable ticker symbols
    ticker_map = {
        "Concrete": "ULTRACEMCO.NS",
        "Steel": "TATASTEEL.NS",
        "Bricks": "ACC.NS"
    }

    try:
        symbol = ticker_map.get(material_name)
        if not symbol:
            return None

        ticker = yf.Ticker(symbol)
        # Fetch last 5 days to avoid empty results on market holidays
        data = ticker.history(period="5d")

        if data.empty or 'Close' not in data:
            return None

        # Use the last available non-null closing price
        last_valid_price = data['Close'].dropna().iloc[-1]
        return round(last_valid_price, 2)

    except Exception as e:
        print(f"DEBUG: Error fetching price for {material_name}: {e}")
        return None



# Set asyncio event loop policy for Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# File paths
BASE_DIR = Path(__file__).resolve().parent / "data"

MATERIALS_FILE = BASE_DIR / 'construction_materials_dataset.csv'
NEW_MATERIALS_FILE = BASE_DIR / 'new_materials.csv'
USERS_FILE = BASE_DIR / 'users.csv'

MODEL_FILES = {
    'model': BASE_DIR / 'rf_model.joblib',
    'le_material': BASE_DIR / 'le_material.joblib',
    'le_climate': BASE_DIR / 'le_climate.joblib',
    'le_project': BASE_DIR / 'le_project.joblib',
    'le_availability': BASE_DIR / 'le_availability.joblib',
    'le_budget': BASE_DIR / 'le_budget.joblib',
    'le_durability': BASE_DIR / 'le_durability.joblib',
    'le_sustainability': BASE_DIR / 'le_sustainability.joblib'
}


# Apply custom CSS
st.markdown("""
<style>
    /* Main Styles */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header Styles */
    .header {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E88E5;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Card Styles */
    .stCard {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1.5rem;
        /* Padding will be injected dynamically from Python */
    }

    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333;
    }

    .card-description {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Card Styles */
    .metric-card {
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        height: 100%;
    }

    .metric-card-good {
        border: 1px solid rgba(52, 199, 89, 0.2);
        background-color: rgba(52, 199, 89, 0.05);
        color: #34C759;
    }

    .metric-card-warning {
        border: 1px solid rgba(255, 149, 0, 0.2);
        background-color: rgba(255, 149, 0, 0.05);
        color: #FF9500;
    }

    .metric-card-danger {
        border: 1px solid rgba(255, 59, 48, 0.2);
        background-color: rgba(255, 59, 48, 0.05);
        color: #FF3B30;
    }

    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-unit {
        font-size: 0.8rem;
        opacity: 0.8;
    }

    .metric-status {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }

    /* Form Styles */
    .form-field {
        margin-bottom: 1rem;
    }

    .form-label {
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .normal-range {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }

    /* Button Styles */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Recommendation Styles */
    .recommendation-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .recommendation-icon {
        flex-shrink: 0;
    }

    .recommendation-text {
        font-size: 0.9rem;
    }

    /* Apply gradient background */
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Table styles for consistency */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    /* Tab styles */
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 0.5rem;
        margin-right: 0.5rem;
        padding: 0.5rem 1rem;
        color: #333;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }

    /* Help Floating Button */
    #help-toggle {
        display: none;
    }

    .floating-help-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1001;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        cursor: pointer;
        text-align: center;
        line-height: 60px;
    }

    .help-popup {
        display: none;
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 320px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        padding: 1rem;
        z-index: 1000;
        font-family: sans-serif;
    }

    #help-toggle:checked ~ .help-popup {
        display: block;
    }
    
</style>
<!-- Hidden checkbox -->
<input type="checkbox" id="help-toggle">

<!-- Label as floating button -->
<label for="help-toggle" class="floating-help-button">?</label>

<!-- Help popup content -->
<div class="help-popup">
  <h4 style="margin-top: 0;">Need Help?</h4>
  <p>Welcome to the Construction Material Recommendation System!</p>
  <ul style="padding-left: 1.2rem; font-size: 0.9rem;">
    <li><strong>Home:</strong> Explore platform features.</li>
    <li><strong>Register/Login:</strong> Create or access your account.</li>
    <li><strong>User Dashboard:</strong> Get material recommendations and compare them.</li>
    <li><strong>Supplier Dashboard:</strong> Manage materials and availability.</li>
    <li><strong>Admin Dashboard:</strong> Oversee materials and users.</li>
  </ul>
  <p style="font-size: 0.8rem; color: #666;">Click the ? button again to close this popup.</p>
</div>
""", unsafe_allow_html=True)

# Dynamic stCard padding injection
page = st.session_state.get("page", "home")
card_padding = "1.5rem" if page == "home" else "0rem"

st.markdown(f"""
<style>
    .stCard {{
        padding: {card_padding};
    }}
</style>
""", unsafe_allow_html=True)


# Header with logo
st.markdown("""
<div class="header">
    <div class="app-title">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
        Construction Material Recommendation System
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'logged_in': False,
        'user': None,
        'role': None,
        'page': 'home',
        'comparison_materials': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Password hashing utilities
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Load users from CSV
def load_users():
    try:
        if USERS_FILE.exists():
            return pd.read_csv(USERS_FILE)
        else:
            return pd.DataFrame(columns=['username', 'password', 'role'])
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame(columns=['username', 'password', 'role'])

# Save users to CSV
def save_users(users_df):
    try:
        users_df.to_csv(USERS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving users: {e}")

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        missing_files = []
        for name, file in MODEL_FILES.items():
            if not file.exists():
                missing_files.append(f"{name}: {file}")
        if not MATERIALS_FILE.exists():
            missing_files.append(f"Materials file: {MATERIALS_FILE}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files) + "\nPlease ensure all files are present and run ml1.ipynb to generate them.")

        model = joblib.load(MODEL_FILES['model'])
        le_material = joblib.load(MODEL_FILES['le_material'])
        le_climate = joblib.load(MODEL_FILES['le_climate'])
        le_project = joblib.load(MODEL_FILES['le_project'])
        le_availability = joblib.load(MODEL_FILES['le_availability'])
        le_budget = joblib.load(MODEL_FILES['le_budget'])
        le_durability = joblib.load(MODEL_FILES['le_durability'])
        le_sustainability = joblib.load(MODEL_FILES['le_sustainability'])
        df = pd.read_csv(MATERIALS_FILE)

        default_values = {
            'material_name': 'Unknown',
            'material_application': 'Unknown',
            'climate_zone': 'Unknown',
            'durability_requirement': '10-25 years',
            'budget_constraint': 'Standard',
            'sustainability_focus': 'Standard',
            'supplier_name': 'Unknown',
            'availability_status': 'In Stock',
            'project_type': 'Unknown',
            'category': 'Structural',
            'price_unit': '$/m²',
            'description': '',
            'image_url': '',
            'lead_time_days': 10,
            'project_specifications_match_score': 50.0,
            'installation_score': 50,
            'maintenance_score': 50,
            'location': 'Unknown'
        }
        for col, default in default_values.items():
            if col in df.columns:
                df[col] = df[col].replace('', default).fillna(default)
            else:
                df[col] = default

        # Assign sample locations to suppliers (replace with real data)
        sample_locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
        df['location'] = df['supplier_name'].apply(lambda x: np.random.choice(sample_locations))

        if NEW_MATERIALS_FILE.exists():
            new_df = pd.read_csv(NEW_MATERIALS_FILE)
            for col, default in default_values.items():
                if col in new_df.columns:
                    new_df[col] = new_df[col].replace('', default).fillna(default)
                else:
                    new_df[col] = default
            new_df['location'] = new_df['supplier_name'].apply(lambda x: np.random.choice(sample_locations))
            all_columns = list(df.columns)
            df = df.reindex(columns=all_columns).fillna({
                'description': '', 'category': 'Structural', 'price_unit': '$/m²',
                'installation_score': 50, 'maintenance_score': 50, 'image_url': '',
                'location': 'Unknown'
            })
            new_df = new_df.reindex(columns=all_columns).fillna({
                'description': '', 'category': 'Structural', 'price_unit': '$/m²',
                'installation_score': 50, 'maintenance_score': 50, 'image_url': '',
                'location': 'Unknown'
            })
            df = pd.concat([df, new_df], ignore_index=True)

        return model, le_material, le_climate, le_project, le_availability, le_budget, le_durability, le_sustainability, df
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None, None, None, None, None, None, None, None

# Collaborative_filtering function
def collaborative_filtering(df, preferred_material):
    try:
        user_item_matrix = df.pivot_table(index='material_name', columns='project_type', values='project_specifications_match_score', fill_value=0)
        similarity_matrix = cosine_similarity(user_item_matrix)
        similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        if preferred_material not in similarity_df.columns:
            return []
        similar_materials = similarity_df[preferred_material].sort_values(ascending=False)[1:6].index.tolist()
        return similar_materials
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return []

# Recommendation_function
def recommend_materials(model, df, le_material, le_climate, le_project, le_availability, le_budget, le_durability, le_sustainability,
                       climate, project, max_budget, min_durability, min_sustainability, preferred_material=None):
    try:
        climate_encoded = le_climate.transform([climate])[0]
        project_encoded = le_project.transform([project])[0]
        
        budget_levels = ['Economy', 'Standard', 'Premium', 'Luxury']
        durability_levels = ['5-10 years', '10-25 years', '25-50 years', '50+ years']
        sustainability_levels = ['Standard', 'Eco-friendly']
        
        budget_allowed = budget_levels[:budget_levels.index(max_budget) + 1]
        durability_allowed = durability_levels[:durability_levels.index(min_durability) + 1]
        sustainability_allowed = sustainability_levels[:sustainability_levels.index(min_sustainability) + 1]

        filtered_df = df[
            (df['climate_zone'] == climate) &
            (df['project_type'] == project) &
            (df['budget_constraint'].isin(budget_allowed)) &
            (df['durability_requirement'].isin(durability_allowed)) &
            (df['sustainability_focus'].isin(sustainability_allowed)) &
            (df['availability_status'] == 'In Stock')
        ].copy()

        if filtered_df.empty:
            return pd.DataFrame()

        # According to the dataset feature names are used during model training
        filtered_df['climate_zone_encoded'] = le_climate.transform(filtered_df['climate_zone'])
        filtered_df['project_type_encoded'] = le_project.transform(filtered_df['project_type'])
        filtered_df['durability_requirement_encoded'] = le_durability.transform(filtered_df['durability_requirement'])
        filtered_df['budget_constraint_encoded'] = le_budget.transform(filtered_df['budget_constraint'])
        filtered_df['sustainability_focus_encoded'] = le_sustainability.transform(filtered_df['sustainability_focus'])
        filtered_df['availability_status_encoded'] = le_availability.transform(filtered_df['availability_status'])

        features = [
            'climate_zone_encoded', 
            'project_type_encoded', 
            'durability_requirement_encoded',
            'budget_constraint_encoded', 
            'sustainability_focus_encoded', 
            'availability_status_encoded', 
            'lead_time_days', 
            'project_specifications_match_score'
        ]
        
        missing_features = [f for f in features if f not in filtered_df.columns]
        if missing_features:
            st.error(f"Missing features in data: {missing_features}")
            return pd.DataFrame()

        X = filtered_df[features]
        probs = model.predict_proba(X)
        
        if probs.size == 0:
            st.error("No valid predictions were made. Please adjust your input criteria.")
            return pd.DataFrame()

        max_probs = probs.max(axis=1)
        content_indices = np.argsort(max_probs)[-31:][::-1]
        content_recommendations = filtered_df.iloc[content_indices].copy()
        content_recommendations['content_score'] = max_probs[content_indices]

        if preferred_material:
            collab_materials = collaborative_filtering(df, preferred_material)
            collab_df = df[df['material_name'].isin(collab_materials) & (df['availability_status'] == 'In Stock')].copy()
        else:
            collab_df = pd.DataFrame()

        if not collab_df.empty:
            combined_df = pd.concat([content_recommendations, collab_df], ignore_index=True).drop_duplicates(subset='material_name')
            combined_df['hybrid_score'] = combined_df.get('project_specifications_match_score', 0) * 0.7 + combined_df.get('content_score', 0) * 30
            top_df = combined_df.sort_values(by='hybrid_score', ascending=False).head(31)
        else:
            top_df = content_recommendations

        required_columns = [
            'material_name', 'material_application', 'durability_requirement',
            'budget_constraint', 'sustainability_focus', 'supplier_name', 'supplier_contact_number', 'lead_time_days'
        ]
        optional_columns = ['description', 'category', 'price_unit', 'installation_score', 'maintenance_score', 'image_url']
        
        available_columns = [col for col in required_columns + optional_columns if col in top_df.columns]
        recommendations = top_df[available_columns]

        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()
    
#Home Page
def home_page():
    # Load and encode the hero image
    with open("images/mount.jpg", "rb") as f:
        data = f.read()
        encoded_img = base64.b64encode(data).decode()
    # Handle query params for navigation
    query_params = st.experimental_get_query_params()
    if "action" in query_params:
        action = query_params["action"][0]
        if action == "register":
            st.session_state.page = "register"
            st.experimental_set_query_params()  # Clear after reading
            st.rerun()
        elif action == "login":
            st.session_state.page = "login"
            st.experimental_set_query_params()
            st.rerun()

    # Style and layout
    st.markdown(f"""
    <style>
    .hero-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(to right, #eef5fc, #f6f9ff);
        padding: 4rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        flex-wrap: wrap;
    }}
    .hero-text {{
        max-width: 50%;
        flex: 1;
    }}
    .hero-text h1 {{
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-bottom: 1rem;
    }}
    .hero-text p {{
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }}
    .hero-buttons {{
        margin-top: 1.5rem;
    }}
    .hero-buttons a {{
        margin-right: 1rem;
        padding: 0.6rem 1.2rem;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        text-decoration: none;
        background-color: #1E88E5;
        color: white;
    }}
    .hero-buttons a:hover {{
        background-color: #1565C0;
    }}
    .secondary-btn {{
        background-color: white;
        color: #1E88E5;
        border: 1px solid #1E88E5;
    }}
    .secondary-btn:hover {{
        background-color: #f0f8ff;
    }}
    .hero-image {{
        flex: 1;
        max-width: 620px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-radius: 16px;
        overflow: hidden;
        margin-top: 2rem;
    }}
    @media (max-width: 768px) {{
        .hero-container {{
            flex-direction: column;
            align-items: flex-start;
            padding: 2rem;
        }}
        .hero-text {{
            max-width: 100%;
        }}
        .hero-text h1 {{
            font-size: 2rem;
        }}
        .hero-text p {{
            font-size: 0.9rem;
        }}
        .hero-buttons {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 1rem;
        }}
        .hero-buttons a {{
            margin-right: 0;
            width: 100%;
            text-align: center;
        }}
        .hero-image {{
            max-width: 100%;
            margin-top: 1.5rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-text">
            <h1>Construction Material Recommendation Platform</h1>
            <p>Discover the best materials for your construction projects with AI-powered recommendations tailored to your needs.</p>
            <div class="hero-buttons">
                <a href="/?action=register">Get Started</a>
                <a href="/?action=login" class="secondary-btn">Login to Account</a>
            </div>
        </div>
        <div class="hero-image">
            <img src="data:image/png;base64,{encoded_img}" style="width: 100%;" />
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                </svg>
                <div class="card-title">Material Recommendation</div>
            </div>
            <div class="card-description">
                Get personalized material suggestions based on project type, climate, budget, and sustainability preferences.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more →</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                    <circle cx="12" cy="10" r="3"></circle>
                </svg>
                <div class="card-title">Material Search</div>
            </div>
            <div class="card-description">
                Easily search for materials by name, application, or supplier to find exactly what you need.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more →</a>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stCard">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1E88E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z"></path>
                    <circle cx="12" cy="10" r="3"></circle>
                </svg>
                <div class="card-title">Supplier Management</div>
            </div>
            <div class="card-description">
                Suppliers can manage their material inventory and update availability in real-time.
            </div>
            <a href="#learn-more" style="color: #1E88E5; text-decoration: none; font-weight: 500;">Learn more →</a>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 2rem;">
        <h2 style="font-size: 1.75rem; font-weight: 600; color: #333;">Ready to optimize your material selection?</h2>
        <p style="color: #666; margin-bottom: 1.5rem;">
            Join our platform to access AI-driven material recommendations and streamline your construction projects.
        </p>
        <a href="#create-account" style="background-color: #1E88E5; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; font-weight: 500;">CREATE YOUR ACCOUNT</a>
    </div>
    """, unsafe_allow_html=True)

# Login page function
def login_page():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Login</div>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('<div class="form-label">Username</div>', unsafe_allow_html=True)
        username = st.text_input("", placeholder="Enter your username", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('<div class="form-label">Password</div>', unsafe_allow_html=True)
        password = st.text_input("", placeholder="Enter your password", type="password", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            users_df = load_users()
            user_row = users_df[users_df['username'] == username]
            if not user_row.empty and check_password(password, user_row.iloc[0]['password']):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.role = user_row.iloc[0]['role']
                st.session_state.page = 'dashboard'
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Register page function
def register_page():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Register</div>', unsafe_allow_html=True)
    
    with st.form("register_form"):
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('<div class="form-label">Username</div>', unsafe_allow_html=True)
        username = st.text_input(label="None", placeholder="Choose a username", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('<div class="form-label">Password</div>', unsafe_allow_html=True)
        password = st.text_input(label="None", placeholder="Choose a password", type="password", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-field">', unsafe_allow_html=True)
        st.markdown('<div class="form-label">Role</div>', unsafe_allow_html=True)
        role = st.selectbox(label="None", options=["User", "Supplier", "Admin"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if not username or not password:
                st.warning("Please fill in all fields.")
            else:
                users_df = load_users()
                if username in users_df['username'].values:
                    st.error("Username already exists.")
                else:
                    new_user = pd.DataFrame({
                        'username': [username],
                        'password': [hash_password(password)],
                        'role': [role]
                    })
                    updated_users = pd.concat([users_df, new_user], ignore_index=True)
                    save_users(updated_users)
                    st.success("Registered successfully! Please log in.")
                    st.session_state.page = 'login'
                    st.rerun()
    
    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# User dashboard function
def user_dashboard():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">User Dashboard - Welcome, {st.session_state.user}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Manage your construction projects and find the best materials</div>', unsafe_allow_html=True)
    
    try:
        model, le_material, le_climate, le_project, le_availability, le_budget, le_durability, le_sustainability, df = load_model_and_encoders()
        if df is None:
            st.error("Failed to load data. Please check the required files and try again.")
            return
        # Sample statistics about the system
        st.subheader("System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Materials", f"{len(df)}")
        
        with col2:
            st.metric("Material Categories", f"{df['material_application'].nunique()}")
        
        with col3:
            st.metric("Supported Climate Zones", f"{df['climate_zone'].nunique()}")
        
        with col4:
            st.metric("Project Types", f"{df['project_type'].nunique()}")
        
        tabs = st.tabs(["Material Recommendation", "Specify Your Project Requirements", "Material Search", "Compare Materials", "Projects", "Settings"])
        
        with tabs[0]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Project Specifications</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Enter your project details to get AI-powered material recommendations</div>', unsafe_allow_html=True)
            
            with st.form(key="recommendation_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Project Type</div>', unsafe_allow_html=True)
                    project_type = st.selectbox("", le_project.classes_, key="rec_project", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Climate Zone</div>', unsafe_allow_html=True)
                    climate_zone = st.selectbox("", le_climate.classes_, key="rec_climate", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Max Budget</div>', unsafe_allow_html=True)
                    budget_options = ['Economy', 'Standard', 'Premium', 'Luxury']
                    max_budget = st.selectbox("", budget_options, index=2, key="rec_budget", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                with col4:
                    material_application_options = ["Structural", "Finishing", "Insulation", "Decorative", "Flooring", "Facade", "Other"]
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Material Application</div>', unsafe_allow_html=True)
                    material_application = st.selectbox("", material_application_options, key="rec_material_app", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col5:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Min Durability</div>', unsafe_allow_html=True)
                    durability_options = ['5-10 years', '10-25 years', '25-50 years', '50+ years']
                    min_durability = st.selectbox("", durability_options, index=1, key="rec_durability", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col6:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Min Sustainability</div>', unsafe_allow_html=True)
                    sustainability_options = ['Standard', 'Eco-friendly']
                    min_sustainability = st.selectbox("", sustainability_options, index=0, key="rec_sustainability", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                col7, col8 = st.columns([1, 1])
                with col7:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Preferred Material (Optional)</div>', unsafe_allow_html=True)
                    preferred_material = st.selectbox("", ['None'] + list(df['material_name'].unique()), key="rec_preferred", label_visibility="collapsed")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col8:
                    additional_requirements = st.text_area("Additional Requirements", placeholder="Describe any specific requirements for your project...", key="rec_additional")
                
                col9, col10 = st.columns([1, 1])
                with col9:
                    reset_button = st.form_submit_button("Reset")
                with col10:
                    submit_button = st.form_submit_button("Get AI Recommendations")
                
                if reset_button:
                    keys_to_clear = ["rec_project", "rec_climate", "rec_budget", "rec_durability", "rec_sustainability", "rec_preferred", "rec_material_app", "rec_additional"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                
                if submit_button:
                    preferred = preferred_material if preferred_material != 'None' else None
                    recommendations = recommend_materials(
                        model, df, le_material, le_climate, le_project, le_availability, le_budget, le_durability, le_sustainability,
                        climate_zone, project_type, max_budget, min_durability, min_sustainability, preferred
                    )
                    
                    if recommendations.empty:
                        st.warning("No materials match the specified criteria. Try adjusting the filters.")
                    else:
                        st.markdown('<div class="card-title">Recommended Materials</div>', unsafe_allow_html=True)
                        format_dict = {
                            col: '{:.0f}' for col in ['lead_time_days', 'installation_score', 'maintenance_score'] if col in recommendations.columns
                        }
                        st.dataframe(recommendations.style.format(format_dict))
                        
                        st.markdown('<div class="card-title">Material Comparison</div>', unsafe_allow_html=True)
                        if 'project_specifications_match_score' in recommendations.columns:
                            chart_data = recommendations[['material_name', 'project_specifications_match_score']].set_index('material_name')
                            st.bar_chart(chart_data)
                        else:
                            st.warning("Project specifications match score not available for comparison.")
            
            # Section: Live Material Prices
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Live Material Prices</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Check the current market price of key construction materials</div>', unsafe_allow_html=True)

            material_options = ["Steel", "Concrete", "Bricks"]
            selected_material = st.selectbox("Select Material", material_options)

            if st.button("Get Live Price"):
             with st.spinner("Fetching live price..."):
              live_price = get_live_price(selected_material)

             if live_price is not None:
              st.success(f"Current Price of {selected_material}: ₹ {live_price}")
             else:
              st.warning("Price data not available. Showing approximate value.")
              fallback_prices = {"Steel": 68.5, "Concrete": 240.0, "Bricks": 10.5}
              approx_price = fallback_prices.get(selected_material, "N/A")
              st.info(f"Approximate Price of {selected_material}: ₹ {approx_price}")


            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Specify Your Project Requirements</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Provide detailed project specifications for tailored material recommendations</div>', unsafe_allow_html=True)
            
            # Create tabs for different project aspects
            req_tab1, req_tab2 = st.tabs(["Project Details", "Material Preferences"])
            
            with req_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    project_area = st.number_input(
                        "Project Area (sq ft):",
                        min_value=100,
                        max_value=100000,
                        value=1000,
                        step=100,
                        help="Enter the total area of your project in square feet."
                    )
                    
                    floors = st.number_input(
                        "Number of Floors:",
                        min_value=1,
                        max_value=100,
                        value=1,
                        step=1,
                        help="Enter the number of floors in your building project."
                    )
                    
                    timeline = st.slider(
                        "Project Timeline (months):",
                        min_value=1,
                        max_value=60,
                        value=12,
                        step=1,
                        help="Estimated timeline for project completion in months."
                    )
                
                with col2:
                    occupancy_type = st.selectbox(
                        "Occupancy Type:",
                        options=["Residential", "Commercial", "Industrial", "Educational", "Healthcare", "Other"],
                        help="Select the intended occupancy type for your project."
                    )
                    
                    exposure_level = st.select_slider(
                        "Exposure Level:",
                        options=["Low", "Medium", "High", "Extreme"],
                        value="Medium",
                        help="Select the level of exposure to elements the materials will face."
                    )
                    
                    special_requirements = st.multiselect(
                        "Special Requirements:",
                        options=["Fire Resistance", "Water Resistance", "Sound Insulation", "Thermal Insulation", 
                                "Chemical Resistance", "Impact Resistance", "Low Maintenance"],
                        default=[],
                        help="Select any special requirements for your project."
                    )
            
            with req_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    material_application = st.selectbox(
                        "Material Application:",
                        options=sorted(df['material_application'].unique()),
                        help="Select the primary application for the materials."
                    )
                    
                    preferred_material = st.selectbox(
                        "Preferred Material (Optional):",
                        options=['None'] + sorted(df['material_name'].unique()),
                        help="Optionally specify a material you prefer to include in recommendations."
                    )
                    
                    avoid_materials = st.multiselect(
                        "Materials to Avoid:",
                        options=sorted(df['material_name'].unique()),
                        default=[],
                        help="Select materials you would like to exclude from recommendations."
                    )
                
                with col2:
                    local_sourcing = st.checkbox(
                        "Prioritize Locally Sourced Materials",
                        value=False,
                        help="Check to prioritize materials that can be sourced locally."
                    )
                    
                    certifications = st.multiselect(
                        "Required Certifications:",
                        options=["LEED", "BREEAM", "Green Star", "Energy Star", "FSC", "Cradle to Cradle", "None"],
                        default=["None"],
                        help="Select any required material certifications for your project."
                    )
                    
                    aesthetic_preference = st.select_slider(
                        "Aesthetic Importance:",
                        options=["Low", "Medium", "High"],
                        value="Medium",
                        help="How important is the aesthetic aspect of materials for your project?"
                    )
            
            if st.button("Save Project Requirements"):
                project_details = {
                    'area': project_area,
                    'floors': floors,
                    'timeline': timeline,
                    'occupancy': occupancy_type,
                    'exposure': exposure_level,
                    'requirements': special_requirements,
                    'material_application': material_application,
                    'local_sourcing': local_sourcing,
                    'certifications': certifications,
                    'aesthetics': aesthetic_preference,
                    'preferred_material': preferred_material if preferred_material != 'None' else None,
                    'avoid_materials': avoid_materials
                }
                st.session_state.project_details = project_details
                st.success("Project requirements saved successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Material Search</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Search for materials by name, application, supplier, or find nearby suppliers by location</div>', unsafe_allow_html=True)
            
            with st.form("search_form"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Search Materials</div>', unsafe_allow_html=True)
                    search_term = st.text_input("", placeholder="Enter material name, application, or supplier", label_visibility="collapsed", key="search_term")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Your Location</div>', unsafe_allow_html=True)
                    user_location = st.text_input("", placeholder="Enter your city (e.g., Mumbai)", label_visibility="collapsed", key="user_location")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                submit_button = st.form_submit_button("Search")
                
                if submit_button:
                    if search_term:
                        search_df = df[
                            df['material_name'].str.contains(search_term, case=False, na=False) |
                            df['material_application'].str.contains(search_term, case=False, na=False) |
                            df['supplier_name'].str.contains(search_term, case=False, na=False) |
                            df['supplier_contact_number'].str.contains(search_term, case=False, na=False)
                        ]
                        display_columns = [col for col in [
                            'material_name', 'material_application', 'durability_requirement', 'budget_constraint',
                            'sustainability_focus', 'supplier_name', 'availability_status', 'location', 'supplier_contact_number'
                        ] if col in search_df.columns]
                        st.markdown('<div class="card-title">Search Results</div>', unsafe_allow_html=True)
                        st.dataframe(search_df[display_columns])
                    else:
                        st.warning("Please enter a search term.")
                    
                    if user_location:
                        nearby_suppliers = filter_nearby_suppliers(df, user_location)
                        if not nearby_suppliers.empty:
                            display_columns = [col for col in [
                                'supplier_name', 'location', 'material_name', 'material_application', 'availability_status', 'supplier_contact_number'
                            ] if col in nearby_suppliers.columns]
                            st.markdown('<div class="card-title">Nearby Suppliers</div>', unsafe_allow_html=True)
                            st.dataframe(nearby_suppliers[display_columns].drop_duplicates())
                        else:
                            st.warning(f"No suppliers found near {user_location}.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[3]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Material Comparison</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Compare different construction materials to find the best option for your project.</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-field">', unsafe_allow_html=True)
            st.markdown('<div class="form-label">Select Materials to Compare</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                material_options = sorted(df['material_name'].unique())
                selected_material = st.selectbox("", options=material_options, label_visibility="collapsed", key="compare_material_name")
                
                material_ids = df[df['material_name'] == selected_material]['material_id'].tolist()
                
                if material_ids:
                    material_id_options = material_ids
                    selected_material_id = st.selectbox("", options=material_id_options, label_visibility="collapsed", key="compare_material_id")
                    
                    if selected_material_id:
                        material_row = df[df['material_id'] == selected_material_id].iloc[0]
                        
                        st.markdown(f"**Application:** {material_row['material_application']}")
                        st.markdown(f"**Climate Zone:** {material_row['climate_zone']}")
                        st.markdown(f"**Durability:** {material_row['durability_requirement']}")
                        st.markdown(f"**Budget Category:** {material_row['budget_constraint']}")
                        st.markdown(f"**Sustainability:** {material_row['sustainability_focus']}")
            
            with col2:
                if st.button("Add to Comparison", key="add_to_compare"):
                    if selected_material_id not in [m[0] for m in st.session_state.comparison_materials]:
                        material_properties = {
                            'application': material_row['material_application'],
                            'climate': material_row['climate_zone'],
                            'durability': material_row['durability_requirement'],
                            'budget': material_row['budget_constraint'],
                            'sustainability': material_row['sustainability_focus'],
                            'supplier': material_row['supplier_name'],
                            'availability': material_row['availability_status'],
                            'lead_time': material_row['lead_time_days'],
                            'match_score': material_row['project_specifications_match_score'] * 100
                        }
                        
                        st.session_state.comparison_materials.append((selected_material_id, selected_material, material_properties))
                        st.success(f"Added {selected_material} (ID: {selected_material_id}) to comparison")
                    else:
                        st.warning(f"{selected_material} (ID: {selected_material_id}) is already in comparison")
            
            
            if st.session_state.comparison_materials:
                st.markdown(f'<div class="card-title">Comparing {len(st.session_state.comparison_materials)} Materials</div>', unsafe_allow_html=True)
                
                col_actions = st.columns(len(st.session_state.comparison_materials) + 1)
                
                with col_actions[-1]:
                    if st.button("Clear All", key="clear_compare"):
                        st.session_state.comparison_materials = []
                        st.rerun()
                
                for i, (mat_id, mat_name, _) in enumerate(st.session_state.comparison_materials):
                    with col_actions[i]:
                        if st.button(f"Remove {mat_name}", key=f"remove_compare_{i}"):
                            st.session_state.comparison_materials.pop(i)
                            st.rerun()
                
                materials_info = []
                attributes = set()
                
                for material_id, material_name, properties in st.session_state.comparison_materials:
                    material_row = df[df['material_id'] == material_id].iloc[0]
                    material_info = {
                        'Material': material_name,
                        'ID': material_id,
                        'Application': material_row.get('material_application', 'N/A'),
                        'Climate Zone': material_row.get('climate_zone', 'N/A'),
                        'Durability': material_row.get('durability_requirement', 'N/A'),
                        'Budget': material_row.get('budget_constraint', 'N/A'),
                        'Sustainability': material_row.get('sustainability_focus', 'Standard'),
                        'Supplier': material_row.get('supplier_name', 'N/A'),
                        'Availability': material_row.get('availability_status', 'N/A'),
                        'Lead Time (days)': material_row.get('lead_time_days', 'N/A'),
                        'Match Score (%)': material_row.get('project_specifications_match_score', 0) * 100
                    }
                    
                    for k, v in properties.items():
                        if k not in material_info:
                            material_info[k] = v
                            attributes.add(k)
                    
                    materials_info.append(material_info)
                
                tab1, tab2, tab3 = st.tabs(["Side-by-Side", "Charts", "Detailed View"])
                
                with tab1:
                    comparison_df = pd.DataFrame(materials_info).set_index('Material')
                    st.dataframe(comparison_df, use_container_width=True)
                
                with tab2:
                    if len(materials_info) > 0:
                        radar_attributes = st.multiselect(
                            "Select attributes to compare:",
                            options=['Match Score (%)', 'Lead Time (days)'],
                            default=['Match Score (%)', 'Lead Time (days)'],
                            key="radar_attributes"
                        )
                        
                        if radar_attributes:
                            fig = go.Figure()
                            
                            for material in materials_info:
                                fig.add_trace(go.Scatterpolar(
                                    r=[material.get(attr, 0) for attr in radar_attributes],
                                    theta=radar_attributes,
                                    fill='toself',
                                    name=material['Material']
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                    )),
                                showlegend=True,
                                title="Material Attribute Comparison"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        selected_attribute = st.selectbox(
                            "Select attribute to compare:",
                            options=['Match Score (%)', 'Lead Time (days)', 'Budget'],
                            index=0,
                            key="bar_attribute"
                        )
                        
                        comparison_data = pd.DataFrame({
                            'Material': [m['Material'] for m in materials_info],
                            selected_attribute: [m.get(selected_attribute, 0) for m in materials_info]
                        })
                        
                        fig = px.bar(
                            comparison_data,
                            x='Material',
                            y=selected_attribute,
                            color='Material',
                            title=f"Comparison by {selected_attribute}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    for i, material in enumerate(materials_info):
                        with st.expander(f"{material['Material']} (ID: {material['ID']})", expanded=i==0):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('<div class="card-title">Basic Information</div>', unsafe_allow_html=True)
                                st.markdown(f"**Material Type:** {material['Material']}")
                                st.markdown(f"**Application:** {material['Application']}")
                                st.markdown(f"**Climate Zone:** {material['Climate Zone']}")
                                st.markdown(f"**Durability:** {material['Durability']}")
                                st.markdown(f"**Budget Category:** {material['Budget']}")
                            
                            with col2:
                                st.markdown('<div class="card-title">Availability Information</div>', unsafe_allow_html=True)
                                st.markdown(f"**Supplier:** {material['Supplier']}")
                                st.markdown(f"**Availability Status:** {material['Availability']}")
                                st.markdown(f"**Lead Time:** {material['Lead Time (days)']} days")
                                st.markdown(f"**Match Score:** {material['Match Score (%)']}%")
                                st.markdown(f"**Sustainability:** {material['Sustainability']}")
                            
                            if material['Material'].lower() in ['steel', 'aluminum', 'copper']:
                                st.markdown('<div class="card-title">Market Information</div>', unsafe_allow_html=True)
                                st.markdown(f"**Current Market Price:** {get_live_price()}")
                                st.markdown("*Price data from stock market, updated in real-time.*")
                            
                            st.markdown('<div class="card-title">Similar Materials</div>', unsafe_allow_html=True)
                            similar_materials = df[
                                (df['material_name'] != material['Material']) & 
                                (df['material_application'] == material['Application'])
                            ].head(3)
                            
                            for _, similar in similar_materials.iterrows():
                                st.markdown(f"- **{similar['material_name']}** (ID: {similar['material_id']}) - "
                                            f"{similar['durability_requirement']}, {similar['budget_constraint']}")
            else:
                st.info("You haven't added any materials to compare. Select materials and add them to comparison.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[4]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Projects</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Manage your construction projects</div>', unsafe_allow_html=True)
            
            with st.form("project_form"):
                st.markdown('<div class="form-field">', unsafe_allow_html=True)
                st.markdown('<div class="form-label">Project Name</div>', unsafe_allow_html=True)
                project_name = st.text_input("", placeholder="Enter project name", label_visibility="collapsed", key="project_name")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="form-field">', unsafe_allow_html=True)
                st.markdown('<div class="form-label">Select Materials</div>', unsafe_allow_html=True)
                project_materials = st.multiselect("", df['material_name'].unique(), key="project_materials_multiselect", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                submit_button = st.form_submit_button("Save Project")
                
                if submit_button:
                    if project_name and project_materials:
                        st.success(f"Project '{project_name}' saved with materials: {', '.join(project_materials)}")
                    else:
                        st.warning("Please enter a project name and select at least one material.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[5]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Settings</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Update your account settings</div>', unsafe_allow_html=True)
            
            with st.form("settings_form"):
                st.markdown('<div class="form-field">', unsafe_allow_html=True)
                st.markdown('<div class="form-label">New Password</div>', unsafe_allow_html=True)
                new_password = st.text_input("", type="password", placeholder="Enter new password", label_visibility="collapsed", key="settings_password")
                st.markdown('</div>', unsafe_allow_html=True)
                
                submit_button = st.form_submit_button("Update Password")
                
                if submit_button:
                    if new_password:
                        users_df = load_users()
                        users_df.loc[users_df['username'] == st.session_state.user, 'password'] = hash_password(new_password)
                        save_users(users_df)
                        st.success("Password updated successfully.")
                    else:
                        st.warning("Please enter a new password.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Admin dashboard function 
def admin_dashboard():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">Admin Dashboard - Welcome, {st.session_state.user}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Manage materials and user accounts</div>', unsafe_allow_html=True)
    
    try:
        _, _, _, _, _, _, _, _, df = load_model_and_encoders()
        if df is None:
            st.error("Failed to load material data. Please check the required files.")
            return
        
        tabs = st.tabs(["Material Management", "User Management"])
        
        with tabs[0]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Material Management</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">View, add, edit, or delete materials</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-field">', unsafe_allow_html=True)
            st.markdown('<div class="form-label">Action</div>', unsafe_allow_html=True)
            action = st.selectbox("", ["View Materials", "Add Material", "Edit Material", "Delete Material"], label_visibility="collapsed", key="admin_action")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if action == "View Materials":
                available_columns = [col for col in [
                    'material_id', 'material_name', 'material_application', 'climate_zone', 'durability_requirement',
                    'budget_constraint', 'sustainability_focus', 'supplier_name', 'supplier_contact_number', 'availability_status',
                    'lead_time_days', 'category', 'price', 'price_unit', 'description', 'installation_score',
                    'maintenance_score', 'image_url'
                ] if col in df.columns]
                df['live_price'] = df['material_name'].apply(lambda x: get_live_price("TATASTEEL.NS") if 'steel' in x.lower() else 'N/A')
                available_columns.append('live_price')
                st.dataframe(df[available_columns])
            
            elif action == "Add Material":
                with st.form("admin_add_material_form"):
                    material_id = f"M{str(uuid.uuid4())[:4].upper()}"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Material Name</div>', unsafe_allow_html=True)
                        material_name = st.text_input("", placeholder="Enter material name", label_visibility="collapsed", key="admin_add_material_name")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Material Application</div>', unsafe_allow_html=True)
                        material_application = st.text_input("", placeholder="Enter material application", label_visibility="collapsed", key="admin_add_material_application")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Climate Zone</div>', unsafe_allow_html=True)
                        climate_zone = st.text_input("", placeholder="Enter climate zone", label_visibility="collapsed", key="admin_add_climate_zone")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Supplier Name</div>', unsafe_allow_html=True)
                        supplier_name = st.text_input("", placeholder="Enter supplier name", label_visibility="collapsed", key="admin_add_supplier_name")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Availability Status</div>', unsafe_allow_html=True)
                        availability_status = st.selectbox("", ["In Stock", "Out of Stock"], label_visibility="collapsed", key="admin_add_availability_status")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Lead Time (Days)</div>', unsafe_allow_html=True)
                        lead_time_days = st.number_input("", min_value=0, max_value=60, value=10, label_visibility="collapsed", key="admin_add_lead_time_days")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Project Type</div>', unsafe_allow_html=True)
                        project_type = st.text_input("", placeholder="Enter project type", label_visibility="collapsed", key="admin_add_project_type")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Durability Requirement</div>', unsafe_allow_html=True)
                        durability_requirement = st.selectbox("", ['5-10 years', '10-25 years', '25-50 years', '50+ years'], label_visibility="collapsed", key="admin_add_durability_requirement")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Project Specifications Match Score</div>', unsafe_allow_html=True)
                        project_specifications_match_score = st.number_input("", min_value=0.0, max_value=100.0, value=50.0, label_visibility="collapsed", key="admin_add_project_specifications_match_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Budget Constraint</div>', unsafe_allow_html=True)
                        budget_constraint = st.selectbox("", ['Economy', 'Standard', 'Premium', 'Luxury'], label_visibility="collapsed", key="admin_add_budget_constraint")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Sustainability Focus</div>', unsafe_allow_html=True)
                        sustainability_focus = st.selectbox("", ['Standard', 'Eco-friendly'], label_visibility="collapsed", key="admin_add_sustainability_focus")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Category</div>', unsafe_allow_html=True)
                        category_options = ["Structural", "Finishing", "Insulation", "Decorative"]
                        category = st.selectbox("", category_options, label_visibility="collapsed", key="admin_add_category")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col6:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Price ($)</div>', unsafe_allow_html=True)
                        price = st.number_input("", min_value=0.0, value=10.0, step=0.01, label_visibility="collapsed", key="admin_add_price")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Price Unit</div>', unsafe_allow_html=True)
                        price_unit = st.selectbox("", ["$/m²", "$/kg", "$/unit"], label_visibility="collapsed", key="admin_add_price_unit")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Image URL</div>', unsafe_allow_html=True)
                        image_url = st.text_input("", placeholder="https://example.com/image.jpg", label_visibility="collapsed", key="admin_add_image_url")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Description</div>', unsafe_allow_html=True)
                    description = st.text_area("", placeholder="Describe the material and its application", label_visibility="collapsed", key="admin_add_description")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col7, col8 = st.columns(2)
                    with col7:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Installation Score (0-100)</div>', unsafe_allow_html=True)
                        installation_score = st.number_input("", min_value=0, max_value=100, value=50, label_visibility="collapsed", key="admin_add_installation_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col8:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Maintenance Score (0-100)</div>', unsafe_allow_html=True)
                        maintenance_score = st.number_input("", min_value=0, max_value=100, value=50, label_visibility="collapsed", key="admin_add_maintenance_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    submit_button = st.form_submit_button("Add Material")
                    
                    if submit_button:
                        if not all([material_name, material_application, climate_zone, supplier_name, project_type]):
                            st.warning("Please fill in all required fields (Material Name, Material Application, Climate Zone, Supplier Name, Project Type).")
                        else:
                            try:
                                new_material = {
                                    'material_id': material_id,
                                    'material_name': material_name,
                                    'material_application': material_application,
                                    'climate_zone': climate_zone,
                                    'durability_requirement': durability_requirement,
                                    'budget_constraint': budget_constraint,
                                    'sustainability_focus': sustainability_focus,
                                    'supplier_name': supplier_name,
                                    'availability_status': availability_status,
                                    'lead_time_days': lead_time_days,
                                    'project_type': project_type,
                                    'project_specifications_match_score': project_specifications_match_score,
                                    'description': description,
                                    'category': category,
                                    'price': price,
                                    'price_unit': price_unit,
                                    'installation_score': installation_score,
                                    'maintenance_score': maintenance_score,
                                    'image_url': image_url
                                }
                                
                                new_df = pd.DataFrame([new_material])
                                if not NEW_MATERIALS_FILE.exists():
                                    new_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                else:
                                    existing_df = pd.read_csv(NEW_MATERIALS_FILE)
                                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                                    updated_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                
                                st.success("Material added successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding material: {e}")
            
            elif action == "Edit Material":
                if df.empty or 'material_id' not in df.columns:
                    st.warning("No materials available to edit.")
                else:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Select Material ID</div>', unsafe_allow_html=True)
                    material_id = st.selectbox("", df['material_id'].unique(), label_visibility="collapsed", key="admin_edit_material_id")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    material_data = df[df['material_id'] == material_id].iloc[0]
                    
                    with st.form("admin_edit_material_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Material Name</div>', unsafe_allow_html=True)
                            material_name = st.text_input("", value=material_data['material_name'], label_visibility="collapsed", key="admin_edit_material_name")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Material Application</div>', unsafe_allow_html=True)
                            material_application = st.text_input("", value=material_data['material_application'], label_visibility="collapsed", key="admin_edit_material_application")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Climate Zone</div>', unsafe_allow_html=True)
                            climate_zone = st.text_input("", value=material_data['climate_zone'], label_visibility="collapsed", key="admin_edit_climate_zone")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Supplier Name</div>', unsafe_allow_html=True)
                            supplier_name = st.text_input("", value=material_data['supplier_name'], label_visibility="collapsed", key="admin_edit_supplier_name")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Availability Status</div>', unsafe_allow_html=True)
                            availability_status = st.selectbox("", ["In Stock", "Out of Stock"], index=0 if material_data['availability_status'] == "In Stock" else 1, label_visibility="collapsed", key="admin_edit_availability_status")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Lead Time (Days)</div>', unsafe_allow_html=True)
                            lead_time_days = st.number_input("", min_value=0, max_value=60, value=int(material_data['lead_time_days']), label_visibility="collapsed", key="admin_edit_lead_time_days")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Project Type</div>', unsafe_allow_html=True)
                            project_type = st.text_input("", value=material_data['project_type'], label_visibility="collapsed", key="admin_edit_project_type")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Durability Requirement</div>', unsafe_allow_html=True)
                            durability_options = ['5-10 years', '10-25 years', '25-50 years', '50+ years']
                            durability_value = material_data['durability_requirement'] if material_data['durability_requirement'] in durability_options else '10-25 years'
                            durability_requirement = st.selectbox("", durability_options, index=durability_options.index(durability_value), label_visibility="collapsed", key="admin_edit_durability_requirement")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Project Specifications Match Score</div>', unsafe_allow_html=True)
                            project_specifications_match_score = st.number_input("", min_value=0.0, max_value=100.0, value=float(material_data['project_specifications_match_score']), label_visibility="collapsed", key="admin_edit_project_specifications_match_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Budget Constraint</div>', unsafe_allow_html=True)
                            budget_options = ['Economy', 'Standard', 'Premium', 'Luxury']
                            budget_value = material_data['budget_constraint'] if material_data['budget_constraint'] in budget_options else 'Standard'
                            budget_constraint = st.selectbox("", budget_options, index=budget_options.index(budget_value), label_visibility="collapsed", key="admin_edit_budget_constraint")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col5, col6 = st.columns(2)
                        with col5:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Sustainability Focus</div>', unsafe_allow_html=True)
                            sustainability_options = ['Standard', 'Eco-friendly']
                            sustainability_value = material_data['sustainability_focus'] if material_data['sustainability_focus'] in sustainability_options else 'Standard'
                            sustainability_focus = st.selectbox("", sustainability_options, index=sustainability_options.index(sustainability_value), label_visibility="collapsed", key="admin_edit_sustainability_focus")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Category</div>', unsafe_allow_html=True)
                            category_options = ["Structural", "Finishing", "Insulation", "Decorative"]
                            category_value = material_data.get('category', 'Structural') if material_data.get('category', 'Structural') in category_options else 'Structural'
                            category = st.selectbox("", category_options, index=category_options.index(category_value), label_visibility="collapsed", key="admin_edit_category")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col6:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Price ($)</div>', unsafe_allow_html=True)
                            price = st.number_input("", min_value=0.0, value=float(material_data.get('price', 10.0)), step=0.01, label_visibility="collapsed", key="admin_edit_price")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Price Unit</div>', unsafe_allow_html=True)
                            price_unit_options = ["$/m²", "$/kg", "$/unit"]
                            price_unit_value = material_data.get('price_unit', '$/m²') if material_data.get('price_unit', '$/m²') in price_unit_options else '$/m²'
                            price_unit = st.selectbox("", price_unit_options, index=price_unit_options.index(price_unit_value), label_visibility="collapsed", key="admin_edit_price_unit")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Image URL</div>', unsafe_allow_html=True)
                            image_url = st.text_input("", value=material_data.get('image_url', ''), label_visibility="collapsed", key="admin_edit_image_url")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Description</div>', unsafe_allow_html=True)
                        description = st.text_area("", value=material_data.get('description', ''), label_visibility="collapsed", key="admin_edit_description")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        col7, col8 = st.columns(2)
                        with col7:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Installation Score (0-100)</div>', unsafe_allow_html=True)
                            installation_score = st.number_input("", min_value=0, max_value=100, value=int(material_data.get('installation_score', 50)), label_visibility="collapsed", key="admin_edit_installation_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col8:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Maintenance Score (0-100)</div>', unsafe_allow_html=True)
                            maintenance_score = st.number_input("", min_value=0, max_value=100, value=int(material_data.get('maintenance_score', 50)), label_visibility="collapsed", key="admin_edit_maintenance_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        submit_button = st.form_submit_button("Update Material")
                        
                        if submit_button:
                            if not all([material_name, material_application, climate_zone, supplier_name, project_type]):
                                st.warning("Please fill in all required fields (Material Name, Material Application, Climate Zone, Supplier Name, Project Type).")
                            else:
                                try:
                                    updated_material = {
                                        'material_id': material_id,
                                        'material_name': material_name,
                                        'material_application': material_application,
                                        'climate_zone': climate_zone,
                                        'durability_requirement': durability_requirement,
                                        'budget_constraint': budget_constraint,
                                        'sustainability_focus': sustainability_focus,
                                        'supplier_name': supplier_name,
                                        'availability_status': availability_status,
                                        'lead_time_days': lead_time_days,
                                        'project_type': project_type,
                                        'project_specifications_match_score': project_specifications_match_score,
                                        'description': description,
                                        'category': category,
                                        'price': price,
                                        'price_unit': price_unit,
                                        'installation_score': installation_score,
                                        'maintenance_score': maintenance_score,
                                        'image_url': image_url
                                    }
                                    
                                    new_df = pd.read_csv(NEW_MATERIALS_FILE) if NEW_MATERIALS_FILE.exists() else pd.DataFrame(columns=updated_material.keys())
                                    if material_id in new_df['material_id'].values:
                                        new_df.loc[new_df['material_id'] == material_id, updated_material.keys()] = list(updated_material.values())
                                    else:
                                        new_df = pd.concat([new_df, pd.DataFrame([updated_material])], ignore_index=True)
                                    new_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                    
                                    st.success("Material updated successfully.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating material: {e}")
            
            elif action == "Delete Material":
                if df.empty or 'material_id' not in df.columns:
                    st.warning("No materials available to delete.")
                else:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Select Material ID</div>', unsafe_allow_html=True)
                    material_id = st.selectbox("", df['material_id'].unique(), label_visibility="collapsed", key="admin_delete_material_id")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("Delete Material"):
                        try:
                            new_df = pd.read_csv(NEW_MATERIALS_FILE) if NEW_MATERIALS_FILE.exists() else pd.DataFrame(columns=['material_id'])
                            new_df = new_df[new_df['material_id'] != material_id]
                            new_df.to_csv(NEW_MATERIALS_FILE, index=False)
                            st.success("Material deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting material: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">User Management</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">View, edit, or delete user accounts</div>', unsafe_allow_html=True)
            
            users_df = load_users()
            if users_df.empty:
                st.warning("No users registered.")
            else:
                st.dataframe(users_df)
                
                st.markdown('<div class="form-field">', unsafe_allow_html=True)
                st.markdown('<div class="form-label">Select Username to Delete</div>', unsafe_allow_html=True)
                username_to_delete = st.selectbox("", users_df['username'].unique(), label_visibility="collapsed", key="admin_delete_user")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("Delete User"):
                    if username_to_delete == st.session_state.user:
                        st.error("You cannot delete your own account.")
                    else:
                        try:
                            updated_users = users_df[users_df['username'] != username_to_delete]
                            save_users(updated_users)
                            st.success("User deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting user: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Supplier dashboard function
def supplier_dashboard():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">Supplier Dashboard - Welcome, {st.session_state.user}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-description">Manage your material inventory and view supplier insights</div>', unsafe_allow_html=True)
    
    try:
        _, _, _, _, _, _, _, _, df = load_model_and_encoders()
        if df is None:
            st.error("Failed to load material data. Please check the required files.")
            return
        
        # Tabs for Material Management and Supplier Information.
        tabs = st.tabs(["Material Management", "Supplier Information"])
        
        with tabs[0]:
            supplier_df = df[df['supplier_name'] == st.session_state.user]
            if supplier_df.empty:
                st.warning("No materials associated with your supplier account.")
            
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Material Management</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">View, add, or edit your materials</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-field">', unsafe_allow_html=True)
            st.markdown('<div class="form-label">Action</div>', unsafe_allow_html=True)
            action = st.selectbox("", ["View Materials", "Add Material", "Edit Material"], label_visibility="collapsed", key="supplier_action")
            st.markdown('</div>', unsafe_allow_html=True)

            if action == "View Materials":
                available_columns = [col for col in [
                    'material_id', 'material_name', 'material_application', 'climate_zone', 'durability_requirement',
                    'budget_constraint', 'sustainability_focus', 'supplier_name', 'supplier_contact_number', 'availability_status',
                    'lead_time_days', 'category', 'price', 'price_unit', 'description', 'installation_score',
                    'maintenance_score', 'image_url'
                ] if col in df.columns]
                df['live_price'] = df['material_name'].apply(lambda x: get_live_price("TATASTEEL.NS") if 'steel' in x.lower() else 'N/A')
                available_columns.append('live_price')
                st.dataframe(df[available_columns])
            
            elif action == "Add Material":
                with st.form("supplier_add_material_form"):
                    material_id = f"M{str(uuid.uuid4())[:4].upper()}"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Material Name</div>', unsafe_allow_html=True)
                        material_name = st.text_input("", placeholder="Enter material name", label_visibility="collapsed", key="supplier_add_material_name")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Material Application</div>', unsafe_allow_html=True)
                        material_application = st.text_input("", placeholder="Enter material application", label_visibility="collapsed", key="supplier_add_material_application")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Climate Zone</div>', unsafe_allow_html=True)
                        climate_zone = st.text_input("", placeholder="Enter climate zone", label_visibility="collapsed", key="supplier_add_climate_zone")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Supplier Name</div>', unsafe_allow_html=True)
                        supplier_name = st.text_input("", value=st.session_state.user, disabled=True, label_visibility="collapsed", key="supplier_add_supplier_name")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Availability Status</div>', unsafe_allow_html=True)
                        availability_status = st.selectbox("", ["In Stock", "Out of Stock"], label_visibility="collapsed", key="supplier_add_availability_status")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Lead Time (Days)</div>', unsafe_allow_html=True)
                        lead_time_days = st.number_input("", min_value=0, max_value=60, value=10, label_visibility="collapsed", key="supplier_add_lead_time_days")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Project Type</div>', unsafe_allow_html=True)
                        project_type = st.text_input("", placeholder="Enter project type", label_visibility="collapsed", key="supplier_add_project_type")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Durability Requirement</div>', unsafe_allow_html=True)
                        durability_requirement = st.selectbox("", ['5-10 years', '10-25 years', '25-50 years', '50+ years'], label_visibility="collapsed", key="supplier_add_durability_requirement")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Project Specifications Match Score</div>', unsafe_allow_html=True)
                        project_specifications_match_score = st.number_input("", min_value=0.0, max_value=100.0, value=50.0, label_visibility="collapsed", key="supplier_add_project_specifications_match_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Budget Constraint</div>', unsafe_allow_html=True)
                        budget_constraint = st.selectbox("", ['Economy', 'Standard', 'Premium', 'Luxury'], label_visibility="collapsed", key="supplier_add_budget_constraint")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Sustainability Focus</div>', unsafe_allow_html=True)
                        sustainability_focus = st.selectbox("", ['Standard', 'Eco-friendly'], label_visibility="collapsed", key="supplier_add_sustainability_focus")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Category</div>', unsafe_allow_html=True)
                        category_options = ["Structural", "Finishing", "Insulation", "Decorative"]
                        category = st.selectbox("", category_options, label_visibility="collapsed", key="supplier_add_category")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col6:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Price ($)</div>', unsafe_allow_html=True)
                        price = st.number_input("", min_value=0.0, value=10.0, step=0.01, label_visibility="collapsed", key="supplier_add_price")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Price Unit</div>', unsafe_allow_html=True)
                        price_unit = st.selectbox("", ["$/m²", "$/kg", "$/unit"], label_visibility="collapsed", key="supplier_add_price_unit")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Image URL</div>', unsafe_allow_html=True)
                        image_url = st.text_input("", placeholder="https://example.com/image.jpg", label_visibility="collapsed", key="supplier_add_image_url")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Description</div>', unsafe_allow_html=True)
                    description = st.text_area("", placeholder="Describe the material and its application", label_visibility="collapsed", key="supplier_add_description")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col7, col8 = st.columns(2)
                    with col7:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Installation Score (0-100)</div>', unsafe_allow_html=True)
                        installation_score = st.number_input("", min_value=0, max_value=100, value=50, label_visibility="collapsed", key="supplier_add_installation_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col8:
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Maintenance Score (0-100)</div>', unsafe_allow_html=True)
                        maintenance_score = st.number_input("", min_value=0, max_value=100, value=50, label_visibility="collapsed", key="supplier_add_maintenance_score")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    submit_button = st.form_submit_button("Add Material")
                    
                    if submit_button:
                        if not all([material_name, material_application, climate_zone, supplier_name, project_type]):
                            st.warning("Please fill in all required fields (Material Name, Material Application, Climate Zone, Supplier Name, Project Type).")
                        else:
                            try:
                                new_material = {
                                    'material_id': material_id,
                                    'material_name': material_name,
                                    'material_application': material_application,
                                    'climate_zone': climate_zone,
                                    'durability_requirement': durability_requirement,
                                    'budget_constraint': budget_constraint,
                                    'sustainability_focus': sustainability_focus,
                                    'supplier_name': supplier_name,
                                    'availability_status': availability_status,
                                    'lead_time_days': lead_time_days,
                                    'project_type': project_type,
                                    'project_specifications_match_score': project_specifications_match_score,
                                    'description': description,
                                    'category': category,
                                    'price': price,
                                    'price_unit': price_unit,
                                    'installation_score': installation_score,
                                    'maintenance_score': maintenance_score,
                                    'image_url': image_url
                                }
                                
                                new_df = pd.DataFrame([new_material])
                                if not NEW_MATERIALS_FILE.exists():
                                    new_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                else:
                                    existing_df = pd.read_csv(NEW_MATERIALS_FILE)
                                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                                    updated_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                
                                st.success("Material added successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding material: {e}")
            elif action == "Edit Material":
                if df.empty or 'material_id' not in df.columns:
                    st.warning("No materials available to edit.")
                else:
                    st.markdown('<div class="form-field">', unsafe_allow_html=True)
                    st.markdown('<div class="form-label">Select Material ID</div>', unsafe_allow_html=True)
                    material_id = st.selectbox("", df['material_id'].unique(), label_visibility="collapsed", key="admin_edit_material_id")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    material_data = df[df['material_id'] == material_id].iloc[0]
                    
                    with st.form("admin_edit_material_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Material Name</div>', unsafe_allow_html=True)
                            material_name = st.text_input("", value=material_data['material_name'], label_visibility="collapsed", key="admin_edit_material_name")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Material Application</div>', unsafe_allow_html=True)
                            material_application = st.text_input("", value=material_data['material_application'], label_visibility="collapsed", key="admin_edit_material_application")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Climate Zone</div>', unsafe_allow_html=True)
                            climate_zone = st.text_input("", value=material_data['climate_zone'], label_visibility="collapsed", key="admin_edit_climate_zone")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Supplier Name</div>', unsafe_allow_html=True)
                            supplier_name = st.text_input("", value=material_data['supplier_name'], label_visibility="collapsed", key="admin_edit_supplier_name")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Availability Status</div>', unsafe_allow_html=True)
                            availability_status = st.selectbox("", ["In Stock", "Out of Stock"], index=0 if material_data['availability_status'] == "In Stock" else 1, label_visibility="collapsed", key="admin_edit_availability_status")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Lead Time (Days)</div>', unsafe_allow_html=True)
                            lead_time_days = st.number_input("", min_value=0, max_value=60, value=int(material_data['lead_time_days']), label_visibility="collapsed", key="admin_edit_lead_time_days")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Project Type</div>', unsafe_allow_html=True)
                            project_type = st.text_input("", value=material_data['project_type'], label_visibility="collapsed", key="admin_edit_project_type")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Durability Requirement</div>', unsafe_allow_html=True)
                            durability_options = ['5-10 years', '10-25 years', '25-50 years', '50+ years']
                            durability_value = material_data['durability_requirement'] if material_data['durability_requirement'] in durability_options else '10-25 years'
                            durability_requirement = st.selectbox("", durability_options, index=durability_options.index(durability_value), label_visibility="collapsed", key="admin_edit_durability_requirement")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Project Specifications Match Score</div>', unsafe_allow_html=True)
                            project_specifications_match_score = st.number_input("", min_value=0.0, max_value=100.0, value=float(material_data['project_specifications_match_score']), label_visibility="collapsed", key="admin_edit_project_specifications_match_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Budget Constraint</div>', unsafe_allow_html=True)
                            budget_options = ['Economy', 'Standard', 'Premium', 'Luxury']
                            budget_value = material_data['budget_constraint'] if material_data['budget_constraint'] in budget_options else 'Standard'
                            budget_constraint = st.selectbox("", budget_options, index=budget_options.index(budget_value), label_visibility="collapsed", key="admin_edit_budget_constraint")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col5, col6 = st.columns(2)
                        with col5:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Sustainability Focus</div>', unsafe_allow_html=True)
                            sustainability_options = ['Standard', 'Eco-friendly']
                            sustainability_value = material_data['sustainability_focus'] if material_data['sustainability_focus'] in sustainability_options else 'Standard'
                            sustainability_focus = st.selectbox("", sustainability_options, index=sustainability_options.index(sustainability_value), label_visibility="collapsed", key="admin_edit_sustainability_focus")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Category</div>', unsafe_allow_html=True)
                            category_options = ["Structural", "Finishing", "Insulation", "Decorative"]
                            category_value = material_data.get('category', 'Structural') if material_data.get('category', 'Structural') in category_options else 'Structural'
                            category = st.selectbox("", category_options, index=category_options.index(category_value), label_visibility="collapsed", key="admin_edit_category")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col6:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Price ($)</div>', unsafe_allow_html=True)
                            price = st.number_input("", min_value=0.0, value=float(material_data.get('price', 10.0)), step=0.01, label_visibility="collapsed", key="admin_edit_price")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Price Unit</div>', unsafe_allow_html=True)
                            price_unit_options = ["$/m²", "$/kg", "$/unit"]
                            price_unit_value = material_data.get('price_unit', '$/m²') if material_data.get('price_unit', '$/m²') in price_unit_options else '$/m²'
                            price_unit = st.selectbox("", price_unit_options, index=price_unit_options.index(price_unit_value), label_visibility="collapsed", key="admin_edit_price_unit")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Image URL</div>', unsafe_allow_html=True)
                            image_url = st.text_input("", value=material_data.get('image_url', ''), label_visibility="collapsed", key="admin_edit_image_url")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="form-field">', unsafe_allow_html=True)
                        st.markdown('<div class="form-label">Description</div>', unsafe_allow_html=True)
                        description = st.text_area("", value=material_data.get('description', ''), label_visibility="collapsed", key="admin_edit_description")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        col7, col8 = st.columns(2)
                        with col7:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Installation Score (0-100)</div>', unsafe_allow_html=True)
                            installation_score = st.number_input("", min_value=0, max_value=100, value=int(material_data.get('installation_score', 50)), label_visibility="collapsed", key="admin_edit_installation_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col8:
                            st.markdown('<div class="form-field">', unsafe_allow_html=True)
                            st.markdown('<div class="form-label">Maintenance Score (0-100)</div>', unsafe_allow_html=True)
                            maintenance_score = st.number_input("", min_value=0, max_value=100, value=int(material_data.get('maintenance_score', 50)), label_visibility="collapsed", key="admin_edit_maintenance_score")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        submit_button = st.form_submit_button("Update Material")
                        
                        if submit_button:
                            if not all([material_name, material_application, climate_zone, supplier_name, project_type]):
                                st.warning("Please fill in all required fields (Material Name, Material Application, Climate Zone, Supplier Name, Project Type).")
                            else:
                                try:
                                    updated_material = {
                                        'material_id': material_id,
                                        'material_name': material_name,
                                        'material_application': material_application,
                                        'climate_zone': climate_zone,
                                        'durability_requirement': durability_requirement,
                                        'budget_constraint': budget_constraint,
                                        'sustainability_focus': sustainability_focus,
                                        'supplier_name': supplier_name,
                                        'availability_status': availability_status,
                                        'lead_time_days': lead_time_days,
                                        'project_type': project_type,
                                        'project_specifications_match_score': project_specifications_match_score,
                                        'description': description,
                                        'category': category,
                                        'price': price,
                                        'price_unit': price_unit,
                                        'installation_score': installation_score,
                                        'maintenance_score': maintenance_score,
                                        'image_url': image_url
                                    }
                                    
                                    new_df = pd.read_csv(NEW_MATERIALS_FILE) if NEW_MATERIALS_FILE.exists() else pd.DataFrame(columns=updated_material.keys())
                                    if material_id in new_df['material_id'].values:
                                        new_df.loc[new_df['material_id'] == material_id, updated_material.keys()] = list(updated_material.values())
                                    else:
                                        new_df = pd.concat([new_df, pd.DataFrame([updated_material])], ignore_index=True)
                                    new_df.to_csv(NEW_MATERIALS_FILE, index=False)
                                    
                                    st.success("Material updated successfully.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating material: {e}")                     
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Supplier Information</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">View information about suppliers, including available materials, lead times, and availability status.</div>', unsafe_allow_html=True)
            
            # Supplier Dashboard
            supplier_df = df[['supplier_name', 'supplier_contact_number', 'material_name', 'availability_status', 'lead_time_days']].drop_duplicates()
            
            # Filter options
            st.markdown('<div class="card-title">Supplier Database</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                material_filter = st.multiselect(
                    "Filter by Material:",
                    options=sorted(df['material_name'].unique()),
                    default=[],
                    key="supplier_material_filter"
                )
            
            with col2:
                availability_filter = st.multiselect(
                    "Filter by Availability:",
                    options=sorted(df['availability_status'].unique()),
                    default=[],
                    key="supplier_availability_filter"
                )
            
            with col3:
                lead_time = st.slider(
                    "Maximum Lead Time (days):",
                    min_value=0,
                    max_value=60,
                    value=60,
                    step=5,
                    key="supplier_lead_time_filter"
                )
            
            # Apply_filters
            filtered_suppliers = supplier_df.copy()
            
            if material_filter:
                filtered_suppliers = filtered_suppliers[filtered_suppliers['material_name'].isin(material_filter)]
            
            if availability_filter:
                filtered_suppliers = filtered_suppliers[filtered_suppliers['availability_status'].isin(availability_filter)]
            
            filtered_suppliers = filtered_suppliers[filtered_suppliers['lead_time_days'] <= lead_time]
            
            # Display filtered_suppliers
            if len(filtered_suppliers) > 0:
                st.dataframe(filtered_suppliers, use_container_width=True)
                
                # Visualization of supplier_data
                st.markdown('<div class="card-title">Supplier Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Count of materials by supplier
                    supplier_counts = filtered_suppliers.groupby('supplier_name')['material_name'].count().reset_index()
                    supplier_counts.columns = ['Supplier', 'Material Count']
                    
                    fig = px.bar(
                        supplier_counts.sort_values('Material Count', ascending=False).head(10),
                        x='Supplier',
                        y='Material Count',
                        title='Top 10 Suppliers by Available Materials',
                        color='Material Count'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Availability status by supplier
                    availability_by_supplier = filtered_suppliers.groupby(['supplier_name', 'availability_status']).size().reset_index(name='count')
                    
                    fig = px.sunburst(
                        availability_by_supplier,
                        path=['supplier_name', 'availability_status'],
                        values='count',
                        title='Material Availability by Supplier'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Lead time analysis
                lead_time_stats = filtered_suppliers.groupby('supplier_name')['lead_time_days'].agg(['mean', 'min', 'max']).reset_index()
                lead_time_stats.columns = ['Supplier', 'Average Lead Time', 'Minimum Lead Time', 'Maximum Lead Time']
                
                st.markdown('<div class="card-title">Lead Time Analysis</div>', unsafe_allow_html=True)
                st.dataframe(lead_time_stats.sort_values('Average Lead Time'), use_container_width=True)
                
                fig = px.box(
                    filtered_suppliers,
                    x='supplier_name',
                    y='lead_time_days',
                    title='Lead Time Distribution by Supplier',
                    labels={
                        'supplier_name': 'Supplier',
                        'lead_time_days': 'Lead Time (days)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Supplier contact information
                st.markdown('<div class="card-title">Supplier Contact Information</div>', unsafe_allow_html=True)
                st.info("Note: This is a demonstration. In a real-world application, this section would contain actual supplier contact details.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_supplier = st.selectbox(
                        "Select Supplier for Contact Details",
                        options=sorted(filtered_suppliers['supplier_name'].unique()),
                        key="supplier_contact_select"
                    )
                
                with col2:
                    if selected_supplier:
                        st.write(f"**Supplier:** {selected_supplier}")
                        st.write("**Contact:** Contact information would be displayed here")
                        st.write("**Website:** Supplier website would be linked here")
                        st.write("**Address:** Supplier address would be displayed here")
                        
                        material_count = filtered_suppliers[filtered_suppliers['supplier_name'] == selected_supplier]['material_name'].count()
                        in_stock_count = filtered_suppliers[(filtered_suppliers['supplier_name'] == selected_supplier) & 
                                                          (filtered_suppliers['availability_status'] == 'In Stock')]['material_name'].count()
                        
                        st.metric(
                            "Materials Available",
                            f"{material_count}",
                            f"{in_stock_count} in stock"
                        )
            else:
                st.info("No suppliers match the selected filters.")
            
            # Additional supplier insights
            st.markdown('<div class="card-title">Supplier Insights</div>', unsafe_allow_html=True)
            
            # Create tabs for different insights
            insight_tab1, insight_tab2 = st.tabs(["Supplier Performance", "Material Availability Trends"])
            
            with insight_tab1:
                # Supplier performance metrics
                supplier_performance = df.groupby('supplier_name').agg({
                    'material_name': 'count',
                    'lead_time_days': 'mean',
                    'availability_status': lambda x: (x == 'In Stock').mean() * 100
                }).reset_index()
                
                supplier_performance.columns = ['Supplier', 'Material Count', 'Average Lead Time', 'In Stock Percentage']
                supplier_performance['Performance Score'] = (
                    supplier_performance['In Stock Percentage'] / 100 * 0.6 + 
                    (1 - supplier_performance['Average Lead Time'] / 60) * 0.4
                ) * 100
                
                # Create a scatter plot
                fig = px.scatter(
                    supplier_performance,
                    x='Average Lead Time',
                    y='In Stock Percentage',
                    size='Material Count',
                    color='Performance Score',
                    hover_name='Supplier',
                    title='Supplier Performance Matrix',
                    labels={
                        'Average Lead Time': 'Average Lead Time (days)',
                        'In Stock Percentage': 'Materials In Stock (%)'
                    },
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top performers
                st.markdown('<div class="card-title">Top Performing Suppliers</div>', unsafe_allow_html=True)
                st.dataframe(
                    supplier_performance.sort_values('Performance Score', ascending=False).head(5),
                    use_container_width=True
                )
            
            with insight_tab2:
                # Material availability trends
                st.info("Note: In a real application, this would show historical availability trends over time.")
                
                # Aggregate by material type and availability
                material_availability = df.groupby(['material_name', 'availability_status']).size().reset_index(name='count')
                
                # Create horizontal bar chart
                fig = px.bar(
                    material_availability.pivot_table(
                        index='material_name',
                        columns='availability_status',
                        values='count',
                        fill_value=0
                    ).reset_index(),
                    x=['In Stock', 'Limited', 'Out of Stock'],
                    y='material_name',
                    title='Material Availability Status',
                    orientation='h',
                    labels={'material_name': 'Material', 'value': 'Count'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simulate a time trend with a line chart
                st.markdown('<div class="card-title">Simulated Availability Trend</div>', unsafe_allow_html=True)
                st.write("This is a demonstration of how availability trends would be displayed in a real application.")
                
                # Simulate monthly data for the past year
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                materials = ['Steel', 'Concrete', 'Wood', 'Glass', 'Aluminum']
                
                trend_data = []
                
                for material in materials:
                    # Generate a realistic trend with some randomness
                    base_value = np.random.uniform(60, 90)
                    
                    for i, month in enumerate(months):
                        # Add seasonal pattern and some noise
                        seasonal = 10 * np.sin(i / 12 * 2 * np.pi)
                        noise = np.random.normal(0, 5)
                        value = max(0, min(100, base_value + seasonal + noise))
                        
                        trend_data.append({
                            'Month': month,
                            'Material': material,
                            'Availability (%)': value
                        })
                
                trend_df = pd.DataFrame(trend_data)
                
                fig = px.line(
                    trend_df,
                    x='Month',
                    y='Availability (%)',
                    color='Material',
                    title='Material Availability Trend Over Time (Simulated)',
                    labels={'Month': 'Month', 'Availability (%)': 'Availability (%)'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Main app logic
def main():
    init_session_state()

    if st.session_state.logged_in:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div style="font-size: 1.2rem; font-weight: 600; color: #333;">Welcome, {st.session_state.user} ({st.session_state.role})</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.role = None
                st.session_state.page = 'home'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.role == "Admin":
            admin_dashboard()
        elif st.session_state.role == "Supplier":
            supplier_dashboard()
        else:
            user_dashboard()
    else:
        if st.session_state.page == 'home':
            home_page()
        elif st.session_state.page == 'login':
            login_page()
        elif st.session_state.page == 'register':
            register_page()

    # ✅ Floating chatbot should appear on all views
    floating_chatbot()

if __name__ == "__main__":
    main()

    
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; border-top: 1px solid rgba(0, 0, 0, 0.1);">
    <p style="color: #666; font-size: 0.9rem;">
        © 2025 Construction Material Platform. All rights reserved. | 
        <a href="#privacy" style="color: #1E88E5; text-decoration: none;">Privacy Policy</a> | 
        <a href="#terms" style="color: #1E88E5; text-decoration: none;">Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True)