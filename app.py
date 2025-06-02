import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Medical Specialty Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1B9A95;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4A90A4;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1B9A95 0%, #4ECDC4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(27, 154, 149, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .confidence-card {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(78, 205, 196, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #E8F8F8 0%, #B8E6E1 100%);
        border-left: 4px solid #1B9A95;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        color: #2C5F5D;
        box-shadow: 0 4px 12px rgba(27, 154, 149, 0.15);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #1B9A95, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(27, 154, 149, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(27, 154, 149, 0.4);
        background: linear-gradient(45deg, #17877D, #44A08D);
    }
    .sidebar .stSelectbox > div > div {
        background-color: #E8F8F8;
        border-color: #1B9A95;
    }
    .stMetric {
        background: linear-gradient(135deg, #F0FFFE 0%, #E8F8F8 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1B9A95;
    }
    .stMetric > div {
        color: #2C5F5D;
    }
    .stTextArea > div > div > textarea {
        border-color: #4ECDC4;
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #1B9A95;
        box-shadow: 0 0 0 2px rgba(27, 154, 149, 0.2);
    }
    .healthcare-accent {
        color: #1B9A95;
        font-weight: bold;
    }
    .success-message {
        background: linear-gradient(135deg, #B8E6E1 0%, #7FDDDD 100%);
        color: #2C5F5D;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
    }
    .specialty-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 12px;
        margin: 1rem 0;
    }
    .specialty-item {
        background: linear-gradient(135deg, #F0FFFE 0%, #E8F8F8 100%);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #1B9A95;
        color: #2C5F5D;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(27, 154, 149, 0.1);
    }
    .specialty-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(27, 154, 149, 0.2);
        background: linear-gradient(135deg, #E8F8F8 0%, #B8E6E1 100%);
    }
    .modal-header {
        color: #1B9A95;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4ECDC4;
    }
    .stats-box {
        background: linear-gradient(135deg, #4ECDC4 0%, #1B9A95 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(27, 154, 149, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Load model and vectorizer
@st.cache_resource
def load_assets():
    try:
        model = load_model("medical_specialty_model.keras")
        vectorizer = joblib.load("vectorizer.pkl")
        index_to_specialty = joblib.load("index_to_specialty.pkl")
        return model, vectorizer, index_to_specialty
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        return None, None, None

# Get top predictions with confidence scores
def get_top_predictions(prediction, index_to_specialty, top_k=5):
    indices = np.argsort(prediction[0])[::-1][:top_k]
    predictions = []
    for idx in indices:
        predictions.append({
            'Specialty': index_to_specialty[idx],
            'Confidence': float(prediction[0][idx])
        })
    return predictions

# Function to display all medical specialties modal
def show_all_specialties_modal(index_to_specialty):
    if index_to_specialty is None:
        st.error("❌ Unable to load medical specialties data.")
        return
    
    # Get all specialties sorted alphabetically
    all_specialties = sorted(list(index_to_specialty.values()))
    total_count = len(all_specialties)
    
    # Modal header with statistics
    st.markdown(f"""
    <div class="modal-header">
        🏥 All Medical Specialties
    </div>
    <div class="stats-box">
        <h3 style="margin: 0;">Total Specialties Available: {total_count}</h3>
        <p style="margin: 0.5rem 0 0 0;">Our AI model can classify transcriptions into any of these medical specialties</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    search_term = st.text_input("🔍 Search specialties:", placeholder="Type to search...")
    
    if search_term:
        filtered_specialties = [spec for spec in all_specialties if search_term.lower() in spec.lower()]
        st.info(f"Found {len(filtered_specialties)} specialties matching '{search_term}'")
    else:
        filtered_specialties = all_specialties
    
    # Display specialties in a grid layout
    if filtered_specialties:
        # Create HTML grid
        grid_html = '<div class="specialty-grid">'
        for specialty in filtered_specialties:
            grid_html += f'<div class="specialty-item">🩺 {specialty}</div>'
        grid_html += '</div>'
        
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Display count
        if search_term:
            st.caption(f"Showing {len(filtered_specialties)} of {total_count} total specialties")
        else:
            st.caption(f"Showing all {total_count} medical specialties")
    else:
        st.warning("No specialties found matching your search term.")
    
    # Additional information
    with st.expander("ℹ️ About Medical Specialties Classification", expanded=False):
        st.markdown("""
        **How it works:**
        - Our AI model analyzes the content and context of medical transcriptions
        - It uses advanced natural language processing to identify key medical terms and patterns
        - The model then classifies the transcription into the most appropriate medical specialty
        
        **Specialty Categories Include:**
        - Primary care specialties (Family Medicine, Internal Medicine)
        - Surgical specialties (General Surgery, Orthopedics, Neurosurgery)
        - Medical specialties (Cardiology, Gastroenterology, Endocrinology)
        - Diagnostic specialties (Radiology, Pathology)
        - And many more specialized fields
        
        **Note:** The AI provides probability scores for its predictions, helping you understand the confidence level of each classification.
        """)

# Sample medical transcriptions for demo
SAMPLE_TRANSCRIPTS = {
    "Cardiovascular, Pulmonary": """Patient presents with chest pain radiating to left arm. ECG shows ST elevation in leads II, III, aVF. Troponin levels elevated at 2.5 ng/mL. Patient has history of hypertension and diabetes. Recommend cardiac catheterization and initiate dual antiplatelet therapy.""",
    
    "Surgery": """Patient presents with erythematous, scaly plaques on bilateral knees and elbows. Lesions are well-demarcated with silvery scale. Patient reports pruritus and a family history of psoriasis. Due to severity and resistance to medical management, surgical intervention is required. Recommend immediate referral to dermatologic surgery for excision and further evaluation.""",
    
    "Gastroenterology": """A 50-year-old female whose 51-year-old sister has a history of multiple colon polyps, which may slightly increase her risk for colon cancer in the future.""",
    
    "Orthopedics": """Patient sustained injury to right knee during sports activity. Physical examination reveals positive anterior drawer test and Lachman test. MRI shows complete tear of anterior cruciate ligament. Recommend arthroscopic ACL reconstruction."""
}

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Medical Specialty Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered classification of medical transcriptions</p>', unsafe_allow_html=True)
    
    # Load assets
    model, vectorizer, index_to_specialty = load_assets()
    
    if model is None:
        st.error("🚨 Failed to load model. Please ensure all model files are available.")
        return
    
    # Add "View All Specialties" button in the header area
    col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
    with col_header2:
        if st.button("📋 View All Medical Specialties", 
                    help="Click to see all medical specialties that this AI can classify",
                    type="secondary"):
            st.session_state.show_specialties_modal = True
    
    # Show modal if button was clicked
    if st.session_state.get('show_specialties_modal', False):
        with st.container():
            # Close button
            col_close1, col_close2 = st.columns([4, 1])
            with col_close2:
                if st.button("❌ Close", key="close_modal"):
                    st.session_state.show_specialties_modal = False
                    st.rerun()
            
            # Show the modal content
            show_all_specialties_modal(index_to_specialty)
            
            # Add some spacing
            st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="color: #1B9A95;">📋 Quick Start</h2>', unsafe_allow_html=True)
        st.markdown("""
        **How to use:**
        1. Enter or paste a medical transcription
        2. Click 'Analyze Transcription'
        3. View predicted specialty and confidence scores
        """)
        
        st.markdown('<h2 style="color: #1B9A95;">📊 Model Info</h2>', unsafe_allow_html=True)
        total_specialties = len(index_to_specialty) if index_to_specialty else 0
        st.metric("Total Specialties", total_specialties)
        
        # Add button to view all specialties in sidebar too
        if st.button("🔍 Browse All Specialties", key="sidebar_specialties"):
            st.session_state.show_specialties_modal = True
            st.rerun()
        
        st.markdown('<h2 style="color: #1B9A95;">🎯 Sample Transcripts</h2>', unsafe_allow_html=True)
        selected_sample = st.selectbox(
            "Try a sample:",
            ["Select a sample..."] + list(SAMPLE_TRANSCRIPTS.keys())
        )
        
        if selected_sample != "Select a sample...":
            if st.button("Load Sample", key="load_sample", help="Load the selected sample transcript"):
                st.session_state.sample_text = SAMPLE_TRANSCRIPTS[selected_sample]
        
        # Statistics section
        st.markdown('<h2 style="color: #1B9A95;">📈 Session Stats</h2>', unsafe_allow_html=True)
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        st.metric("Predictions Made", st.session_state.prediction_count)
    
    # Main content area (only show if modal is not open)
    if not st.session_state.get('show_specialties_modal', False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 style="color: #1B9A95;">📝 Clinical Transcription Input</h3>', unsafe_allow_html=True)
            
            # Text input with sample text if available
            default_text = st.session_state.get('sample_text', '')
            user_input = st.text_area(
                "Enter medical transcription:",
                value=default_text,
                height=250,
                placeholder="Paste your medical transcription here...",
                help="Enter a clinical note, consultation, or medical transcription for specialty classification"
            )
            
            # Character count
            char_count = len(user_input)
            st.caption(f"Characters: **{char_count}**")
            
            # Clear and Analyze buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🗑️ Clear Text"):
                    st.session_state.sample_text = ''
                    st.rerun()
            
            with col_btn2:
                analyze_clicked = st.button("🔍 Analyze Transcription", type="primary")
        
        with col2:
            st.markdown('<h3 style="color: #1B9A95;">ℹ️ About This Tool</h3>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong class="healthcare-accent">Purpose:</strong> This AI model classifies medical transcriptions into their appropriate medical specialties.
            
            <strong class="healthcare-accent">Accuracy:</strong> Trained on thousands of medical documents for high precision.
            
            <strong class="healthcare-accent">Use Cases:</strong><br>
            • Clinical documentation routing<br>
            • Medical record organization<br>  
            • Specialty referral assistance
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction section
        if analyze_clicked:
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a medical transcription to analyze.")
            else:
                with st.spinner("🔄 Analyzing transcription..."):
                    try:
                        # Process and predict
                        processed_text = preprocess_text(user_input)
                        vector = vectorizer.transform([processed_text]).toarray()
                        prediction = model.predict(vector, verbose=0)
                        
                        # Get top predictions
                        top_predictions = get_top_predictions(prediction, index_to_specialty)
                        
                        # Update session state
                        st.session_state.prediction_count += 1
                        
                        # Display results
                        st.markdown('<h3 style="color: #1B9A95;">🎯 Prediction Results</h3>', unsafe_allow_html=True)
                        
                        # Main prediction
                        main_pred = top_predictions[0]
                        col_pred1, col_pred2 = st.columns(2)
                        
                        with col_pred1:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h3>🩺 Primary Specialty</h3>
                                <h2>{main_pred['Specialty']}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_pred2:
                            confidence_pct = main_pred['Confidence'] * 100
                            st.markdown(f"""
                            <div class="confidence-card">
                                <h3>🔮 Confidence Score</h3>
                                <h2>{confidence_pct:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Top 5 predictions chart
                        st.markdown('<h3 style="color: #1B9A95;">📊 Top 5 Predictions</h3>', unsafe_allow_html=True)
                        df_predictions = pd.DataFrame(top_predictions)
                        df_predictions['Confidence_Pct'] = df_predictions['Confidence'] * 100
                        
                        # Healthcare-themed color scale
                        fig = px.bar(
                            df_predictions,
                            x='Confidence_Pct',
                            y='Specialty',
                            orientation='h',
                            title="Confidence Scores by Medical Specialty",
                            color='Confidence_Pct',
                            color_continuous_scale=[[0, '#B8E6E1'], [0.5, '#4ECDC4'], [1, '#1B9A95']]
                        )
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'},
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2C5F5D'),
                            title_font=dict(color='#1B9A95', size=16)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        with st.expander("📋 Detailed Results", expanded=False):
                            df_display = df_predictions.copy()
                            df_display['Confidence'] = df_display['Confidence_Pct'].apply(lambda x: f"{x:.2f}%")
                            df_display = df_display[['Specialty', 'Confidence']]
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                        
                        # Interpretation guide with healthcare colors
                        if main_pred['Confidence'] > 0.8:
                            st.markdown("""
                            <div class="success-message">
                                ✅ <strong>High confidence prediction</strong> - Very reliable result
                            </div>
                            """, unsafe_allow_html=True)
                        elif main_pred['Confidence'] > 0.6:
                            st.info("ℹ️ Moderate confidence prediction - Good result with some uncertainty")
                        else:
                            st.warning("⚠️ Low confidence prediction - Consider manual review")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()