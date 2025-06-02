import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Initialize session state variables **AFTER importing streamlit**
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False  # Track chatbot open/close state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat messages

# Load trained models & encoders
with open("models.pkl", "rb") as model_file:
    multi_model, encoder, label_encoders = pickle.load(model_file)

# Load dataset for recommendations
df = pd.read_csv("gym_recommendation.csv")

# Load fitness chatbot dataset
qa_df = pd.read_csv("fitness_qa.csv")  # Ensure this file is in the same directory
qa_dict = dict(zip(qa_df["Question"].str.lower(), qa_df["Answer"]))

# Fix typo and normalize case in the dataset
df["Level"] = df["Level"].str.strip().str.lower().replace({"obuse": "obese"})
df["Fitness Type"] = df["Fitness Type"].str.strip().str.lower()

# Set page config
st.set_page_config(page_title="Gym Recommendation System", page_icon="🏋️", layout="wide")

# Hero section
st.markdown("""
    <div style='text-align: center;'>
        <h1>🏋️ Personalized Gym Recommendation System</h1>
        <p>Find the best fitness plan based on your body type, health conditions, and goals.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Enter Your Details")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
height = st.sidebar.number_input("Height (m)", min_value=1.2, max_value=2.5, step=0.01, format="%.2f")
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.5, format="%.2f")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1, format="%d")
hypertension = st.sidebar.radio("Hypertension?", ["No", "Yes"])
diabetes = st.sidebar.radio("Diabetes?", ["No", "Yes"])

if st.sidebar.button("Generate Recommendations"):
    # Calculate BMI
    bmi = round(weight / (height ** 2), 2)
    
    # Determine BMI Category
    if bmi < 18.5:
        bmi_category = "Underweight"
        fitness_goal = "Weight Gain"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal"
        fitness_goal = "Weight Gain"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
        fitness_goal = "Weight Loss"
    else:
        bmi_category = "Obese"
        fitness_goal = "Weight Loss"

    st.success(f"Your BMI: {bmi} | Category: {bmi_category} | Goal: {fitness_goal}")
    
    # Prepare user input for encoding
    user_input_df = pd.DataFrame([[sex, hypertension, diabetes, fitness_goal]],
                                 columns=["Sex", "Hypertension", "Diabetes", "Fitness Goal"])
    encoded_user_input = encoder.transform(user_input_df)
    user_features = np.concatenate([[height, weight], encoded_user_input.flatten()])

    # Predict fitness type
    predictions = multi_model.predict([user_features])[0]
    fitness_type = label_encoders["Fitness Type"].inverse_transform([predictions[1]])[0]

    st.info(f"**Recommended Fitness Type:** {fitness_type}")

    # Normalize user inputs for matching
    bmi_category = bmi_category.lower()
    fitness_type = fitness_type.lower()

    # Fetch recommendations
    recommendations = df[(df["Level"] == bmi_category) & (df["Fitness Type"] == fitness_type)]
    
    if not recommendations.empty:
        st.markdown("""
            <div style='padding:15px; border-radius:8px; color:white; text-align:center;'>
                <h2 style='color:#f4a261;'>💪 Personalized Recommendations</h2>
                <p style='font-style:italic; text-align:center;'>"Alright let's get started with your fitness journey"</p>
            </div>
        """, unsafe_allow_html=True)

        panel_colors = {
            "Exercises": "#3D3C3A",
            "Equipment": "#3D3C3A",
            "Diet": "#36454F",
            "Recommendation": "#34282C"
        }
        icons = {
            "Exercises": "🏋️",
            "Equipment": "🛠️",
            "Diet": "🍎",
            "Recommendation": "📌"
        }

        for col in ["Exercises", "Equipment", "Diet", "Recommendation"]:
            st.markdown(f"""
                <div style='background-color:{panel_colors[col]}; padding:12px; margin-bottom:10px; border-radius:8px; color:white;'>
                    <b>{icons[col]} {col}:</b> {recommendations[col].values[0]}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this category. Try adjusting your inputs.")

    # Diet Macro Breakdown
    st.subheader("🍽️ Macro Breakdown")
    diet_recommendation = recommendations["Diet"].values[0] if not recommendations.empty else "Balanced Diet"
    
    # Assign approximate percentage values based on diet content
    vegetables = 30 if "vegetables" in diet_recommendation.lower() else 20
    protein_intake = 40 if "protein" in diet_recommendation.lower() else 30
    juice = 30 if "juice" in diet_recommendation.lower() else 20
    
    graph_width, graph_height = 3, 3  # Adjust graph size here
    fig, ax = plt.subplots(figsize=(graph_width, graph_height))
    ax.pie([vegetables, protein_intake, juice], labels=["Vegetables", "Protein Intake", "Juice"], autopct='%1.1f%%', startangle=90, colors=['#ffcc99','#ff6666','#66b3ff'])
    ax.set_title("Diet Breakdown")
    st.pyplot(fig)
    
    # Scatter Plot for BMI vs. Age
    st.subheader("📈 BMI vs. Age Distribution")
    fig, ax = plt.subplots(figsize=(graph_width + 2, graph_height + 1))  # Adjust size here
    sns.scatterplot(x=df["Age"], y=df["BMI"], hue=df["Level"], palette="coolwarm", alpha=0.7, ax=ax)
    ax.scatter(age, bmi, color='red', label='Your Data', s=100)
    ax.set_xlabel("Age")
    ax.set_ylabel("BMI")
    ax.set_title("BMI vs Age Scatter Plot")
    ax.legend()
    st.pyplot(fig)
# Chatbot UI
import google.generativeai as genai  # Import Gemini API

# Google Gemini API Key (Replace with your actual API key)
GEMINI_API_KEY = "" # You can put your api key here 
genai.configure(api_key=GEMINI_API_KEY)

# Sidebar Chatbot Container
with st.sidebar:
    st.markdown("---")
    st.markdown("## 💬 Chatbot")

    # Open Chatbot Button
    if st.button("Open Chatbot 🤖"):
        st.session_state.chat_open = True  # Store chatbot state

    # Display chatbot only if open
    if st.session_state.get("chat_open", False):
        st.markdown("### 🤖 Fitness Chatbot")
        
        # Display chat history
        for msg in st.session_state.get("chat_history", []):
            st.write(f"**{msg['role']}**: {msg['text']}")

        # Chat Input & Send Button (Same Line)
        chat_col1, chat_col2 = st.columns([5, 1])  # Larger input, smaller button

        with chat_col1:
            user_input = st.text_input("Ask a question:", key="chat_input", label_visibility="collapsed")

        with chat_col2:
            if st.button("📤"):  # Symbol button for Send
                if user_input.strip():
                    # Check dataset for a predefined answer
                    response = qa_dict.get(user_input.lower())
                    
                    # If not found, generate an AI response using Gemini API
                    if response is None:
                        try:
                            model = genai.GenerativeModel("gemini-1.5-pro")  # Using Gemini Pro Model
                            ai_response = model.generate_content(user_input)
                            response = ai_response.text.strip()
                        except Exception as e:
                            response = "Sorry, I am unable to generate an answer right now."
                    
                    # Store chat history
                    st.session_state.chat_history.append({"role": "You", "text": user_input})
                    st.session_state.chat_history.append({"role": "Chatbot", "text": response})
                
                st.rerun()  # Update only chat section, keeping predictions intact

        # Close chatbot button
        if st.button("Close Chatbot 🤖"):
            st.session_state.chat_open = False
            st.rerun()  # Rerun only chatbot section without clearing predictions
