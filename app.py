import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import speech_recognition as sr  # For voice commands

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("recipe dataset.csv")

        if "images" in df.columns:
            df.rename(columns={"images": "image"}, inplace=True)

        if "ingredients" not in df.columns:
            df["ingredients"] = ""

        df["image"] = df["image"].apply(lambda x: x if isinstance(x, str) and x.startswith("http") else None)
        df["ingredients"].fillna("", inplace=True)

        return df
    except FileNotFoundError:
        st.error("❌ Recipe dataset not found! Ensure 'recipe dataset.csv' is available.")
        return pd.DataFrame()

# Find recipes based on ingredients
def find_recipes(available_ingredients, recipe_type, df, top_n=10):
    if df.empty:
        return pd.DataFrame()

    df_filtered = df.copy()

    if recipe_type and recipe_type != "All":
        df_filtered = df_filtered[df_filtered["recipe_type"] == recipe_type]

    available_ingredients = " ".join([ingredient.lower().strip() for ingredient in available_ingredients])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_filtered["ingredients"])
    available_ingredients_vector = vectorizer.transform([available_ingredients])

    cosine_similarities = cosine_similarity(available_ingredients_vector, tfidf_matrix).flatten()
    df_filtered["similarity"] = cosine_similarities

    top_recipes = df_filtered.sort_values(by="similarity", ascending=False).head(top_n).copy()
    return top_recipes[["title", "prep", "servings", "total", "recipe_type", "url", "image", "ingredients", "directions"]]

# ✅ TTS Function (Now Uses Temp Files)
def text_to_speech(recipe_text):
    if not recipe_text.strip():
        st.error("❌ No text found for speech synthesis!")
        return None

    try:
        if len(recipe_text) > 500:
            recipe_text = recipe_text[:500] + "..."

        tts = gTTS(text=recipe_text, lang="en", tld="com")

        # Save audio in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"⚠️ TTS Error: {str(e)}")
        return None

# 🔥 Improved Voice Command Function
def listen_to_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise to improve accuracy
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.write("🎙️ Listening... Speak now! (e.g., 'tomato, onion, bell pepper')")
        audio = recognizer.listen(source, timeout=5)  # Set a timeout for listening

    try:
        text = recognizer.recognize_google(audio)
        st.write(f"🎤 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("⚠️ Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"⚠️ Could not request results from Google Speech Recognition service; {e}")
        return None

# Weekly Planner Layout
def weekly_planner(recipe_dataset):
    st.subheader("📅 Weekly Meal Planner")
    
    # Create a dictionary to store meals for each day
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_plan = {day: None for day in days}

    # Display a dropdown for each day to select a recipe
    for day in days:
        meal_plan[day] = st.selectbox(
            f"Select a recipe for {day}:",
            ["None"] + list(recipe_dataset["title"].unique()) if not recipe_dataset.empty else ["None"]
        )

    return meal_plan

# Generate Shopping List
def generate_shopping_list(meal_plan, recipe_dataset):
    st.subheader("🛒 Shopping List")
    
    # Get all selected recipes
    selected_recipes = [meal_plan[day] for day in meal_plan if meal_plan[day] != "None"]
    
    if not selected_recipes:
        st.warning("⚠️ No recipes selected for the week.")
        return
    
    # Aggregate ingredients from selected recipes
    shopping_list = set()
    for recipe_title in selected_recipes:
        recipe = recipe_dataset[recipe_dataset["title"] == recipe_title]
        if not recipe.empty:
            ingredients = recipe["ingredients"].iloc[0].split(",")
            shopping_list.update([ingredient.strip() for ingredient in ingredients])
    
    # Display the shopping list
    if shopping_list:
        st.write("Here's your shopping list for the week:")
        for item in shopping_list:
            st.write(f"- {item}")
    else:
        st.warning("⚠️ No ingredients found for the selected recipes.")

# BMI Calculator Layout
def bmi_calculator():
    st.subheader("🧮 BMI Calculator")
    weight = st.number_input("Enter your weight (kg):", min_value=0.0, max_value=300.0, value=70.0)
    height = st.number_input("Enter your height (cm):", min_value=0.0, max_value=300.0, value=170.0)

    if st.button("Calculate BMI"):
        if weight > 0 and height > 0:
            height_m = height / 100  # Convert height from cm to meters
            bmi = weight / (height_m ** 2)
            st.success(f"Your BMI is: **{bmi:.2f}**")

            # BMI Categories
            if bmi < 18.5:
                st.warning("You are underweight.")
            elif 18.5 <= bmi < 24.9:
                st.success("You have a normal weight.")
            elif 25 <= bmi < 29.9:
                st.warning("You are overweight.")
            else:
                st.error("You are obese.")
        else:
            st.error("Please enter valid weight and height.")

# Streamlit App
def main():
    st.set_page_config(page_title="🍽️ Recipe Recommendation App", layout="wide")

    st.title("🍽️ Recipe Recommendation App")
    st.write("Filter recipes by type and get recommendations based on your available ingredients! 🥦🍅🍗")

    # Load the dataset
    recipe_dataset = load_data()
    
    st.sidebar.header("🔍 Filters")
    recipe_type = st.sidebar.selectbox(
        "📌 Select Recipe Type",
        ["All"] + list(recipe_dataset["recipe_type"].dropna().unique()) if not recipe_dataset.empty else ["All"]
    )

    st.sidebar.header("🥕 Available Ingredients")
    available_ingredients = st.sidebar.text_input("Enter your available ingredients (comma-separated):")
    available_ingredients = [ingredient.strip().lower() for ingredient in available_ingredients.split(",") if ingredient.strip()]

    # Voice Command Button
    if st.sidebar.button("🎤 Use Voice Commands"):
        voice_input = listen_to_voice()
        if voice_input:
            available_ingredients = [ingredient.strip().lower() for ingredient in voice_input.split(",")]
            st.sidebar.write("**🎤 Voice Input Ingredients:**", available_ingredients)

    # Toggle between BMI Calculator and Weekly Meal Planner
    st.sidebar.header("🔀 Toggle Features")
    feature = st.sidebar.radio("Choose a feature:", ["BMI Calculator", "Weekly Meal Planner"])

    if feature == "BMI Calculator":
        bmi_calculator()
    elif feature == "Weekly Meal Planner":
        st.header("📅 Weekly Meal Planner")
        meal_plan = weekly_planner(recipe_dataset)

        # Generate Shopping List Button
        if st.button("🛒 Generate Shopping List"):
            generate_shopping_list(meal_plan, recipe_dataset)

    # Recipe Recommendations Section
    if available_ingredients:
        with st.spinner("🔎 Finding the best recipes for you..."):
            recommended_recipes = find_recipes(available_ingredients, recipe_type if recipe_type != "All" else None, recipe_dataset)

        st.success("🎉 Here are your recommended recipes! 🍽️")

        if not recommended_recipes.empty:
            displayed_images = set()
            for _, recipe in recommended_recipes.iterrows():
                col1, col2 = st.columns([3, 1])
                img_url = recipe["image"] if pd.notna(recipe["image"]) else "https://via.placeholder.com/200?text=No+Image"

                if img_url not in displayed_images:
                    with col2:
                        st.image(
                            img_url,
                            width=500,
                            caption=recipe["title"],
                            use_container_width=True
                        )
                        displayed_images.add(img_url)

                    with col1:
                        st.subheader(recipe["title"])
                        st.write(f"🕒 **Prep Time:** {recipe['prep']}")
                        st.write(f"🍽️ **Servings:** {recipe['servings']}")
                        st.write(f"⏳ **Total Time:** {recipe['total']}")
                        st.write(f"📖 **Recipe Type:** {recipe['recipe_type']}")
                        st.markdown(
                            f"""
                            <a href="{recipe['url']}" target="_blank" 
                            style="display: block; width: 200px; text-align: center; background-color: #007bff; color: white; padding: 10px; border-radius: 4px; text-decoration: none; font-weight: bold; transition: background-color 0.3s ease;">
                                View Recipe
                            </a>
                            <style>
                                a:hover {{
                                    background-color: rgba(0, 87, 179, 0.8);
                                }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                    if st.button(f"🔊 Listen to {recipe['title']} ingredients"):
                        audio_file = text_to_speech(recipe["ingredients"])
                        if audio_file:
                            st.audio(audio_file, format="audio/mp3")
                    
                    if st.button(f"🔊 Listen to {recipe['title']} instructions"):
                        audio_file = text_to_speech(recipe["directions"])
                        if audio_file:
                            st.audio(audio_file, format="audio/mp3")

                    st.write("---")
        else:
            st.warning("⚠️ No recipes found with the available ingredients.")

if __name__ == "__main__":
    main()