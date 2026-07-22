import streamlit as st
import joblib
import google.generativeai as genai
import os
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Food Waste Chef", page_icon="👨‍🍳", layout="wide")

# --- LOAD ML MODEL ---
@st.cache_resource
def load_ml_components():
    # Load the files saved from your Jupyter Notebook
    model = joblib.load('cuisine_predictor_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_ml_components()

from dotenv import load_dotenv

api_key = st.secrets["GEMINI_API_KEY"]

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API Key not found in Streamlit Secrets!")

# --- SESSION STATE INITIALIZATION ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "ingredients_list" not in st.session_state:
    st.session_state.ingredients_list = []
if "top_cuisines" not in st.session_state:
    st.session_state.top_cuisines = []
if "selected_cuisines" not in st.session_state:
    st.session_state.selected_cuisines = []
if "recipes_data" not in st.session_state:
    st.session_state.recipes_data = []
if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = ""
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Callback to handle adding ingredients smoothly
def add_ingredient():
    val = st.session_state.new_ing_input.strip()
    if val and val.lower() not in [i.lower() for i in st.session_state.ingredients_list]:
        st.session_state.ingredients_list.append(val)
    st.session_state.new_ing_input = "" # Clear the input box

st.title("👨‍🍳 The AI Food Waste Chef")
st.write("Turn your random, expiring ingredients into a cohesive, gourmet meal!")
st.divider()

# ==========================================
# STEP 1: Ingredient Input (List & Buttons)
# ==========================================
st.header("1. What's in your fridge?")
st.write("Add your leftover ingredients one by one:")

# Input field and Add button
col_input, col_btn = st.columns([3, 1])
with col_input:
    st.text_input("Type an ingredient...", key="new_ing_input", on_change=add_ingredient)
with col_btn:
    st.write("") # Spacing to align button
    st.button("➕ Add", on_click=add_ingredient, use_container_width=True)

# Display current ingredients with Delete buttons beside them
if st.session_state.ingredients_list:
    st.markdown("### Your Leftovers:")
    for i, ing in enumerate(st.session_state.ingredients_list):
        c1, c2, c3 = st.columns([0.5, 3, 1])
        c1.markdown("🥘")
        c2.markdown(f"**{ing}**")
        if c3.button("❌ Delete", key=f"del_{i}_{ing}"):
            st.session_state.ingredients_list.pop(i)
            st.rerun()

    # Analyze Button
    st.write("")
    if st.button("🔍 Analyze Ingredients", type="primary"):
        if len(st.session_state.ingredients_list) > 0:
            processed_input = ' '.join([i.replace(' ', '_') for i in st.session_state.ingredients_list])
            vec_input = vectorizer.transform([processed_input])
            
            # Get probabilities to find the top 3 cuisines
            probabilities = model.predict_proba(vec_input)[0]
            top_3_indices = probabilities.argsort()[-3:][::-1]
            st.session_state.top_cuisines = model.classes_[top_3_indices]
            
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Please add at least one ingredient first!")

# ==========================================
# STEP 2: Checkbox Cuisine Selection
# ==========================================
if st.session_state.step >= 2:
    st.divider()
    st.header("2. Choose Your Cuisine")
    st.success("Our Supervised Learning model analyzed your ingredients and found these matching flavor profiles:")
    st.write("Select up to 3 cuisines to explore:")
    
    # Checkbox layout
    temp_selected_cuisines = []
    for cuisine in st.session_state.top_cuisines:
        if st.checkbox(f"**{cuisine.capitalize()}**", key=f"chk_{cuisine}"):
            temp_selected_cuisines.append(cuisine)
            
    if temp_selected_cuisines:
        if st.button("💡 Generate Recipe Ideas"):
            if not api_key:
                st.error("API Key required in the sidebar!")
            else:
                st.session_state.selected_cuisines = temp_selected_cuisines
                st.session_state.step = 3
                st.rerun()

# ==========================================
# STEP 3 & 4: Gen AI Ideas & Dropdown Select
# ==========================================
if st.session_state.step >= 3:
    st.divider()
    st.header("3. Recipe Ideas & Cost Analysis")
    
    if not st.session_state.recipes_data:
        with st.spinner("Chef Gemini is analyzing costs and brainstorming 10 recipes..."):
            gen_model = genai.GenerativeModel('gemini-3.5-flash')
            
            prompt = f"""
            I have these leftover ingredients: {', '.join(st.session_state.ingredients_list)}. 
            I want to cook a dish inspired by these cuisines: {', '.join(st.session_state.selected_cuisines)}.
            
            Suggest exactly 10 recipe ideas.
            
            IMPORTANT: Format your response EXACTLY like this for each recipe, separating each recipe with three dashes (---):
            1. [Recipe Name] ([Cuisine])
            - Description: [Brief Description]
            - Missing Ingredients: [List of missing ingredients]
            - Cost & Availability: [Cost and availability evaluation]
            - Important Ingredients: [Importance of missing ingredients and if they can be skipped]
            - Brief Summary of Cooking Process: [Summary of steps]

            For example, for a recipe called "Classic Garlic Pomodoro Pasta (Italian)", the output should look like this:
            1. Classic Garlic Pomodoro Pasta (Italian)
            - Description: A simple, rustic Italian classic where tomatoes are simmered with garlic and olive oil to create a light, fresh sauce for your pasta.
            - Missing Ingredients: Garlic, fresh basil, olive oil, and parmesan cheese.
            - Cost & Availability: Very cheap (under SGD$5 total) and available at any local grocery store.
            - Important Ingredients: Garlic and olive oil are absolutely essential to create the flavor base and cannot be skipped. The basil and parmesan add a fresh, salty finish; while highly recommended, they can be skipped if you are on a strict budget.
            - Brief Summary of Cooking Process: Start by sautéing minced garlic in olive oil until fragrant. Add chopped tomatoes and simmer until they break down into a sauce. Toss with cooked pasta and finish with fresh basil and grated parmesan. Serve immediately for the best flavor.

            ---
            """
            response = gen_model.generate_content(prompt)
            
            blocks = response.text.split("---")
            parsed_recipes = []
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                    
                lines = block.split('\n')
                name_line = ""
                content_lines = []
                
                # Find the title (line starting with a number) vs the bullet points
                for line in lines:
                    if re.match(r'^\d+\.', line.strip()) and not name_line:
                        name_line = line.strip()
                    elif line.strip():
                        content_lines.append(line.strip())
                
                if name_line:
                    # Clean the number off for the dropdown box (e.g. "1. Pasta (Italian)" -> "Pasta (Italian)")
                    clean_name = re.sub(r'^\d+\.\s*', '', name_line)
                    parsed_recipes.append({
                        "name": clean_name,           # Used for the selectbox dropdown
                        "full_title": name_line,      # Used for the display header
                        "description": '\n'.join(content_lines) # The formatted bullet points
                    })
            
            st.session_state.recipes_data = parsed_recipes
            st.rerun()
            
    if st.session_state.recipes_data:
        # Display the formatted recipe ideas
        for rec in st.session_state.recipes_data:
            st.markdown(f"#### {rec['full_title']}")
            st.markdown(rec['description'])
            st.write("---")
            
        # STEP 4: Select Box
        st.header("4. Select a Recipe to Cook")
        recipe_names = [r["name"] for r in st.session_state.recipes_data]
        selected = st.selectbox("Choose the recipe you want to make:", options=recipe_names)
        
        if st.button("👨‍🍳 Start Cooking!", type="primary"):
            st.session_state.selected_recipe = selected
            st.session_state.step = 5
            st.rerun()

# ==========================================
# STEP 5 & 6: Chat Interface 
# ==========================================
if st.session_state.step == 5:
    st.divider()
    
    # 1. Initialize the Chat on first load
    if st.session_state.chat_session is None:
        with st.spinner("Preparing your detailed recipe and securing the kitchen..."):
            
            # --- GUARDRAIL LAYER 1: API Safety Settings ---
            safety_settings = [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"}
            ]
            
            gen_model = genai.GenerativeModel(
                model_name='gemini-3.5-flash',
                safety_settings=safety_settings
            )
            st.session_state.chat_session = gen_model.start_chat(history=[])
            
            # --- GUARDRAIL LAYER 2: System Prompt ---
            initial_prompt = f"""
            You are an expert Chef Assistant for a food waste application. 
            The user has chosen to cook '{st.session_state.selected_recipe}' using these leftovers: {', '.join(st.session_state.ingredients_list)}. 
            
            TASK:
            First, say what ingredients are present and what is needed and then output the complete, detailed recipe including exact measurements and step-by-step instructions. 
            Then, enthusiastically tell the user you are ready to answer any questions they have while cooking.

            STRICT BEHAVIORAL GUARDRAILS:
            1. CULINARY ONLY: You must ONLY answer questions related to food, cooking, recipes, ingredients, and kitchen safety. 
            2. OFF-TOPIC REFUSAL: If the user asks about politics, coding, math, violence, or ANY non-culinary topic, you must politely refuse and say: "I am your kitchen assistant! I can only help you with cooking and food-related topics."
            3. FOOD SAFETY: Do not recommend eating raw or dangerously undercooked meats. Always prioritize safe food handling practices.
            """
            
            response = st.session_state.chat_session.send_message(initial_prompt)
            st.session_state.messages = [{"role": "assistant", "content": response.text}]
            
    # ---------------------------------------------------------
    # DISPLAYING THE CHAT UI
    # ---------------------------------------------------------
    st.header(f"5. Cooking: {st.session_state.selected_recipe}")
    st.subheader("👨‍🍳 Your Personal Kitchen Assistant")
    
    # Display the chat history 
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat Input for ongoing Q&A
    if user_question := st.chat_input("Ask the Chef a question (e.g., 'What can I substitute for eggs?')..."):
        
        # Display user question immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get and display AI response using the guardrailed chat session
        with st.chat_message("assistant"):
            with st.spinner("Chef is thinking..."):
                response = st.session_state.chat_session.send_message(user_question)
                st.markdown(response.text)
                
        # Save AI response to memory
        st.session_state.messages.append({"role": "assistant", "content": response.text})

# ==========================================
# RESET BUTTON
# ==========================================
st.write("")
st.write("")
st.write("")
st.divider()
if st.button("🔄 Reset Entire App", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()  