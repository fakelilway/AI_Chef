import streamlit as st
import pandas as pd
import spacy
import os
import re
import requests
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download

    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"


@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()


ingredient_list = load_ingredient_data()


# ✅ Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients


ingredient_vectors, filtered_ingredient_list = compute_embeddings()


# ✅ Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(
        dim, metric="angular"
    )  # ✅ Uses angular distance (1 - cosine similarity)

    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)

    index.build(50)  # ✅ More trees = better accuracy
    return index


annoy_index = build_annoy_index()


# ✅ Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return (
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if np.any(vec1) and np.any(vec2)
        else 0
    )


def direct_search_alternatives(ingredient):
    # Check if the ingredient exists in our list
    if ingredient not in filtered_ingredient_list:
        return ["Ingredient not found."]

    # Retrieve the vector for the input ingredient
    input_index = filtered_ingredient_list.index(ingredient)
    input_vector = ingredient_vectors[input_index]

    # Compute cosine similarity scores with every other ingredient
    similarity_scores = []
    for i, other_ingredient in enumerate(filtered_ingredient_list):
        if other_ingredient == ingredient:
            continue  # Skip comparing with itself
        score = cosine_similarity(input_vector, ingredient_vectors[i])
        similarity_scores.append((other_ingredient, score))

    # Sort ingredients by similarity (highest first)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top-3 similar ingredients
    top_alternatives = [alt for alt, score in similarity_scores[:3]]
    return top_alternatives


# ✅ Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    # Check if the ingredient exists in our list
    if ingredient not in filtered_ingredient_list:
        return ["Ingredient not found."]

    # Retrieve the vector for the input ingredient
    input_index = filtered_ingredient_list.index(ingredient)
    input_vector = ingredient_vectors[input_index]

    # Use Annoy's nearest neighbor search to find the top 4 neighbors
    # (we ask for 4 to account for the input ingredient itself which will likely be the closest)
    neighbor_indices = annoy_index.get_nns_by_vector(input_vector, 4)

    # Filter out the input ingredient and return the top-3 nearest neighbors
    alternative_indices = [idx for idx in neighbor_indices if idx != input_index][:3]
    top_alternatives = [filtered_ingredient_list[idx] for idx in alternative_indices]

    return top_alternatives


def extract_ingredients(recipe_text):
    """
    Extracts only the ingredients section from the recipe text.
    Assumes that the generated recipe contains an "Ingredients:" section.
    This regex captures text from "Ingredients:" until a new section marker (e.g., "Instructions:")
    or the end of the text.
    """
    match = re.search(
        r"Ingredients:\s*(.+?)(?:\n[A-Z][a-z]+:|$)", recipe_text, re.DOTALL
    )
    if match:
        # Clean extra whitespace/newlines
        ingredients = re.sub(r"\s+", " ", match.group(1))
        return ingredients.strip()
    else:
        return recipe_text  # Fallback in case extraction fails


def get_recipe_nutrition(recipe_text):
    """
    Uses the extracted ingredients as a query for the Nutrition API.
    Returns the nutritional data as a list of dictionaries.
    """
    # Extract only the ingredients portion for a cleaner nutrition query
    ingredients_query = extract_ingredients(recipe_text)

    api_url = "https://api.api-ninjas.com/v1/nutrition?query={}".format(
        ingredients_query
    )
    headers = {
        "X-Api-Key": "klA9VFbUqkKyT8oT+9ohqA==rZjgY7vYmGkmEI4r"
    }  # API key for the Nutrition API
    response = requests.get(api_url, headers=headers)

    if response.status_code == requests.codes.ok:
        try:
            nutrition_data = response.json()
            return nutrition_data
        except Exception as e:
            print("Error parsing JSON:", e)
            return None
    else:
        print("API error:", response.status_code, response.text)
        return None


# ✅ Generate Recipe
# Add system prompt variations before generate_recipe function
# Variation A: Controlled structured output
system_prompt_structured = "Generate a recipe. First, give a title. Then list ingredients. Then provide step-by-step instructions."

# Variation B: Adjusted detail level (concise)
system_prompt_concise = "Generate a brief recipe. Include a catchy title, a short ingredients list, and concise step-by-step instructions."

# Variation C: Encourage creativity with unusual pairings
system_prompt_creative = "Generate a creative recipe with unconventional ingredient pairings. Include a unique title, a list of ingredients, and imaginative step-by-step instructions."


def generate_recipe(ingredients, cuisine, temperature, top_k, top_p, num_beams):
    input_text = (
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with ingredients listed above.\n"
        f"With specific unit of measurment for each ingredients (pounds, kilograms, grams listed.\n"
        f"{system_prompt_structured} Here are the preaparation and cooking instructions):\n"
    )

    # Build generation arguments only if parameters are not None
    gen_kwargs = {
        "max_length": 250,
        "num_return_sequences": 1,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }

    # None for isolation to avoid other parameters influencing when testing one parameter
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if num_beams is not None:
        gen_kwargs["num_beams"] = num_beams

    outputs = model.generate(
        tokenizer(input_text, return_tensors="pt")["input_ids"], **gen_kwargs
    )
    return (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .replace(input_text, "")
        .strip()
    )


# ✅ Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox(
    "Select a cuisine:",
    [
        "Any",
        "Asian",
        "Indian",
        "Middle Eastern",
        "Mexican",
        "Western",
        "Mediterranean",
        "African",
    ],
)

temperature = st.selectbox("Select Temperature:", [None, 0.5, 1.0, 2.0])
top_k = st.selectbox("Select Top-k sampling value:", [None, 5, 50])
top_p = st.selectbox("Select Top-p sampling value:", [None, 0.7, 0.95])
decoding_strategy = st.selectbox(
    "Select Decoding Strategy:", [None, "Greedy Decoding", "Beam Search"]
)
num_beams = 1 if decoding_strategy == "Greedy Decoding" else 5 if decoding_strategy == "Beam Search" else None


if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(
        ingredients, cuisine, temperature, top_k, top_p, num_beams
    )

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    recipe_text = st.session_state["recipe"]  # Save recipe for nutrition analysis
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(
        label="📂 Save Recipe",
        data=st.session_state["recipe"],
        file_name="recipe.txt",
        mime="text/plain",
    )

    # ✅ Nutrition Information Section
    # AI-powered Nutrition Analysis Feature: Extract ingredients and query the Nutrition API.
    nutrition_data = get_recipe_nutrition(recipe_text)

    if nutrition_data:
        st.markdown("### 🍎 Nutrition Information:")
        for item in nutrition_data:
            st.markdown(
                f"**{item.get('name', 'Item')}**\n"
                f"Calories: {item.get('calories', 'N/A')} kcal  \n"
                f"Protein: {item.get('protein_g', 'N/A')} g  \n"
                f"Fat: {item.get('fat_total_g', 'N/A')} g  \n"
                f"Carbohydrates: {item.get('carbohydrates_total_g', 'N/A')} g  \n"
                f"Fiber: {item.get('fiber_g', 'N/A')} g  \n"
                f"Sugar: {item.get('sugar_g', 'N/A')} g"
            )
            st.markdown("---")
    else:
        st.markdown("### 🍎 Nutrition Information: Not available")

    # ✅ Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio(
        "Select Search Method:",
        ["Annoy (Fastest)", "Direct Search (Best Accuracy)"],
        index=0,
    )

    if (
        st.button("🔄 Find Alternatives", use_container_width=True)
        and ingredient_to_replace
    ):
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives,
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(
            f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:"
        )
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
