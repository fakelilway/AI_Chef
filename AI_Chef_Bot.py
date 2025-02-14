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
    st.write("Downloading spaCy model 'en_core_web_lg'...")
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
    # Normalize to lowercase for consistent matching
    ingredients = pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()
    return [ing.lower().strip() for ing in ingredients]

ingredient_list = load_ingredient_data()

# ✅ Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []
    for ing in ingredient_list:
        # Use the lowercased version for vector computation
        vec = nlp(ing).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec / np.linalg.norm(vec))  # Normalize vectors
    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# ✅ Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  # Angular distance works well on normalized vectors
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    index.build(50)  # More trees yield better accuracy
    return index

annoy_index = build_annoy_index()

# ✅ Direct Cosine Similarity Search (Most Accurate)
def direct_search_alternatives(ingredient):
    ingredient = ingredient.lower().strip()
    if ingredient not in filtered_ingredient_list:
        return ["Ingredient not found."]
    input_index = filtered_ingredient_list.index(ingredient)
    input_vector = ingredient_vectors[input_index]
    
    # Compute cosine similarity for all ingredients using vectorized dot product (vectors are normalized)
    similarities = np.dot(ingredient_vectors, input_vector)
    similarities[input_index] = -np.inf  # Exclude the ingredient itself
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_alternatives = [filtered_ingredient_list[i] for i in top_indices]
    return top_alternatives

# ✅ Annoy Search (Approximate Nearest Neighbors)
def annoy_search_alternatives(ingredient, search_k=-1):
    ingredient = ingredient.lower().strip()
    if ingredient not in filtered_ingredient_list:
        return ["Ingredient not found."]
    input_index = filtered_ingredient_list.index(ingredient)
    input_vector = ingredient_vectors[input_index]
    neighbor_indices = annoy_index.get_nns_by_vector(input_vector, 4, search_k=search_k)
    alternative_indices = [idx for idx in neighbor_indices if idx != input_index][:3]
    return [filtered_ingredient_list[idx] for idx in alternative_indices]

# ✅ Extract Ingredients from Recipe Text
def extract_ingredients(recipe_text):
    """
    Extracts the ingredients section from the recipe text.
    Assumes the recipe contains an "Ingredients:" section.
    """
    match = re.search(r"Ingredients:\s*(.+?)(?:\n[A-Z][a-z]+:|$)", recipe_text, re.DOTALL)
    if match:
        ingredients = re.sub(r"\s+", " ", match.group(1))
        return ingredients.strip()
    else:
        return recipe_text  # Fallback if extraction fails

# ✅ Get Recipe Nutrition using API Ninjas Nutrition API
def get_recipe_nutrition(recipe_text):
    """
    Uses the extracted ingredients as a query for the Nutrition API.
    Returns nutritional data as a list of dictionaries.
    """
    ingredients_query = extract_ingredients(recipe_text)
    api_url = "https://api.api-ninjas.com/v1/nutrition"
    headers = {"X-Api-Key": "klA9VFbUqkKyT8oT+9ohqA==rZjgY7vYmGkmEI4r"}  # Replace with your actual API key
    params = {"query": ingredients_query}
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == requests.codes.ok:
        try:
            nutrition_data = response.json()
            return nutrition_data
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            return None
    else:
        st.error(f"API error: {response.status_code} {response.text}")
        return None

# ✅ Recipe Generation Prompts
system_prompt_structured = (
    "Generate a recipe. First, give a title. Then list ingredients with specific units "
    "(pounds, kilograms, grams, etc.). Then provide step-by-step instructions."
)
system_prompt_concise = (
    "Generate a brief recipe. Include a catchy title, a short ingredients list with units, and concise instructions."
)
system_prompt_creative = (
    "Generate a creative recipe with unconventional ingredient pairings. Include a unique title, a list of ingredients with specific units, and imaginative step-by-step instructions."
)

def generate_recipe(ingredients, cuisine, temperature, top_k, top_p, num_beams):
    input_text = (
        f"Ingredients: {', '.join([ing.strip() for ing in ingredients.split(',')])}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with the ingredients listed above.\n"
        f"{system_prompt_structured}\n"
    )
    gen_kwargs = {
        "max_length": 250,
        "num_return_sequences": 1,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if num_beams is not None:
        gen_kwargs["num_beams"] = num_beams

    outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

# ✅ Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
ingredients_input = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox(
    "Select a cuisine:",
    ["Any", "Asian", "Indian", "Middle Eastern", "Mexican", "Western", "Mediterranean", "African"],
)
temperature = st.selectbox("Select Temperature:", [None, 0.5, 1.0, 2.0])
top_k = st.selectbox("Select Top-k sampling value:", [None, 5, 50])
top_p = st.selectbox("Select Top-p sampling value:", [None, 0.7, 0.95])
decoding_strategy = st.selectbox("Select Decoding Strategy:", [None, "Greedy Decoding", "Beam Search"])
num_beams = 1 if decoding_strategy == "Greedy Decoding" else 5 if decoding_strategy == "Beam Search" else None

if st.button("Generate Recipe", use_container_width=True) and ingredients_input:
    st.session_state["recipe"] = generate_recipe(ingredients_input, cuisine, temperature, top_k, top_p, num_beams)

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    recipe_text = st.session_state["recipe"]  # Save recipe for nutrition analysis
    st.text_area("Recipe:", recipe_text, height=200)
    st.download_button(
        label="📂 Save Recipe",
        data=recipe_text,
        file_name="recipe.txt",
        mime="text/plain",
    )

    # ✅ Nutrition Information Section
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
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)
    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives,
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
