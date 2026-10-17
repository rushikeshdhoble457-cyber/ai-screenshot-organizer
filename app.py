import sqlite3
from io import BytesIO

import streamlit as st
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# -------------------- CONFIGURATION --------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DB_NAME = "screenshots.db"


# -------------------- LOAD AI MODEL --------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


st.set_page_config(
    page_title="AI Screenshot Organizer",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ AI Screenshot Organizer")
st.write("Organize, categorize, and intelligently search your screenshots.")

with st.spinner("Loading AI semantic search model..."):
    model = load_model()


# -------------------- DATABASE --------------------

def init_database():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                extracted_text TEXT,
                image_data BLOB,
                category TEXT
            )
        """)

        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(screenshots)"
            ).fetchall()
        }

        if "image_data" not in columns:
            conn.execute(
                "ALTER TABLE screenshots ADD COLUMN image_data BLOB"
            )

        if "category" not in columns:
            conn.execute(
                "ALTER TABLE screenshots ADD COLUMN category TEXT"
            )


def categorize_screenshot(text):
    text = text.casefold()

    categories = {
        "Education": [
            "exam", "notes", "certificate", "college",
            "study", "assignment", "marks", "lecture"
        ],
        "Programming": [
            "python", "javascript", "error", "code",
            "programming", "function", "debug", "github"
        ],
        "Finance": [
            "payment", "transaction", "invoice", "bank",
            "balance", "receipt", "rupees", "₹"
        ],
    }

    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category

    return "Other"


def save_to_database(filename, text, image_data):
    category = categorize_screenshot(text)

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO screenshots
                (filename, extracted_text, image_data, category)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filename)
            DO UPDATE SET
                extracted_text = excluded.extracted_text,
                image_data = excluded.image_data,
                category = excluded.category
        """, (filename, text, image_data, category))


def load_from_database():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT filename, extracted_text, image_data, category
            FROM screenshots
            ORDER BY id DESC
        """).fetchall()


init_database()


# -------------------- UPLOAD AND OCR --------------------

uploaded_files = st.file_uploader(
    "Upload screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Extract text from all screenshots"):
    progress = st.progress(0)
    successful = 0

    for index, file in enumerate(uploaded_files):
        try:
            image_data = file.getvalue()
            image = Image.open(BytesIO(image_data))

            extracted_text = pytesseract.image_to_string(
                image,
                lang="eng",
            ).strip()

            save_to_database(
                file.name,
                extracted_text,
                image_data,
            )

            successful += 1

        except Exception as error:
            st.error(f"Could not process {file.name}: {error}")

        progress.progress((index + 1) / len(uploaded_files))

    st.success(f"Processed and saved {successful} screenshot(s).")


# -------------------- SEARCH OPTIONS --------------------

st.subheader("🔎 Search your screenshots")

search_mode = st.radio(
    "Choose search method",
    ["Keyword Search", "AI Semantic Search"],
    horizontal=True,
)

query = st.text_input(
    "What are you looking for?",
    placeholder="Example: Python, study notes, money I paid",
).strip()

selected_category = st.selectbox(
    "Filter by category",
    ["All", "Education", "Programming", "Finance", "Other"],
)


# -------------------- LOAD SAVED SCREENSHOTS --------------------

try:
    saved_screenshots = load_from_database()
except sqlite3.Error as error:
    st.error(f"Database error: {error}")
    saved_screenshots = []


# -------------------- FILTER AND RANK RESULTS --------------------

filtered_screenshots = [
    (filename, text or "", image_data, category or "Other")
    for filename, text, image_data, category in saved_screenshots
    if (
        selected_category == "All"
        or (category or "Other") == selected_category
    )
]

matches = []

if query and search_mode == "Keyword Search":
    normalized_query = query.casefold()

    matches = [
        item for item in filtered_screenshots
        if (
            normalized_query in item[0].casefold()
            or normalized_query in item[1].casefold()
        )
    ]

elif query and search_mode == "AI Semantic Search":
    # Compare the query meaning with screenshot text
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    searchable_items = [
        item for item in filtered_screenshots
        if item[1].strip()
    ]

    if searchable_items:
        document_texts = [
            f"Filename: {item[0]}. Screenshot text: {item[1]}"
            for item in searchable_items
        ]

        document_embeddings = model.encode(
            document_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarities = util.cos_sim(
            query_embedding,
            document_embeddings,
        )[0]

        ranked_indices = similarities.argsort(
            descending=True
        ).tolist()

        # Show reasonably similar results only
        for index in ranked_indices:
            score = float(similarities[index])

            if score >= 0.30:
                matches.append(
                    (*searchable_items[index], score)
                )


# -------------------- DASHBOARD COUNTS --------------------

col1, col2 = st.columns(2)

with col1:
    st.metric("Total saved screenshots", len(saved_screenshots))

with col2:
    st.metric("Matching screenshots", len(matches))


# -------------------- DISPLAY RESULTS --------------------

if query and matches:
    columns = st.columns(2)

    for index, item in enumerate(matches):
        filename, text, image_data, category = item[:4]

        with columns[index % 2]:
            st.markdown(f"**📄 {filename}**")
            st.caption(f"📂 Category: {category}")

            if search_mode == "AI Semantic Search":
                st.caption(f"Semantic similarity: {item[4]:.2f}")

            if image_data:
                st.image(
                    image_data,
                    use_container_width=True,
                )

            with st.expander("View extracted text"):
                st.text(text or "No text detected.")

elif query:
    st.info(
        "No matching screenshots found. Try another query or category."
    )

elif saved_screenshots:
    st.info("Enter a search query to find your saved screenshots.")

else:
    st.info("Upload screenshots and extract their text to get started.")
