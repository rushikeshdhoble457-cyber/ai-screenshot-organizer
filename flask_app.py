
import sqlite3
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, Response, redirect, url_for
from PIL import Image
import pytesseract

# AI semantic search
from sentence_transformers import SentenceTransformer, util


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = BASE_DIR / "screenshots.db"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ---------------- AI MODEL ----------------

model = None


def get_model():
    global model

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# ---------------- DATABASE ----------------

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


def save_screenshot(filename, text, image_data):
    category = categorize_screenshot(text)

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO screenshots
                (filename, extracted_text, image_data, category)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                extracted_text = excluded.extracted_text,
                image_data = excluded.image_data,
                category = excluded.category
        """, (filename, text, image_data, category))


def get_screenshots():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT id, filename, extracted_text, image_data, category
            FROM screenshots
            ORDER BY id DESC
        """).fetchall()


# ---------------- SEARCH ----------------

def search_screenshots(screenshots, query, mode):
    if not query:
        return screenshots

    query = query.casefold()

    if mode == "keyword":
        return [
            item for item in screenshots
            if query in (item[1] or "").casefold()
            or query in (item[2] or "").casefold()
        ]

    searchable = [
        item for item in screenshots
        if (item[2] or "").strip()
    ]

    if not searchable:
        return []

    ai_model = get_model()

    query_embedding = ai_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    documents = [
        f"Filename: {item[1]}. Screenshot text: {item[2]}"
        for item in searchable
    ]

    embeddings = ai_model.encode(
        documents,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(query_embedding, embeddings)[0]
    results = []

    for index, score in enumerate(scores):
        similarity = float(score)

        if similarity >= 0.30:
            results.append((searchable[index], similarity))

    results.sort(key=lambda result: result[1], reverse=True)

    return [item for item, score in results]


# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    error = None

    if request.method == "POST":
        files = request.files.getlist("screenshots")

        if not files or not any(file.filename for file in files):
            error = "Choose at least one screenshot to upload."
        else:
            successful = 0

            for file in files:
                if not file.filename:
                    continue

                try:
                    image_data = file.read()
                    image = Image.open(BytesIO(image_data))
                    image.verify()

                    image = Image.open(BytesIO(image_data))
                    extracted_text = pytesseract.image_to_string(
                        image, lang="eng"
                    ).strip()

                    save_screenshot(
                        file.filename,
                        extracted_text,
                        image_data,
                    )
                    successful += 1

                except Exception:
                    error = (
                        "Some files could not be processed. "
                        "Use valid PNG or JPG screenshots."
                    )

            if successful:
                message = f"Processed and saved {successful} screenshot(s)."

    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "semantic")
    category = request.args.get("category", "all")

    if mode not in {"semantic", "keyword"}:
        mode = "semantic"

    valid_categories = {
        "all", "education", "programming", "finance", "other"
    }

    if category not in valid_categories:
        category = "all"

    try:
        screenshots = get_screenshots()

        if category != "all":
            screenshots = [
                item for item in screenshots
                if (item[4] or "Other").casefold() == category
            ]

        results = search_screenshots(screenshots, query, mode)

    except Exception as exc:
        app.logger.exception("Could not load or search screenshots")
        results = []
        error = error or (
            "Unable to load search results. Check the database and "
            "AI model installation, then inspect the terminal error."
        )

    return render_template(
        "index.html",
        screenshots=results,
        total_count=len(get_screenshots()),
        query=query,
        mode=mode,
        category=category,
        message=message,
        error=error,
    )


from flask import Response

@app.route("/image/<int:image_id>", endpoint="screenshot_image")
def get_screenshot_image(image_id):
    with sqlite3.connect(DB_NAME) as conn:
        row = conn.execute(
            "SELECT image_data FROM screenshots WHERE id = ?",
            (image_id,)
        ).fetchone()

    if row is None or row[0] is None:
        return "Image not found", 404

    return Response(row[0], mimetype="image/png")

@app.route("/clear-library", methods=["POST"])
def clear_library():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM screenshots")
        conn.commit()

    return redirect(url_for("index"))


if __name__ == "__main__":
    init_database()
    app.run(debug=True)
