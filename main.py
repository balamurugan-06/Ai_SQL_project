from flask import Flask, request, jsonify, render_template
import psycopg2
import os
import time
from groq import Groq
from dotenv import load_dotenv
import pandas as pd
import duckdb

uploaded_tables = {}


app = Flask(__name__)

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

client = Groq(api_key=GROQ_API_KEY)
sql_cache = {}

# ==========================
# Neon PostgreSQL
# ==========================
uploaded_tables = {}   # {table_name: [columns]}
sql_cache = {}


# ==========================
# SQL Generator
# ==========================
# ==========================
# SQL Generator
# ==========================

def extract_query_structure(question, column_info):
    q = question.lower()
    dimension = None

    for col in column_info["possible_dimensions"]:
        col_lower = col.lower()

        # Match partial words
        if col_lower in q or any(word in col_lower for word in q.split()):
            dimension = col
            break

    return {
        "dimension": dimension
    }

def detect_columns(df):
    mapping = {
        "numeric": [],
        "categorical": [],
        "date": [],
        "possible_metrics": [],
        "possible_dimensions": []
    }

    for col in df.columns:
        dtype = df[col].dtype

        # Detect numeric → metrics
        if pd.api.types.is_numeric_dtype(dtype):
            mapping["numeric"].append(col)
            mapping["possible_metrics"].append(col)

        # Detect datetime
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping["date"].append(col)
            mapping["possible_dimensions"].append(col)

        # Everything else → dimension
        else:
            mapping["categorical"].append(col)
            mapping["possible_dimensions"].append(col)

    return mapping

def generate_sql(user_input, table_name):
    global uploaded_tables

    if table_name not in uploaded_tables:
        return None

    # Normalize question
    user_input = normalize_question(user_input)

    # Load dataframe
    df = uploaded_tables[table_name]

    # Detect schema
    column_info = detect_columns(df)
    structure = extract_query_structure(user_input, column_info)
    intent = classify_intent(user_input)

    # Metric selection (SAFE)
    metric_columns = column_info["possible_metrics"]

    if len(metric_columns) >= 1:
        metric_expr = metric_columns[0]
    else:
        metric_expr = "*"

    # Dimension detection
    dimension = structure.get("dimension")

    if not dimension and column_info["possible_dimensions"]:
        dimension = column_info["possible_dimensions"][0]

    # Schema text (ONLY selected table)
    schema_text = f"""
    Table: {table_name}
    Columns: {', '.join(df.columns)}
    """

    # Prompt
    prompt = f"""
You are a STRICT SQL generator.

Schema:
{schema_text}

User Question:
{user_input}

Intent:
{intent}

Rules:
- Use ONLY provided columns
- Only SELECT queries
- No nested aggregation
- If aggregation needed → use SUM({metric_expr})
- If grouping → GROUP BY {dimension} if applicable
- Use LIMIT 50
- No explanations

Return ONLY SQL.
"""

    # Call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    sql = response.choices[0].message.content.strip()

    # Clean SQL
    if "SELECT" in sql.upper():
        sql = sql[sql.upper().find("SELECT"):]
        sql = sql.split(";")[0]

        # Safety check
        if not sql.upper().startswith("SELECT"):
            return None

        # Prevent dangerous queries
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
        if any(word in sql.upper() for word in forbidden):
            return None

        # Add LIMIT if missing
        if "LIMIT" not in sql.upper():
            sql += " LIMIT 50"

        sql += ";"
        return sql

    return None

def clean_sql(sql):
    sql_upper = sql.upper()

    if "AVG(" in sql_upper and "average" not in sql_upper:
        sql = sql.replace("AVG", "--REMOVED_AVG")

    if "COUNT(" in sql_upper and "count" not in sql_upper:
        sql = sql.replace("COUNT", "--REMOVED_COUNT")

    return sql

# ==========================
# Routes
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():

    data = request.get_json()

    user_input = normalize_question(data.get("question", ""))
    table_name = data.get("table")

    if not table_name:
        return jsonify({"error": "No table selected"})

    if table_name not in uploaded_tables:
        return jsonify({"error": "Invalid table selected"})

    sql_query = generate_sql(user_input, table_name)
    sql_query = clean_sql(sql_query)

    if not sql_query:
        return jsonify({"error": "SQL generation failed"})

    try:
        print("Executing:", sql_query)

        if not sql_query.strip().upper().startswith("SELECT"):
            return jsonify({"error": "Only SELECT queries allowed"})
        
        result = duckdb.query(sql_query).to_df()

        if result.empty:
            return jsonify({
                "generated_sql": sql_query,
                "message": "No data found"
            })

        return jsonify({
            "generated_sql": sql_query,
            "columns": result.columns.tolist(),
            "data": result.values.tolist()
        })

    except Exception as e:
        return jsonify({
            "generated_sql": sql_query,
            "error": str(e)
        })


@app.route("/chart-suggestion", methods=["POST"])
def chart_suggestion():

    data = request.json
    columns = data.get("columns")
    rows = data.get("data")

    if not rows:
        return jsonify({"error": "No data"})

    numeric_columns = []
    categorical_columns = []
    date_columns = []

    for i, col in enumerate(columns):

        sample_values = [row[i] for row in rows[:10] if row[i] is not None]

        is_numeric = True
        is_date = True

        for val in sample_values:
            try:
                float(val)
            except:
                is_numeric = False

            try:
                pd.to_datetime(val)
            except:
                is_date = False

        if is_numeric:
            numeric_columns.append(col)
        elif is_date:
            date_columns.append(col)
        else:
            categorical_columns.append(col)

    suggestions = []

    # Time-based chart
    if len(date_columns) >= 1 and len(numeric_columns) >= 1:
        suggestions = ["line", "area"]

    # Category vs metric
    elif len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        suggestions = ["bar", "pie"]

    # Numeric vs numeric
    elif len(numeric_columns) >= 2:
        suggestions = ["scatter", "line"]

    # Single metric
    elif len(numeric_columns) == 1:
        suggestions = ["bar"]

    else:
        suggestions = ["table"]

    return jsonify({
    "suggestions": suggestions,
    "categorical": categorical_columns,
    "numeric": numeric_columns,
    "date": date_columns
})


@app.route("/visualize", methods=["POST"])
def visualize():

    data = request.json
    columns = data.get("columns")
    rows = data.get("data")

    if not rows:
        return jsonify({"error": "No data to visualize"})

    numeric_columns = []
    categorical_columns = []
    date_columns = []

    for i, col in enumerate(columns):

        sample_values = [row[i] for row in rows[:10] if row[i] is not None]

        is_numeric = True
        is_date = True

        for val in sample_values:
            try:
                float(val)
            except:
                is_numeric = False

            try:
                pd.to_datetime(val)
            except:
                is_date = False

        if is_numeric:
            numeric_columns.append(col)
        elif is_date:
            date_columns.append(col)
        else:
            categorical_columns.append(col)

    # Smart chart selection (OUTSIDE loop)
    # Priority 1: Time series
    if len(date_columns) >= 1 and len(numeric_columns) >= 1:
        chart_type = "line"

    # Priority 2: Category vs metric
    elif len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        chart_type = "bar"

    # Priority 3: Small dataset
    elif len(rows) <= 5 and len(numeric_columns) >= 1:
        chart_type = "pie"

    # Priority 4: Numeric comparison
    elif len(numeric_columns) >= 2:
        chart_type = "scatter"

    else:
        chart_type = "table"

    return jsonify({
    "chart_type": chart_type,
    "columns": columns,
    "data": rows,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "date_columns": date_columns
})

    


@app.route("/get_data", methods=["POST"])
def get_data():

    data = request.get_json()
    table_name = data.get("table")

    if not table_name:
        return jsonify({"error": "No table selected"})

    if table_name not in uploaded_tables:
        return jsonify({"error": "Invalid table"})

    try:
        df = uploaded_tables[table_name]

        return jsonify({
            "columns": df.columns.tolist(),
            "data": df.head(100).values.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    


@app.route("/insight", methods=["POST"])
def generate_insight():

    data = request.json
    columns = data.get("columns")
    rows = data.get("data")

    if not rows:
        return jsonify({"error": "No data available"})

    import pandas as pd

    df = pd.DataFrame(rows, columns=columns)

    # Limit rows
    sample_rows = rows[:20]

    # Convert to readable format
    table_text = "Columns: " + ", ".join(columns) + "\n"

    for row in sample_rows:
        table_text += ", ".join([str(r) for r in row]) + "\n"

    # Summary
    try:
        summary = df.describe(include='all').to_string()
    except:
        summary = "Summary not available"

    prompt = f"""
You are a senior data analyst.

Analyze the dataset and provide insights in STRICT format:

Key Insight:
<1-2 lines>

Trend:
<1-2 lines>

Recommendation:
<1-2 lines>

Rules:
- Be concise
- Use business language
- No extra explanation

Data Sample:
{table_text}

Statistical Summary:
{summary}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        insight = response.choices[0].message.content.strip()

        return jsonify({"insight": insight})

    except Exception as e:
        return jsonify({"error": str(e)})
    


@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_tables

    file = request.files["file"]

    if not file:
        return jsonify({"error": "No file uploaded"})

    filename = file.filename

    try:
        uploaded_tables = {}

        if filename.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")

            table_name = filename.replace(".csv", "").replace(" ", "_")

            duckdb.register(table_name, df)
            uploaded_tables[table_name] = df

        elif filename.endswith(".xlsx"):
            excel_file = pd.ExcelFile(file)

            for sheet in excel_file.sheet_names:
                df = excel_file.parse(sheet)

                sheet_clean = sheet.replace(" ", "_")

                duckdb.register(sheet_clean, df)
                uploaded_tables[sheet_clean] = df

        else:
            return jsonify({"error": "Unsupported file type"})

        return jsonify({
            "message": "File uploaded successfully",
            "tables": list(uploaded_tables.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)})


def classify_intent(question):
    q = question.lower()

    if any(word in q for word in ["average", "avg", "mean"]):
        return "average"

    elif any(word in q for word in ["count", "how many", "number of"]):
        return "count"

    elif any(word in q for word in ["max", "maximum", "highest"]):
        return "max"

    elif any(word in q for word in ["min", "minimum", "lowest"]):
        return "min"

    elif any(word in q for word in ["top", "best", "highest", "top 5", "top 10"]):
        return "top"

    elif any(word in q for word in ["total", "sum", "overall", "sales", "revenue"]):
        return "total"

    elif any(word in q for word in ["trend", "over time", "growth"]):
        return "trend"

    elif any(word in q for word in ["compare", "comparison", "difference"]):
        return "compare"

    else:
        return "general"
    

def normalize_question(question):
    q = question.lower()

    replacements = {
        "best selling": "top",
        "most selling": "top",
        "highest sales": "top",
        "top selling": "top",
        "performance": "summary",
        "how many": "count",
        "number of": "count",
        "total number": "count",
        "overall": "total",
        "sum of": "total"
    }

    for k, v in replacements.items():
        q = q.replace(k, v)

    return q


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
