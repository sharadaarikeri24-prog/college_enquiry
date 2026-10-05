import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="MRECW College Enquiry Chatbot",
    page_icon="🎓",
    layout="centered"
)

# -------------------- COLLEGE INFORMATION --------------------
college_data = [
    {
        "id": "B1",
        "text": """B1 Block of Malla Reddy Engineering College for Women (MRECW).
B1 contains:
- Examination Branch Cell
- Data Science Branch"""
    },
    {
        "id": "B2",
        "text": """B2 Block of Malla Reddy Engineering College for Women (MRECW).
B2 contains:
- Administration section
- Half sections of CSE"""
    },
    {
        "id": "B3",
        "text": """B3 Block of Malla Reddy Engineering College for Women (MRECW).
B3 contains:
- HOD section
- AI/ML branch"""
    },
    {
        "id": "B4",
        "text": """B4 Block of Malla Reddy Engineering College for Women (MRECW).
B4 contains:
- Games
- Seniors"""
    },
    {
        "id": "B5",
        "text": """B5 Block of Malla Reddy Engineering College for Women (MRECW).
B5 contains:
- Auditorium
All events happen in the Auditorium."""
    },
    {
        "id": "B6",
        "text": """B6 Block of Malla Reddy Engineering College for Women (MRECW).
B6 contains:
- Remaining half of CSE students"""
    },
    {
        "id": "TRANSPORT",
        "text": """MRECW provides college bus transportation.
Route areas:
- Patancheru
- KPHB
- Kukatpally
- Alwal"""
    },
    {
        "id": "COLLEGE",
        "text": """Malla Reddy Engineering College for Women (MRECW).
The chatbot provides information about:
- Blocks
- Departments
- Examination Branch
- Administration
- AI/ML
- Data Science
- CSE
- Games
- Auditorium
- Events
- Seniors
- Transportation"""
    }
]

# -------------------- EMBEDDING MODEL --------------------
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error("Embedding model load avvaledu. Please install sentence-transformers.")
        st.exception(e)
        st.stop()

embedding_model = load_embedding_model()

# -------------------- VECTOR DATABASE --------------------
@st.cache_resource
def create_vector_database():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="mrecw_college_information"
    )

    existing_ids = collection.get().get("ids", [])

    new_data = [
        (item["id"], item["text"])
        for item in college_data
        if item["id"] not in existing_ids
    ]

    if new_data:
        ids, documents = zip(*new_data)
        embeddings = embedding_model.encode(
            list(documents),
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=list(ids),
            documents=list(documents),
            embeddings=embeddings
        )

    return collection

collection = create_vector_database()

# -------------------- RETRIEVAL --------------------
def retrieve_information(question, number_of_results=3):
    try:
        query_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        total = collection.count()
        if total == 0:
            return ""

        number_of_results = min(number_of_results, total)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=number_of_results
        )

        documents = results.get("documents", [[]])[0]
        return "\n\n".join(documents)

    except Exception as e:
        st.error("Information retrieval error.")
        st.exception(e)
        return ""

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_conversation_history():
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

# -------------------- GENERATE ANSWER --------------------
def generate_answer(user_question):
    history = get_conversation_history()

    retrieval_query = (
        history + "\nCURRENT QUESTION: " + user_question
    )

    context = retrieve_information(retrieval_query)

    if not context:
        context = "No relevant information was found in the college database."

    prompt = f"""
You are an AI-powered college enquiry chatbot for
Malla Reddy Engineering College for Women (MRECW).

Answer student questions using ONLY the information
provided in the COLLEGE INFORMATION section.

COLLEGE INFORMATION:
{context}

RECENT CONVERSATION:
{history}

IMPORTANT RULES:
1. Give simple and clear answers.
2. Do not invent or assume college information.
3. B1 contains Data Science and Examination Branch Cell.
4. B2 contains Administration and half sections of CSE.
5. B3 contains HOD section and AI/ML.
6. B4 contains Games and Seniors.
7. B5 contains Auditorium and all events.
8. B6 contains remaining half of CSE students.
9. CSE information is available in B2 and B6.
10. College bus transportation is available.
11. Transport routes are Patancheru, KPHB, Kukatpally and Alwal.
12. Use recent conversation for follow-up questions.
13. If information is unavailable, respond exactly:
"Sorry, I don't have that information."
14. Do not provide information about other colleges.
15. Keep answers short and student-friendly.

STUDENT QUESTION:
{user_question}

ANSWER:
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as e:
        return (
            "Sorry, I could not connect to Ollama. "
            "Please make sure Ollama is running and "
            "the llama3.2 model is installed.\n\n"
            f"Error: {str(e)}"
        )

# -------------------- USER INTERFACE --------------------
st.title("🎓 MRECW AI College Enquiry Chatbot")

st.write(
    "Ask questions about blocks, departments, "
    "transportation and college facilities."
)

st.divider()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_question = st.chat_input("Ask your question...")

if user_question:
    user_question = user_question.strip()

    if user_question:
        # Show user message
        with st.chat_message("user"):
            st.write(user_question)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(user_question)
            st.write(answer)

        # Save chatbot answer
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
