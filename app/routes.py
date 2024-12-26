#------------------------------------------------------------------------------------------------------
# IMPORTS 

from flask import Blueprint, render_template, request, session, jsonify,flash
import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from .auth import login_required, load_logged_in_user
from flask_socketio import emit, leave_room, join_room as flask_join_room
from . import socketio
import os
from flask import Flask, request, jsonify
import os
import firebase_admin
from flask import Blueprint, request, jsonify
import os
from flask import Flask, request, render_template, redirect, session, jsonify,url_for
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langchain.schema import Document
from dotenv import load_dotenv
import google.generativeai as genai
import sqlite3


bp = Blueprint('routes', __name__)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




# -------------------------------------------------------------------------------------------
# DASHBOARD



from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import os
from datetime import datetime



# Data for test scores

test_scores = [
    {"test_id": 101, "score_achieved": 42, "total_score": 50, "date": datetime(2024, 12, 16)},
    {"test_id": 102, "score_achieved": 48, "total_score": 60, "date": datetime(2024, 12, 17)},
    {"test_id": 103, "score_achieved": 28, "total_score": 40, "date": datetime(2024, 12, 18)},
    {"test_id": 104, "score_achieved": 30, "total_score": 50, "date": datetime(2024, 12, 19)},
     {"test_id": 105, "score_achieved": 17, "total_score": 20, "date": datetime(2024, 12, 21)},
       {"test_id": 106, "score_achieved": 15, "total_score": 30, "date": datetime(2024, 12, 20)},
         {"test_id": 107, "score_achieved": 25, "total_score": 40, "date": datetime(2024, 12, 21)},
               {"test_id": 108, "score_achieved": 20, "total_score": 20, "date": datetime(2024, 12, 21)},
]
# Data for areas of improvement
areas_of_improvement = {
    "101": ["Depth of Explanation", "Time Management"],
    "102": ["Answer Length", "Coverage of Key Points"],
    "103": ["Structured Explanations"],
    "104": ["Time Management", "Depth of Explanation"],
    "105": ["Coverage of Key Points"],
    "106": ["Answer Length"],
    "107": ["Structured Explanations", "Coverage of Key Points"],
    "108": ["Time Management", "Depth of Explanation"]
}

# Data for uploaded files
uploaded_files = [
    {"file_id": 1, "file_name": "AI.pdf", "uploaded_date": "2024-12-16", "file_type": "PDF"},
    {"file_id": 2, "file_name": "DWM.pdf", "uploaded_date": "2024-12-17", "file_type": "PDF"},
    {"file_id": 3, "file_name": "WCN.pdf", "uploaded_date": "2024-12-18", "file_type": "PDF"},
     {"file_id": 4, "file_name": "SAID.pdf", "uploaded_date": "2024-12-19", "file_type": "PDF"},
      {"file_id": 5, "file_name": "DS.pdf", "uploaded_date": "2024-12-20", "file_type": "PDF"},
       {"file_id": 6, "file_name": "CG.pdf", "uploaded_date": "2024-12-20", "file_type": "PDF"},
        {"file_id": 7, "file_name": "OS.pdf", "uploaded_date": "2024-12-21", "file_type": "PDF"},
         {"file_id": 8, "file_name": "CN.pdf", "uploaded_date": "2024-12-19", "file_type": "PDF"},
]

areas_of_improvement = {
    "101": [
        "Depth of Explanation: Answers lack sufficient detail and technical terminology.",
        "Time Management: Allocate sufficient time to cover all key points expected for the assigned marks."
    ],
    "102": [
        "Coverage of Key Points: Some key concepts and components of AI systems are missing in the explanations.",
        "Answer Length: Answers are concise and do not fully address the allocated marks."
    ],
    "103": [
        "Structured Explanations: Practice providing more detailed and organized explanations.",
        "Time Management: Allocate sufficient time to cover all key points expected for the assigned marks."
    ],
    "104": [
        "Depth of Explanation: Provide detailed explanations using technical terminology.",
        "Answer Length: Answers are concise and do not fully address the allocated marks."
    ],
    "105": [
"Depth of Explanation: Provide detailed explanations using technical terminology.",
        "Answer Length: Answers are concise and do not fully address the allocated marks.",
                "Coverage of Key Points: Some key concepts and components of AI systems are missing in the explanations.",
                "Time Management: Allocate sufficient time to cover all key points expected for the assigned marks."
    ]
}



@bp.route("/visualize-scores", methods=["GET"])
def visualize_scores():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    # Filter scores based on date range
    filtered_scores = test_scores
    if start_date and end_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        filtered_scores = [score for score in test_scores if start_date <= score["date"] <= end_date]

    test_ids = [score["test_id"] for score in filtered_scores]
    percentages = [
        (score["score_achieved"] / score["total_score"]) * 100 for score in filtered_scores
    ]

    # Create a line chart
    plt.figure(figsize=(8, 4))
    plt.plot(test_ids, percentages, marker="o", color="blue", label="Test Score Percentages")
    for i, percentage in enumerate(percentages):
        plt.text(test_ids[i], percentages[i], f"{percentage:.1f}%", ha="center", va="bottom")

    # Set X and Y axis labels and title
    plt.xticks(test_ids)
    plt.xlabel("Test IDs")
    plt.ylabel("Percentage (%)")
    plt.title("Test Score Percentages Over Tests")

    # Remove top and right spines
    ax = plt.gca()  # Get the current axis
    ax.spines["top"].set_visible(False)  # Remove the top spine
    ax.spines["right"].set_visible(False)  # Remove the right spine

    # Keep left and bottom spines for X and Y axes
    ax.spines["left"].set_position(("outward", 0))  # Move the left spine slightly outward
    ax.spines["bottom"].set_position(("outward", 0))  # Move the bottom spine slightly outward

    # Disable grid
    plt.grid(False)

    # Save the chart as an image
    plt.savefig("static/scores.png")
    plt.close()

    return jsonify({"message": "Scores visualization generated.", "image_path": "/static/scores.png"})


@bp.route("/get-files", methods=["GET"])
def get_files():
    return jsonify(uploaded_files)


@bp.route("/get-improvements", methods=["GET"])
def get_improvements():
    test_id = request.args.get("test_id")
    improvements = areas_of_improvement.get(test_id, [])
    return jsonify({"test_id": test_id, "improvements": improvements})


@bp.route('/', methods=('GET', 'POST'))
@bp.route('/dashboard', methods=('GET', 'POST'))
@login_required
def dashboard():
    state = session.get('dashboard', {})
    return render_template('dashboard.html', state=state,test_ids=list(areas_of_improvement.keys()))


# -------------------------------------------------------------------------------------------
# EXACONNECT



@bp.route('/exaconnect', methods=('GET', 'POST'))
@login_required
def exaconnect():
    return redirect(url_for("chat.create_room"))




# -------------------------------------------------------------------------------------------
# TESTGEN

model_name ="gemini-1.5-flash"
#model_name ="gemini-pro"

questions_dict = {}
evaluation_dict = {}

# Helper functions
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Consider context as syllabus and generate questions and subtopic based on inputs, provide it in Python dictionary form. The dictionary should have section names as keys, and each section should contain a list of questions. Each question should be a dictionary with the following keys: question_no, question, marks, and subtopic. Context: {context}? Inputs: {question} Output should be a JSON with the following example structure as per section mentioned in Inputs: {{"Section 1": {{"questions": [{{"question_no": 1, "question": "question text", "marks": marks, "subtopic": "subtopic text"}}, {{"question_no": 2, "question": "question text", "marks": marks, "subtopic": "subtopic text"}}]}}, "Section 2": {{"questions": [{{"question_no": 1, "question": "question text", "marks": marks, "subtopic": "subtopic text"}}, {{"question_no": 2, "question": "question text", "marks": marks, "subtopic": "subtopic text"}}]}}}}. If PDF is not provided, just say "PDF unavailable". Don't provide the wrong answer.
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input_chain(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

def convert_dict_to_documents(questions_dict):
    documents = []
    for section, content in questions_dict.items():
        for question in content['questions']:
            print(question)
            doc_content = f"Question: {question['question']}\nAnswer: {question['answer']}\nMarks: {question['marks']}"
            document = Document(page_content=doc_content)
            documents.bpend(document)
    return documents

def get_conversational_chain_testgen():
    prompt_template = prompt_template = """
You are an evaluator for a theoretical answer sheet. Your task is to assess the overall performance of the student based on their responses across all questions provided. Each question has specific marks allocated, and your evaluation should consider the expected depth and completeness of the answers relative to the total marks for the paper.
Context:
{context}
DO:
{question}
Tasks:
1. Calculate and evaluate the marks scored by the student, out of the total marks provided with each questions.
2. Identify key areas where the answersheet could be improved.
3. Offer general suggestions for improving performance across the entire paper.

Output Format is string of dictionary containing:
- marks_scored : [Total Marks Scored] out of [Total Marks].
- total_marks : Total Marks of question paper
- improvement_areas: [List of Weaknesses Identified].
- improvement_suggest: [General Suggestions for Future Improvement].
"""

    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def evaluate_answers(questions_dict):
    documents = convert_dict_to_documents(questions_dict)
    # Embeddings and FAISS indexing omitted for simplicity, adjust accordingly
    

    chain = get_conversational_chain_testgen()
    response = chain.invoke(
        {"input_documents": documents, "question": "Evaluate this answersheet."},
        return_only_outputs=True
    )
    return response["output_text"]


# Routes
@bp.route('/upload_buddy', methods=['POST'])
def upload_buddy():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("pdfs")
        if uploaded_files:
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
    flash("Document Uploaded", "info")
    return jsonify({'done': "done"}), 200

@bp.route('/result', methods=['GET', 'POST'])
def result():
    user_question = ""
    total_sections = 0
    result_dict = {}

    if request.method == 'POST':
        total_sections = int(request.form.get('total_sections', 0))
        sections = []

        for i in range(total_sections):
            section_type = request.form.get(f'section_type_{i}')
            section_questions = request.form.get(f'section_questions_{i}', '0')
            section_marks = request.form.get(f'section_marks_{i}', '0')
            sections.append({
                'type': section_type,
                'questions': section_questions,
                'marks': section_marks
            })

        user_question = ""
        for i, section in enumerate(sections):
            user_question += f"""
            Section no: {i + 1};
            Question Type: {section['type']};
            Number of Questions: {section['questions']};
            Marks per Question: {section['marks']};
            """

    if user_question:
        response = user_input_chain(user_question)
        print(response)


    n = 0
    global questions_dict
    while n<5:
        n+=1
        try:
            print(response)  
            result_dict = json.loads(response)  # Parse the JSON string into a dictionary
            questions_dict=result_dict
            print(questions_dict)
            break
        except json.JSONDecodeError as e:
            if response.startswith("```json"):
                inner_json = response[7:-4].strip()  # Remove `'''json ` and trailing `'''`
                response = inner_json



            print(f"Error occurred: {e}. Retrying...")

    return render_template('result.html', result=result_dict)  # Pass the parsed result to the template

@bp.route('/solve', methods=['GET', 'POST'])
def solve():
    global questions_dict
    if request.method == 'POST':
        # Iterate through the sections and questions to collect answers
        for section, content in questions_dict.items():
            if content["questions"]:
                for question in content["questions"]:
                    q_no = question['question_no']
                    answer = request.form.get(f'{section}_answer_{q_no}', '')
                    question['answer'] = answer
            else:
                for question in content:
                    q_no = question['question_no']
                    answer = request.form.get(f'{section}_answer_{q_no}', '')
                    question['answer'] = answer

        # return redirect(url_for('view_answers'))
    # print(questions_dict)
    return render_template('solve_questions.html', sections=questions_dict)

@bp.route('/view', methods=['GET'])
def view_answers():
    global questions_dict
    print(questions_dict)
    return render_template('view_answers.html', sections=questions_dict)

@bp.route('/evaluate_answers_buddy', methods=['POST'])
def evaluate_answers_buddy():
    # Evaluate the answers from the form
    evaluation = evaluate_answers(questions_dict)

    n = 1
    try:
        result_dict = json.loads(evaluation)  # Parse the JSON string into a dictionary
        evaluation_dict[f"test_{n}"] = result_dict
        
    except json.JSONDecodeError as e:
        if evaluation.startswith("```json"):
            inner_json = evaluation[7:-4].strip()  # Remove `'''json ` and trailing `'''`
            evaluation = inner_json
        print(evaluation)
        result_dict = json.loads(evaluation)  # Parse the JSON string into a dictionary
        evaluation_dict[f"test_{n}"] = result_dict

    return render_template('evaluation_result.html', evaluation=result_dict)



@bp.route('/testgen', methods=('GET', 'POST'))
@login_required
def testgen():
    return render_template('index2.html')




# -------------------------------------------------------------------------------------------
# EXABUDDY


# Define upload folder for temporary storage
UPLOAD_FOLDER = 'uploads_buddy'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state for chat history
history = []
generated = ["Hello! Ask me anything about 🤗"]
past = ["Hey! 👋"]
uploaded_document_names = []  # Global list to store uploaded document names



# Utility to extract text from PDFs
def get_pdf_text_buddy(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks_buddy(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Build and save vector store
def get_vector_store_buddy(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define the conversational chain
def get_conversational_chain_buddy():
    prompt_template = """
    Consider the context as syllabus and answer the question based on the syllabus.
    If the answer is not in the provided context, respond with: "Answer is not available in the context."
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user queries
def user_input_chain_buddy(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain_buddy()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@bp.route('/upload_exabuddy', methods=['POST'])
def upload_files():
    try:
        global uploaded_document_names
        uploaded_files = request.files.getlist("pdfs")
        if uploaded_files:
            for file in uploaded_files:
                uploaded_document_names.append(file.filename)  # Store file name
            raw_text = get_pdf_text_buddy(uploaded_files)
            text_chunks = get_text_chunks_buddy(raw_text)
            get_vector_store_buddy(text_chunks)
            return jsonify({'status': 'Processed successfully', 'files': uploaded_document_names}), 200
        return jsonify({'error': 'No files uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/ask_exabuddy', methods=['POST'])
def ask_question():
    try:
        user_question = request.form.get('question')
        if user_question:
            response = user_input_chain_buddy(user_question)
            past.append(user_question)
            generated.append(response)
            print(user_question)
        return redirect(url_for('routes.buddy'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/clear_exabuddy', methods=['POST'])
def clear_history():
    global history, generated, past
    history.clear()
    generated = ["Hello! Ask me anything about 🤗"]
    past = ["Hey! 👋"]
    return redirect(url_for('routes.buddy'))



@bp.route('/buddy', methods=['GET', 'POST'])
@login_required
def buddy():
    return render_template('routes/buddy.html', generated=generated, past=past, zip=zip,uploaded_document_names=uploaded_document_names)



# -------------------------------------------------------------------------------------------
# PYQS


# Function to get data from the database
def get_data(query, params=()):
    conn = sqlite3.connect('exam_questions.db')
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result


@bp.route('/subjects/<int:semester_id>')
def get_subjects(semester_id):
    # Fetch subjects for a specific semester
    subjects = get_data("SELECT * FROM Subject WHERE semester_id = ?", (semester_id,))
    print(subjects)
    return jsonify(subjects)

@bp.route('/chapters/<int:subject_id>')
def get_chapters(subject_id):
    # Fetch chapters for a specific subject
    chapters = get_data("SELECT * FROM Chapter WHERE subject_id = ?", (subject_id,))
    print(chapters)
    return jsonify(chapters)

@bp.route('/questions/<int:chapter_id>')
def get_questions(chapter_id):
    # Fetch questions for a specific chapter
    questions = get_data("SELECT * FROM Question WHERE chapter_id = ?", (chapter_id,))
    print(questions)
    return jsonify(questions)

@bp.route('/pyqs')
def pyqs():
    # Fetch all semesters
    semesters = get_data("SELECT * FROM Semester")
    print(semesters)
    return render_template('pyqs.html', semesters=semesters)





# -------------------------------------------------------------------------------------------
# EXAGURU


# Global state for chat history
history_guru = []
generated_guru = ["Hello! Ask me anything about 🤗"]
past_guru = ["Hey! 👋"]


# Handle user queries
def user_input_chain_guru(user_question):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(user_question)
    return response.text

@bp.route('/ask_exaguru', methods=['POST'])
def ask_question_exaguru():
    try:
        user_question = request.form.get('question')
        if user_question:
            response = user_input_chain_guru(user_question)
            past_guru.append(user_question)
            generated_guru.append(response)
            print(user_question)
        return redirect(url_for('routes.exaguru'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/clear_exaguru', methods=['POST'])
def clear_history_exaguru():
    global history_guru, generated_guru, past_guru
    history_guru.clear()
    generated_guru = ["Hello! Ask me anything about 🤗"]
    past_guru = ["Hey! 👋"]
    return redirect(url_for('routes.exaguru'))

@bp.route('/exaguru', methods=('GET', 'POST'))
@login_required
def exaguru():
    return render_template('routes/exaguru.html', generated=generated_guru, past=past_guru, zip=zip)

