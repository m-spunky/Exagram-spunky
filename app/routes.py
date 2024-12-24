from flask import Blueprint, render_template, request, session, jsonify
from flask_socketio import emit, join_room, leave_room
import os
import firebase_admin
from firebase_admin import credentials, storage
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import logging
from app.db import get_db
from .auth import login_required, load_logged_in_user
from flask_socketio import emit, leave_room, join_room as flask_join_room
from . import socketio
import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import os

import firebase_admin
from firebase_admin import credentials, storage
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
bp = Blueprint('routes', __name__)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



questions_dict = {}
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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
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
            documents.append(document)
    return documents

def get_conversational_chain_buddy():
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

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def evaluate_answers(questions_dict):
    documents = convert_dict_to_documents(questions_dict)
    # Embeddings and FAISS indexing omitted for simplicity, adjust accordingly
    

    chain = get_conversational_chain_buddy()
    response = chain.invoke(
        {"input_documents": documents, "question": "Evaluate this answersheet."},
        return_only_outputs=True
    )
    return response["output_text"]


@bp.route('/', methods=('GET', 'POST'))
@bp.route('/dashboard', methods=('GET', 'POST'))
@login_required
def dashboard():
    state = session.get('dashboard', {})
    return render_template('dashboard.html', state=state)


@bp.route('/exaconnect', methods=('GET', 'POST'))
@login_required
def exaconnect():
    return redirect(url_for("chat.create_room"))


# Routes
@bp.route('/upload_buddy', methods=['POST'])
def upload_buddy():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("pdfs")
        if uploaded_files:
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
    return 'Done'

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
    while n<5:
        n+=1
        try:
            global questions_dict
            
            result_dict = json.loads(response)  # Parse the JSON string into a dictionary
            
            questions_dict=result_dict

            print(questions_dict)
            break
        except json.JSONDecodeError as e:

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
    print(evaluation)
    try:
        evaluation = json.loads(evaluation)
        print(evaluation)
    except Exception as e:
        print(e)
        return render_template('evaluation_result_2.html', evaluation=evaluation)
    return render_template('evaluation_result.html', evaluation=evaluation)



@bp.route('/testgen', methods=('GET', 'POST'))
@login_required
def testgen():
    return render_template('index2.html')



