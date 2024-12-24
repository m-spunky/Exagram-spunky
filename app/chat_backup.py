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

# Initialize Firebase Admin SDK
# cred = credentials.Certificate("app/firebase_key.json")
# firebase_admin.initialize_app(cred, {'storageBucket': 'canteeno-6136.appspot.com'})

# Uncomment following line to print DEBUG logs
#  logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

# Chat is the root blueprint, no url_prefix specified
bp = Blueprint('chat', __name__)


# File upload configurations
UPLOAD_FOLDER = 'uploads'  # You can change this to your preferred folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'docx', 'txt'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ROUTES
@bp.route('/', methods=('GET', 'POST'))
@bp.route('/create-room', methods=('GET', 'POST'))
@login_required
def create_room():
    """Create a unique chat room."""
    if request.method == "POST":
        db = get_db()
        has_error = False
        f = request.form

        if not f["room_name"]:
            flash("Room name is required.", "warning")
            has_error = True

        room_name = f["room_name"].strip().lower()

        if not has_error:
            try:
                db.execute("""INSERT INTO chat_room (created_by_user, name, password, description)
                        VALUES (?, ?, ?, ?)""", (session["user_id"], room_name,
                                                 generate_password_hash(f["password"]),
                                                 f["description"].strip()))
                db.commit()
            except db.IntegrityError:
                message = Markup(f"A room with the name <b>{room_name}</b> already exists.")
                flash(message, "warning")
            else:
                message = Markup(f"Successfully created the room: <b>{room_name}</b>.")
                flash(message, "info")

    return render_template("chat/create_room.html")


@bp.route('/join-room', methods=('GET', 'POST'))
@login_required
def join_room():
    """Join a chat room."""
    if request.method == "POST":
        db = get_db()
        f = request.form

        # Form check
        if not f["room_name"]:
            flash("Room name is required.", "warning")
        else:
            room_name = f["room_name"].strip().lower()
            room = db.execute("""SELECT chat_room_id, password FROM chat_room
                    WHERE name = ?""", (room_name,)).fetchone()

            if room is None:
                message = Markup(f"Could not find a room with the name <b>{room_name}</b>.")
                flash(message, "warning")
            elif not check_password_hash(room["password"], f["password"]):
                flash("Wrong password.", "warning")
            # Correct inputs, redirect to chat_room route
            else:
                room_id = room["chat_room_id"]
                session["room_id"] = room_id
                return redirect(url_for("chat.live_chat"))

    return render_template("chat/join_room.html")


# Route to handle file upload
# @bp.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Save the file to a folder
#     file_path = os.path.join("uploads", file.filename)
#     file.save(file_path)

#     # Return the file URL or path
#     file_url = f"/uploads/{file.filename}"
#     # emit('file_upload', {'fileUrl': file_url},namespace="/live-chat")
    
#     return jsonify({"fileUrl": file_url})





import requests

# Function to upload a file
def upload_file_io(file_path, upload_url):
    try:
        # Open the file and send it as a POST request
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)
        
        # Print the server's response
        if response.status_code == 200:
            print("Upload successful!")
            return response.json()
        else:
            print(f"Failed to upload. Status code: {response.status_code}")
            print("Response:", response.text)
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Please check the path.")
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

# Function to fetch a file
def fetch_file_io(file_url,filename):
    try:
        save_path = f"uploads/{filename}"
        # Send a GET request to the URL
        response = requests.get(file_url)

        
        # Check if the request was successful
        if response.status_code == 200:
            print("Request successful!")
             # Save the content to a local file
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"File saved to {save_path}")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            print("Response:", response.text)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")


# Route to handle file upload to Firebase Storage
@bp.route('/upload', methods=['POST'])
def upload_file():

    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # FILE.IO
    # # File to upload
    # file_to_upload = file_path
    # upload_service_url = 'https://file.io'

    # # Upload the file
    # result = upload_file_io(file_to_upload, upload_service_url)
    
    # # File URL to fetch (update with the correct URL from the upload response)
    # file_download_url = str(result['link'])
    # filename = str(result['name'])
    # # fetch_file_io(file_download_url,filename)
    # ---------------------------------------------------------------------------------------------

    # FIREBASE

    # Upload to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f'images/{file.filename}')
    blob.upload_from_filename(file_path)

    # Make the file publicly accessible
    blob.make_public()

    # Get the public URL of the file
    file_url = blob.public_url

    # Optionally, remove the temporary file after upload
    os.remove(file_path)

    return jsonify({"fileUrl": file_url})


@bp.route('/live-chat', methods=('GET', 'POST'))
@login_required
def live_chat():
    """A page to send and receive messages via chat room."""
    if "room_id" not in session:
        flash("Unauthorized access to the room. Please enter the password.", "warning")
        return redirect(url_for("chat.join_room"))
    db = get_db()
    chat_room_id = session["room_id"]
    room = db.execute("""SELECT * FROM chat_room
            WHERE chat_room_id = ?""", (chat_room_id,)).fetchone()

    if request.method == "POST":
        pass
        # if 'file' not in request.files:
        #     flash("No file part", "warning")
        #     return redirect(request.url)

        # file = request.files['file']
        # if file.filename == '':
        #     flash("No selected file", "warning")
        #     return redirect(request.url)

        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file_path = os.path.join(UPLOAD_FOLDER, filename)
        #     file.save(file_path)

        #     file_url = url_for('static', filename='uploads/' + filename)
        #     emit('message', {'user': load_logged_in_user()["short_name"], 'msg': file_url}, room=session["room_id"])


    # GET Request
    if room is None:
        flash("The owner closed this room.", "warning")
        return redirect("chat.join_room")

    return render_template("chat/live_chat.html", room=room)


@bp.route('/leave-chat')
@login_required
def leave_chat():
    if "room_id" in session:
        session.pop("room_id")
    return redirect(url_for("dashboard"))


# SocketIO Events
# Note: namespace is not the same as route. Socket.io namespaces
# just allow you to split logic of application over single shared connection.
# Note: You cannot modify session inside socket.io events. Instead
# create a route and redirect to that route.
@socketio.on('connect', namespace="/live-chat")
def test_connect():
    """Test SocketIO connection by passing message between server and client."""
    logging.debug("SocketIO: Connected to client")


@socketio.on('joined', namespace="/live-chat")
def joined(message):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    room = session["room_id"]
    user = load_logged_in_user()
    flask_join_room(room)

    emit('status', {'msg': user["short_name"] + ' has entered the room.'}, room=room)


@socketio.on('message', namespace="/live-chat")
def chat_message(message):
    """Sent by a client when the user entered a new message.
    The message is sent to all people in the room."""
    room = session["room_id"]
    user = load_logged_in_user()

    emit('message', {'user': user["short_name"], 'msg': message['msg']}, room=room)

# SocketIO event to handle file upload from client
@socketio.on('file_upload', namespace="/live-chat")
def handle_file_upload(data):
    room = session["room_id"]
    # Here, we assume that `data['fileUrl']` contains the uploaded file URL
    file_url = data['fileUrl']
    emit('file_upload', {'fileUrl': file_url}, room=room)


@socketio.on('left', namespace="/live-chat")
def left(message):
    """Sent by clients when they leave a room.
    A status message is broadcast to all people in the room."""
    room = session['room_id']
    user = load_logged_in_user()

    leave_room(room)
    emit('status', {'msg': user["short_name"] + ' has left the room.'}, room=room)

# -------------------------------------------------------------------------------------------------------------------
# from flask import (
#     Blueprint, flash, g, redirect, render_template, request, session, url_for
# )
# from markupsafe import Markup
# from werkzeug.security import check_password_hash, generate_password_hash
# from werkzeug.utils import secure_filename
# import os
# import logging
# from app.db import get_db
# from .auth import login_required, load_logged_in_user
# from flask_socketio import emit, leave_room, join_room as flask_join_room
# from . import socketio

# # Uncomment following line to print DEBUG logs
# # logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

# # Chat is the root blueprint, no url_prefix specified
# bp = Blueprint('chat', __name__)

# # File upload configurations
# UPLOAD_FOLDER = 'uploads'  # You can change this to your preferred folder
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'docx', 'txt'}

# # Helper function to check allowed file types
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# # ROUTES
# @bp.route('/', methods=('GET', 'POST'))
# @bp.route('/create-room', methods=('GET', 'POST'))
# @login_required
# def create_room():
#     """Create a unique chat room."""
#     if request.method == "POST":
#         db = get_db()
#         has_error = False
#         f = request.form

#         if not f["room_name"]:
#             flash("Room name is required.", "warning")
#             has_error = True

#         room_name = f["room_name"].strip().lower()

#         if not has_error:
#             try:
#                 db.execute("""INSERT INTO chat_room (created_by_user, name, password, description)
#                         VALUES (?, ?, ?, ?)""", (session["user_id"], room_name,
#                                                  generate_password_hash(f["password"]),
#                                                  f["description"].strip()))
#                 db.commit()
#             except db.IntegrityError:
#                 message = Markup(f"A room with the name <b>{room_name}</b> already exists.")
#                 flash(message, "warning")
#             else:
#                 message = Markup(f"Successfully created the room: <b>{room_name}</b>.")
#                 flash(message, "info")

#     return render_template("chat/create_room.html")


# @bp.route('/join-room', methods=('GET', 'POST'))
# @login_required
# def join_room():
#     """Join a chat room."""
#     if request.method == "POST":
#         db = get_db()
#         f = request.form

#         # Form check
#         if not f["room_name"]:
#             flash("Room name is required.", "warning")
#         else:
#             room_name = f["room_name"].strip().lower()
#             room = db.execute("""SELECT chat_room_id, password FROM chat_room
#                     WHERE name = ?""", (room_name,)).fetchone()

#             if room is None:
#                 message = Markup(f"Could not find a room with the name <b>{room_name}</b>.")
#                 flash(message, "warning")
#             elif not check_password_hash(room["password"], f["password"]):
#                 flash("Wrong password.", "warning")
#             # Correct inputs, redirect to chat_room route
#             else:
#                 room_id = room["chat_room_id"]
#                 session["room_id"] = room_id
#                 return redirect(url_for("chat.live_chat"))

#     return render_template("chat/join_room.html")


# @bp.route('/live-chat', methods=('GET', 'POST'))
# @login_required
# def live_chat():
#     """A page to send and receive messages via chat room."""
#     if "room_id" not in session:
#         flash("Unauthorized access to the room. Please enter the password.", "warning")
#         return redirect(url_for("chat.join_room"))
#     db = get_db()
#     chat_room_id = session["room_id"]
#     room = db.execute("""SELECT * FROM chat_room
#             WHERE chat_room_id = ?""", (chat_room_id,)).fetchone()

#     if request.method == "POST":
#         if 'file' not in request.files:
#             flash("No file part", "warning")
#             return redirect(request.url)

#         file = request.files['file']
#         if file.filename == '':
#             flash("No selected file", "warning")
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(file_path)

#             file_url = url_for('static', filename='uploads/' + filename)
#             emit('message', {'user': load_logged_in_user()["short_name"], 'msg': file_url}, room=session["room_id"])

#     # GET Request
#     if room is None:
#         flash("The owner closed this room.", "warning")
#         return redirect("chat.join_room")

#     return render_template("chat/live_chat.html", room=room)


# @bp.route('/leave-chat')
# @login_required
# def leave_chat():
#     if "room_id" in session:
#         session.pop("room_id")
#     return redirect(url_for("chat.index"))


# # SocketIO Events
# @socketio.on('connect', namespace="/live-chat")
# def test_connect():
#     """Test SocketIO connection by passing message between server and client."""
#     logging.debug("SocketIO: Connected to client")


# @socketio.on('joined', namespace="/live-chat")
# def joined(message):
#     """Sent by clients when they enter a room.
#     A status message is broadcast to all people in the room."""
#     room = session["room_id"]
#     user = load_logged_in_user()
#     flask_join_room(room)

#     emit('status', {'msg': user["short_name"] + ' has entered the room.'}, room=room)


# @socketio.on('message', namespace="/live-chat")
# def chat_message(message):
#     """Sent by a client when the user entered a new message.
#     The message is sent to all people in the room."""
#     room = session["room_id"]
#     user = load_logged_in_user()

#     emit('message', {'user': user["short_name"], 'msg': message['msg']}, room=room)


# @socketio.on('left', namespace="/live-chat")
# def left(message):
#     """Sent by clients when they leave a room.
#     A status message is broadcast to all people in the room."""
#     room = session['room_id']
#     user = load_logged_in_user()

#     leave_room(room)
#     emit('status', {'msg': user["short_name"] + ' has left the room.'}, room=room)
