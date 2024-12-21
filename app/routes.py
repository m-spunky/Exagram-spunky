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


bp = Blueprint('routes', __name__)

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



