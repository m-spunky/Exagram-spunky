from flask import Flask, render_template, session, request, jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key


import firebase_admin
from firebase_admin import credentials, auth
from flask import Flask, request, jsonify, render_template, redirect, session, url_for

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase-key.json')
firebase_admin.initialize_app(cred)

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            # Create a new user
            user = auth.create_user(email=email, password=password)
            return jsonify({'message': 'User created successfully', 'uid': user.uid})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            # Verify the user's credentials
            user = auth.get_user_by_email(email)
            session['user_id'] = user.uid
            return redirect(url_for('dashboard'))
        except Exception as e:
            return jsonify({'error': 'Invalid credentials or user not found'}), 401
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    state = session.get('dashboard', {})
    return render_template('dashboard.html', state=state)

@app.route('/exabuddy', methods=['GET', 'POST'])
def exabuddy():
    if request.method == 'POST':
        session['exabuddy'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('exabuddy', {})
    return render_template('exabuddy.html', state=state)

@app.route('/testgen', methods=['GET', 'POST'])
def testgen():
    if request.method == 'POST':
        session['testgen'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('testgen', {})
    return render_template('testgen.html', state=state)

@app.route('/exaguru', methods=['GET', 'POST'])
def exaguru():
    if request.method == 'POST':
        session['exaguru'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('exaguru', {})
    return render_template('exaguru.html', state=state)

@app.route('/exaconnect', methods=['GET', 'POST'])
def exaconnect():
    if request.method == 'POST':
        session['exaconnect'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('exaconnect', {})
    return render_template('exaconnect.html', state=state)

@app.route('/exavault', methods=['GET', 'POST'])
def exavault():
    if request.method == 'POST':
        session['exavault'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('exavault', {})
    return render_template('exavault.html', state=state)

@app.route('/pyqs', methods=['GET', 'POST'])
def pyqs():
    if request.method == 'POST':
        session['pyqs'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('pyqs', {})
    return render_template('pyqs.html', state=state)

@app.route('/more', methods=['GET', 'POST'])
def more():
    if request.method == 'POST':
        session['more'] = request.json.get('state', {})
        return jsonify(success=True)
    state = session.get('more', {})
    return render_template('more.html', state=state)

if __name__ == '__main__':
    app.run(debug=True)
