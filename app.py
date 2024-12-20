from flask import Flask, render_template, session, request, jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Routes for each sidebar option
@app.route('/')
@app.route('/dashboard')
def dashboard():
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
