## Running Server Locally

NOTE : Put firebase_key.json file in app directory & .env file containing GOOGLE_API_KEY in root directory.

1. Create Virtual environment for (python<3.11)
    ```sh
    python -m virtualenv -p <python-executable-path> <env-name>
    ```
2. Install Python dependencies
    ```sh
    pip3 install -r requirements.txt
    ```
3. Initiate the Sqlite3 database with flask CLI command
    ```sh
    flask init-db
4. Start the Flask server
    ```sh
    python app.py
    ```
5. Access the application from browser
    ```sh
    http://localhost:5000
    ```


