from flask_socketio import SocketIO
from app import create_app, socketio
# 25 dollar per month = file.io
app = create_app()

if __name__ == "__main__":
    socketio.run(app)
