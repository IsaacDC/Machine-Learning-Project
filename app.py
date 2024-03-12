from flask import Flask, render_template, session
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from current_time import get_current_time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = '9y2g9t1H8x3PT7Ej8UC92VsU'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chats.db'

db = SQLAlchemy(app)
socketio = SocketIO(app)

class Chats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(250), nullable=False)
    user = db.Column(db.String(25), nullable=False)
    time_sent = db.Column(db.String(7), nullable=False)

    def __repr__(self):
        return ""

with app.app_context():
    db.create_all()

@socketio.on('connect')
def check_connection():
    print("WebSocket connected successfully.")

# Websocket for chat
@socketio.on('message')
def handle_message(message):
    # Adds message to chat database
    user = session.get('user')
    time_sent = get_current_time()
    new_message = Chats(message=message, user=user, time_sent=time_sent)
    db.session.add(new_message)
    db.session.commit()

    # Sends message data to WebSocket in chat.js
    message_data = {"user": user, "message": message, "time_sent": time_sent}
    socketio.emit('message', message_data)

# Main Page
@app.route("/", methods=['POST', 'GET'])
def home_page():
    session['user'] = "User" + str(random.randint(100, 999))
    user = session.get('user')
    latest_chats = db.session.query(Chats).order_by(Chats.id.desc()).limit(25).all()
    latest_chats.reverse()
    return render_template("home.html", user=user, latest_chats=latest_chats)

# Chat History *TESTING PURPOSES*
@app.route("/chat-history", methods=['GET'])
def chat_history():
    chat_history = Chats.query.all()
    return render_template("chatHistory.html", chat_history=chat_history)

# Runs a local server with WebSocket support
if __name__ == '__main__':
    socketio.run(app)