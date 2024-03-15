from flask import Flask, render_template, session, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from current_time import get_current_time
import random

from testTranslater import translate_text

# Assigns Flask application name
app = Flask(__name__)

# Assigns secret key to allow secure communication between site to backend data
app.config['SECRET_KEY'] = '9y2g9t1H8x3PT7Ej8UC92VsU'
# Assigns name of SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chats.db'

# Creates database object using SQLAlchemy
db = SQLAlchemy(app)
# Creates WebSocket using SocketIO
socketio = SocketIO(app)


# Database creation
# Columns defined for data the db will hold
class Chats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(250), nullable=False)
    user = db.Column(db.String(25), nullable=False)
    time_sent = db.Column(db.String(7), nullable=False)

    # Required, but not necessary.
    # Used for debugging via printing database attributes
    def __repr__(self):
        return ""

with app.app_context():
    db.create_all()


# Used for logging purposes
# Prints a confirmation that WebSocket is functioning
@socketio.on('connect')
def check_connection():
    print("WebSocket connected successfully.")


# Websocket functionality when it receives a message
@socketio.on('message')
def handle_message(message):
    # Add message along with extra data to Chats database
    user = session.get('user')
    time_sent = get_current_time()
    new_message = Chats(message=message, user=user, time_sent=time_sent)
    db.session.add(new_message)
    db.session.commit()

    # Sends message data to WebSocket in chat.js
    message_data = {"user": user, "message": message, "time_sent": time_sent}
    socketio.emit('message', message_data)


# Main Route ('/') also known as the Home Page
# Has access to POST and GET data to and from the backend
@app.route("/", methods=['POST', 'GET'])
def home_page():
    # Create a random username and place in session for easy access later on
    session['user'] = "User" + str(random.randint(100, 999))
    user = session.get('user')
    # Retrieve the latest 25 chats from Chats database (By sorting through highest Id)
    latest_chats = db.session.query(Chats).order_by(Chats.id.desc()).limit(25).all()
    latest_chats.reverse()
    # Load 'home.html' with access to user and latest_chats
    return render_template("home.html", user=user, latest_chats=latest_chats)


# Chat History *TESTING PURPOSES*
# Has access to GET data from the backend
@app.route("/chat-history", methods=['GET'])
def chat_history():
    # Get all Chats database data into a single variable
    chat_history = Chats.query.all()
    # Load 'chatHistory,html' with access to chat_history
    return render_template("chatHistory.html", chat_history=chat_history)

@app.route('/translate', method=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    target_lang = data.get('targetLang')

    translated_text = translate_text(text, target_lang)

    return jsonify({'translation' : translate_text})

# Runs a local server with WebSocket support
if __name__ == '__main__':
    socketio.run(app)
