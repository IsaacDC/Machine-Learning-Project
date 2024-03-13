document.addEventListener('DOMContentLoaded', () => {
    const socket = io.connect('http://' + document.domain + ':' + location.port);

    // Adds a new list item containing the message
    // Receives message data from handle_message() in backend
    socket.on('message', (data) => {
        const messageWindow = document.getElementById('message-window');
        const li = document.createElement('li');
        li.innerHTML = '(' + data.time_sent + ') ' + '<strong>' + data.user + '</strong>' + ': ' + data.message;
        messageWindow.appendChild(li);

        // Limits message history to 25 messages by removing top <li> when a new message is added
        const messages = messageWindow.getElementsByTagName('li');
        if (messages.length > 25) {
            messageWindow.removeChild(messages[0])
        }
    });

    // On submit, gets message data and sends it to the Websocket
    document.getElementById('chat-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const newMessage = document.getElementById('new-message');
        const message = newMessage.value.trim();
        if (message) {
            // Message sent to handle_message() in backend
            socket.emit('message', message);
            newMessage.value = '';
        }
    });
});
