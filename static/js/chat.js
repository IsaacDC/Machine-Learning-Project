document.addEventListener('DOMContentLoaded', () => {
    const socket = io.connect('http://' + document.domain + ':' + location.port);

    // Adds a new list item containing the message to the 'message-window' list in home page
    socket.on('message', (data) => {
        const messageWindow = document.getElementById('message-window');
        const li = document.createElement('li');
        li.textContent = '(' + data.time_sent + ') ' + data.user + ': ' + data.message;
        messageWindow.appendChild(li);

        // Limits message history to 25 messages
        const messages = messageWindow.getElementsByTagName('li');
        if (messages.length > 25) {
            messageWindow.removeChild(messages[0])
        }
    });

    // On submit, gets message data and sends it to the websocket to add to chat list
    document.getElementById('chat-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const newMessage = document.getElementById('new-message');
        const message = newMessage.value.trim();
        if (message) {
            socket.emit('message', message);
            newMessage.value = '';
        }
    });
});
