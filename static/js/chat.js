document.addEventListener("DOMContentLoaded", () => {
  const socket = io.connect("http://" + document.domain + ":" + location.port);

    // Adds a new list item containing the message
    // Receives message data from handle_message() in backend
    socket.on('message', (data) => {
        const messageWindow = document.getElementById('message-window');
        const li = document.createElement('li');
        if (data.hasOwnProperty('originLanguage')) {
            li.innerHTML = '(' + data.timeSent + ') ' + '<strong>' + data.user + '</strong>' + ': ' + data.message + '<em>' + ' (Translated from: ' + data.originLanguage + ')' + '</em>';
        }
        else {
            li.innerHTML = '(' + data.timeSent + ') ' + '<strong>' + data.user + '</strong>' + ': ' + data.message;
        }
        messageWindow.appendChild(li);

    // Limits message history to 25 messages by removing top <li> when a new message is added
    const messages = messageWindow.getElementsByTagName("li");
    if (messages.length > 25) {
      messageWindow.removeChild(messages[0]);
    }
  });

    // On submit, gets message data and sends it to the Websocket
    document.getElementById('chat-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const newMessage = document.getElementById('new-message');
        const message = newMessage.value.trim();
        const checkbox = document.getElementById('translateCheckbox');
        if (message) {
            // Create Map with message data
            const messageData = new Map();
            if (checkbox.checked) {
                messageData.set('message', message);
                messageData.set('originLanguage', document.getElementById('originLanguage').value);
                messageData.set('targetLanguage', document.getElementById('targetLanguage').value);
                messageData.set('translate', true);
            }
            else {
                messageData.set('message', message);
                messageData.set('originLanguage', "None");
                messageData.set('targetLanguage', "None");
                messageData.set('translate', false);
            }
            // Message data sent to handle_message() in backend
            const jsonMessageData = JSON.stringify(Object.fromEntries(messageData))
            socket.emit('message', jsonMessageData);
            newMessage.value = '';
        }
    });
});

// Enables visibily to translation options when selected
document.addEventListener('DOMContentLoaded', function() {
    const checkbox = document.getElementById('translateCheckbox');
    const translationOptions = document.querySelector('.translation-options');

    checkbox.addEventListener('change', function() {
        if (checkbox.checked) {
            translationOptions.style.display = 'block';
        }
        else {
            translationOptions.style.display = 'none';
        }
    });
});
