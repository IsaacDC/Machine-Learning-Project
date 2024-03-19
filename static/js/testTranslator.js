function translateAndSend() {
  const inputText = document.getElementById('new-message').value.trim();
  const targetLang = document.getElementById('target-language').value;

  if (inputText) {
    // Send AJAX request to the /translate route
    $.ajax({
      type: 'POST',
      url: '/translate',
      data: {
        input_text: inputText,
        target_lang: targetLang
      },
      success: function(response) {
        const translatedText = response.translation;
        const translationOutput = document.getElementById('translation-output');
        translationOutput.textContent = `Translated: ${translatedText}`;

        // Send the translated message over the WebSocket
        socket.emit('message', translatedText);
        document.getElementById('new-message').value = '';
      },
      error: function() {
        alert('An error occurred during translation.');
      }
    });
  } else {
    const errorSpan = document.getElementById('new-message-error');
    errorSpan.textContent = 'Please enter a message.';
  }
}

// Handle form submission for translation
const chatForm = document.getElementById('chat-form');
chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  translateAndSend();
});