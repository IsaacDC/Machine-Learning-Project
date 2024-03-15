async function translateText(text, targetLang) {
    // Make a request to your Python backend to translate the text
    const response = await fetch('/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text, targetLang })
    });
    const data = await response.json();
    return data.translation;
  }

  async function sendMessage() {
    const inputText = document.getElementById('new-message').value;
    const targetLanguage = document.getElementById('target-language').value;
    const translatedText = await translateText(inputText, targetLanguage);

    // Append the translated message to the chat window
    const messageWindow = document.getElementById('message-window');
    const li = document.createElement('li');
    const time = new Date().toLocaleTimeString();
    li.innerHTML = `<time datetime="${time}">(${time})</time><strong>You:</strong> ${inputText} (Translated: ${translatedText})`;
    messageWindow.appendChild(li);

    // Clear the input field after sending the message
    document.getElementById('new-message').value = '';
  }