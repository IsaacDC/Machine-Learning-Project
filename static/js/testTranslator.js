function sendMessage() {
  var message = document.getElementById('new-message').value;
  var targetLang = document.getElementById('target-language').value;

  // Send AJAX request to Flask for translation
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/translate', true);
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.onreadystatechange = function () {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
          var response = JSON.parse(xhr.responseText);
          var translatedMessage = response.translation;
          sendTranslatedMessage(translatedMessage);
      }
  };
  xhr.send('input_text=' + encodeURIComponent(message) + '&target_lang=' + targetLang);
}

function sendTranslatedMessage(translatedMessage) {
  // Emit WebSocket event to send translated message
  socketio.emit('message', translatedMessage);
}

document.getElementById('translation-form').addEventListener('submit', function(event) {
  event.preventDefault();

  var formData = new FormData(this);
  fetch('/translate', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('translation-result').innerText = data.translation;
  });
});
