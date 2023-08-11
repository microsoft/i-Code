let conversationContext = '';
let recorder;
let context;
let cnt = 0;
let asr_res = '';

function displayMsgDiv(str, who) {
  const time = new Date();
  let hours = time.getHours();
  let minutes = time.getMinutes();
  const ampm = hours >= 12 ? 'pm' : 'am';
  hours = hours % 12;
  hours = hours ? hours : 12; // the hour "0" should be "12"
  hours = hours < 10 ? '0' + hours : hours;
  minutes = minutes < 10 ? '0' + minutes : minutes;
  const strTime = hours + ':' + minutes + ' ' + ampm;
  let msgHtml = "<div class='msg-card-wide mdl-card " + who + "'><div class='mdl-card__supporting-text'>";
  msgHtml += str;
  msgHtml += "</div><div class='" + who + "-line'>" + strTime + '</div></div>';

  $('#messages').append(msgHtml);
  $('#messages').scrollTop($('#messages')[0].scrollHeight);

  if (who == 'user') {
    $('#q').val('');
    $('#q').attr('disabled', 'disabled');
    $('#p2').fadeTo(500, 1);
  } else {
    $('#q').removeAttr('disabled');
    $('#p2').fadeTo(500, 0);
  }
}

$(document).ready(function () {
  $('#q').attr('disabled', 'disabled');
  $('#p2').fadeTo(500, 1);
  $('#h').val('0');

  // Simply get the caption of the camera frame and send to conversation every 10 seconds
  // TODO:
  //    1. Figure out how to get the caption of the key frame
  //    2. only send the caption of the key frame to the conversation
  // var intervalId = setInterval(function() { CV() }, 10000);

  $.ajax({
    url: '/api/conversation',
    convText: '',
    context: ''
  })
    .done(function (res) {
      conversationContext = res.results.context;
      play(res.results.responseText);
      displayMsgDiv(res.results.responseText, 'bot');
    })
    .fail(function (jqXHR, e) {
      console.log('Error: ' + jqXHR.responseText);
    })
    .catch(function (error) {
      console.log(error);
    });
});


function callConversation(res) {
  $('#q').attr('disabled', 'disabled');

  $.post('/api/conversation', {
    convText: res,
    context: JSON.stringify(conversationContext),
  })
    .done(function (res, status) {
      conversationContext = res.results.context;
      play(res.results.responseText);
      displayMsgDiv(res.results.responseText, 'bot');

      $('#q').val('');
      recordMic.src = './static/img/mic.gif';
    })
    .fail(function (jqXHR, e) {
      console.log('Error: ' + jqXHR.responseText);
    });
}

function play(inputText) {
  let buf;
  $.post('/api/text-to-speech', {
    text: inputText
  });
}

function ASR (res) {
  $.post('/api/speech-to-text', {
    asr: JSON.stringify(asr_res),
  })
  .done(function (res, status) {
    console.log(res.results.asr)
    callConversation(res.results.asr);
    displayMsgDiv(res.results.asr, 'user');
  })
  .fail(function (jqXHR, e) {
    console.log('Error: ' + jqXHR.responseText);
  });
}

function CV(res){
  $.post('/api/computer-vision', {
    cv: JSON.stringify(asr_res),
  })
  .done(function (res, status) {
    console.log(res.results.cv)
    callConversation(res.results.cv);
    displayMsgDiv(res.results.cv, 'user');
  })
  .fail(function (jqXHR, e) {
    console.log('Error: ' + jqXHR.responseText);
  });
}

const recordMic = document.getElementById('stt2');
recordMic.onclick = function () {
  const fullPath = recordMic.src;
  const filename = fullPath.replace(/^.*[\\/]/, '');
  const url = '/api/speech-to-text';

  if (filename == 'mic.gif') {
    recordMic.src = './static/img/mic_active.png';
    $('#q').val('I am listening ...');
    ASR();
  } else {
    $('#q').val('');
    recordMic.src = './static/img/mic.gif';
  }
  cnt += 1

  // const request = new XMLHttpRequest();
  // request.open('POST', url, true);
  // request.onload = function () {
  //   callConversation(request.response);
  //   displayMsgDiv(request.response, 'user');
  // };

  // if (filename == 'mic.gif') {
  //   try {
  //     recordMic.src = './static/img/mic_active.png';
  //     $('#q').val('I am listening ...');

  //     const request = new XMLHttpRequest();
  //     request.open('POST', url, true);
  //     // console.log(request.response)

  //   } catch (ex) {
  //     console.log("Recognizer error .....");
  //   }
  // } else {
  //   callConversation(request.response);
  //   displayMsgDiv(request.response, 'user');
  //   $('#q').val('');
  //   recordMic.src = './static/img/mic.gif';
  // }
};

window.onload = function init() {
  try {
    // webkit shim
    window.AudioContext = window.AudioContext || window.webkitAudioContext;
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
    // eslint-disable-next-line
    window.URL = window.URL || window.webkitURL;

  } catch (e) {
    alert('No web audio support in this browser!');
  }
};
