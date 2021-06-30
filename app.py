from flask import request, render_template
from flask import Flask
import os
from tensorflow.keras.models import load_model

from media import *

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('main.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    f = request.files['file']
    f2 = request.files['file2']
    f.save(os.path.join(UPLOAD_FOLDER, f.filename))
    f2.save(os.path.join(UPLOAD_FOLDER, f2.filename))

    media = [Media(os.path.join(UPLOAD_FOLDER, m)) for m in os.listdir(UPLOAD_FOLDER) if ((m == f.filename) or (m == f2.filename))]
    model = load_model('ann copy.hdf5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    for m in media:
      m.mfcc()
    for s in m.subtitles():
      print(s)
      s.determine_speech(model)
      s.to_srt()
      s.to_vtt()
    prefix, ext = f.filename.split('.')
    f = open(f'static/{prefix}.srt', 'r').read()
    f = f.replace('\n', '<br>')
    
    return render_template('uploaded.html', filename=prefix, filecontent=f)
    # return "success!"
