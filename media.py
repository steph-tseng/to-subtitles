import os
import librosa
import pysrt
from pysrt import SubRipTime
import sys
from pathlib import Path
import re
from datetime import timedelta
import chardet
from moviepy.editor import *

import numpy as np

from ffmpeg import Transcode

class Media:
  """
  The Media class represents a media file to be retrieved and analyzed
  """

  # Supported media formats
  FORMATS = ['.mkv', '.mp4', '.wmv', '.avi', '.flv']

  # The frequency of the generated audio
  FREQ = 16000

  # The number of coefficients to extract from the mfcc
  N_MFCC = 13

  # The number of samples in each mfcc coefficient
  HOP_LEN = 512.0

  # The length (seconds) of each item in the mfcc analysis
  LEN_MFCC = HOP_LEN/FREQ

  def __init__(self, filepath, subtitles=None, WPS=5):
      prefix, ext = os.path.splitext(filepath)
      if ext == '.srt':
        return self.from_srt(filepath)
      if ext == '.txt':
        return self.from_txt(filepath)
      if not ext:
        raise ValueError(f'Unknown file: "{filepath}"')
      if ext not in Media.FORMATS:
        raise ValueError(f'Filetype {ext} not supported: "{filepath}"')
      self.__subtitles = subtitles
      self.filepath = os.path.abspath(filepath)
      self.filename = os.path.basename(prefix)
      self.extension = ext
      self.offset = timedelta()
      self.WPS = WPS # Words (spoken) per second
  
  def from_srt(self, filepath):
    prefix, ext = os.path.splitext(filepath)
    if ext != 'srt':
      raise ValueError('Filetype must be .srt')
    prefix = os.path.basename(re.sub(r'\.\w\w$', '', prefix))
    dir = os.path.dirname(filepath)
    for f in os.listdir(dir):
      _, ext = os.path.splitext(f)
      if f.startswith(prefix) and ext in Media.FORMATS:
        return self.__init__(os.path.join(dir, f), subtitles=[filepath])
    raise ValueError(f'No media for subtitle: "{filepath}"')

  def from_txt(self, filepath):
    prefix, ext = os.path.splitext(filepath)
    if ext != '.txt':
      raise ValueError('Filetype must be .txt')
    prefix = os.path.basename(re.sub(r'\.\w\w$', '', prefix))
    dir = os.path.dirname(filepath)
    for f in os.listdir(dir):
      _, ext = os.path.splitext(f)
      if prefix in f and ext in Media.FORMATS:
        return self.__init__(os.path.join(dir, f), subtitles=[filepath])
    raise ValueError(f'No media for subtitle: "{filepath}"')
  
  def subtitles(self):
    if self.__subtitles is not None:
      for s in self.__subtitles:
        yield(Text(self, s))
    else:
      dir = os.path.dirname(self.filepath)
      for f in os.listdir(dir):
        if '.txt' in f and self.filename in f:
          yield(Text(self, os.path.join(dir, f)))

  def mfcc(self, duration=60*15, seek=True):
    transcode = Transcode(self.filepath, duration=duration, seek=seek)
    self.offset = transcode.start
    print('Transcoding...')
    transcode.run()
    y, sr = librosa.load(transcode.output, sr=Media.FREQ)
    self.mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(Media.HOP_LEN), n_mfcc=int(Media.N_MFCC))
    clip = AudioFileClip(transcode.output)
    self.dur = clip.duration
    os.remove(transcode.output)
    return self.mfcc

class Subtitle:
  """
  Subtitle class represents a .srt file on the disk and provides the functionality to inspect and manipulate the contents
  """

  def __init__(self, media, path):
    self.media = media
    self.path = path
    self.subs = pysrt.open(self.path)

  def srt_to_transcript(self):
    filename, _ = os.path.splitext(self.path)
    subs = pysrt.open(self.path)
    with open(f'{filename}.txt', 'w+') as f:
        for sub in subs:
            f.write(f'{sub.text}\n')


class Text:
  """
  Text class reads .txt file and converts it to .srt
  """

  def __init__(self, media, path):
    self.media = media
    self.path = path
    self.lines = open(self.path)
  
  def determine_speech(self, model):
    print('determine')
    mfcc = self.media.mfcc.T
    mfcc = mfcc[..., np.newaxis]
    y_pred = model.predict(mfcc)
    y_pred = y_pred.reshape(-1,)
    num_chunks = round(len(y_pred)/self.media.dur)
    chunks = [ y_pred[i:i+num_chunks] for i in range(0, len(y_pred), num_chunks) ]
    self.__secs = [ round(sum(i)/len(i)) for i in chunks ]
    return self.__secs

  def to_srt(self):
    print('before')
    with open(self.path) as f:
        text = f.read()
        text = text.replace('\n\n', '\n').split('\n')

    with open(f'static/{self.media.filename}.srt', 'w+') as f: # creating a new srt file
        print(f'{self.media.filename}.srt')
        print('Creating srt...')
        num = 1
        times = []

        for i, value in enumerate(self.__secs, start=0):
          if i > 1 and i < len(text):
              num_words = len(text[i-1].split(' '))
              if num_words > self.media.WPS:
                  continue
          if value == 1:
              sec = i
              times.append(sec)

        for i, time in enumerate(times):
            num_words = len(text[i].split(' '))
            if num_words > self.media.WPS:
                add = 2
            else:
                add = 1
            if not text[i+1]:
                print(i, text[i])
                break
            if time > 3600:
                hours = time // 3600
            else:
                hours = 0
            mins = (time - hours*3600) // 60
            secs = (time - hours*3600) % 60
            print(f'{num}\n{hours:02}:{mins:02}:{secs:02},000 --> {hours:02}:{mins:02}:{secs+add:02},000\n{text[i]}\n\n')
            f.write(f'{num}\n{hours:02}:{mins:02}:{secs:02},000 --> {hours:02}:{mins:02}:{secs+add:02},000\n{text[i]}\n\n')
            num += 1
    
    self.media.srt = f'output/new_{self.media.filename}.srt'
    return self.media.srt

  def to_vtt(self):
    print('before')
    with open(self.path) as f:
        text = f.read()
        text = text.replace('\n\n', '\n').split('\n')

    with open(f'static/{self.media.filename}.vtt', 'w+') as f: # creating a new vtt file
        f.write('WEBVTT\nKind: subtitles\nLanguage: en')
        print(f'{self.media.filename}.vtt')
        print('Creating vtt...')
        num = 1
        times = []

        for i, value in enumerate(self.__secs, start=0):
          if i > 1 and i < len(text):
              num_words = len(text[i-1].split(' '))
              if num_words > self.media.WPS:
                  continue
          if value == 1:
              sec = i
              times.append(sec)

        for i, time in enumerate(times):
            num_words = len(text[i].split(' '))
            if num_words > self.media.WPS:
                add = 2
            else:
                add = 1
            if not text[i+1]:
                print(i, text[i])
                break
            if time > 3600:
                hours = time // 3600
            else:
                hours = 0
            mins = (time - hours*3600) // 60
            secs = (time - hours*3600) % 60
            print(f'{num}\n{hours:02}:{mins:02}:{secs:02}.000 --> {hours:02}:{mins:02}:{secs+add:02}.000\n{text[i]}\n\n')
            f.write(f'{num}\n{hours:02}:{mins:02}:{secs:02}.000 --> {hours:02}:{mins:02}:{secs+add:02}.000\n{text[i]}\n\n')
            num += 1
    
    self.media.vtt = f'static/{self.media.filename}.vtt'
    return self.media.vtt


# Convert timestamp to seconds
def timeToSec(t):
    total_sec = float(t.milliseconds)/1000
    total_sec += t.seconds
    total_sec += t.minutes*60
    total_sec += t.hours*60*60
    return total_sec

# Return timestamp from cell position
def timeToPos(t, freq=Media.FREQ, hop_len=Media.HOP_LEN):
    return round(timeToSec(t)/(hop_len/freq))


def secondsToBlocks(s, hop_len=Media.HOP_LEN, freq=Media.FREQ):
    return int(float(s)/(hop_len/freq))


def blocksToSeconds(h, freq=Media.FREQ, hop_len=Media.HOP_LEN):
    return float(h)*(hop_len/freq)
