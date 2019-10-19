import anvil.server
anvil.server.connect('ZDE4BVXBH6MECAZKXQXCENFM-MY6ODHGTFVIAC5GR')
import anvil.media
import socket
import io
import os
#from random import randint
from PIL import Image


@anvil.server.callable
def classify_image(file):
    rand = random.Random()
    
    with anvil.media.TempFile(file) as filename:
      img = Image.open(io.BytesIO(file.get_bytes()))
      bs = io.BytesIO()
      #img.save(bs, format="JPEG")
      # save image in upload 
      img.save('/content/cell_images/upload/myphoto.png', 'PNG')
      final_path = '/content/cell_images/upload/myphoto.png'
      prediction = "Sick" if getPrediction(CNN, final_path) == 0 else "Healthy"
      # save image for backup      
      #lol = str(randint(1,1000000))
      #backup_path = "/content/cell_images/backup/"+lol+".png"
      #img.save(backup_path, 'JPEG')
      return(prediction)
try:
  os.remove('/content/cell_images/upload/myphoto.png')
except:
  pass