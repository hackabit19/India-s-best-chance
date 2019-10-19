from anvil import *
import anvil.server

class Form1(Form1Template):

  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.

  def file_loader_1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    result = anvil.server.call('classify_image', file)
    self.result_lbl.text = "%s"%(result)
    self.image_1.source = file

  def result_lbl_show(self, **event_args):
    """This method is called when the Label is shown on the screen"""
    pass



