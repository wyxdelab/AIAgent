import sys
import os
from ui.gradio_app import create_gardio_ui
from ui.css import custom_css
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    demo = create_gardio_ui()
    demo.launch()