import base64
import json
import os
import re
import time
import uuid
import shutil
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import cv2 as cv
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

import SimpleConvoModel
import torch
from torch import nn, optim
from torchvision import transforms

# linearNet = LinearpyTeen.PyTeen()
ConvoNet = SimpleConvoModel.PyTeen()

# function main contains the code for the main page
def main(canvas_result):
    if st.button(label="AI Guess \U0001F914", type="primary"):
        ConvoNet.load_state_dict(torch.load('pyTeenConvo9882.pth'))
        cv.imwrite("canvas.png", canvas_result.image_data)

        img = cv.imread("canvas.png", cv.IMREAD_GRAYSCALE)
        scaling = cv.resize(img, (28, 28))
        my_img_processed = cv.bitwise_not(scaling)
        cv.imwrite('new.png', my_img_processed)

        # Processing the image for model prediction
        input_img = SimpleConvoModel.my_transform(my_img_processed).unsqueeze(0)
        guessed_digit = ConvoNet.predict(input_img)
        ans = str(guessed_digit.item())
        st.subheader(f"The AI guessed that it was a {ans}\n")

        user_feedback = st.checkbox("AI Guess Incorrect?")
        if user_feedback:
            # Create a folder for incorrect guesses if it doesn't exist
            incorrect_folder = 'incorrect_guesses'
            os.makedirs(incorrect_folder, exist_ok=True)

            # Generate a unique folder name using timestamp
            folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_path = os.path.join(incorrect_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Save canvas.png and new.png in the created folder
            shutil.copy("canvas.png", os.path.join(folder_path, "canvas.png"))
            shutil.copy("new.png", os.path.join(folder_path, "new.png"))

            st.success("Feedback received, thank you!")

# full_app function corresponds to the canvas found in a public github repo
def full_app():
    st.sidebar.title("Configuration")
    st.markdown(
        """
    Draw a number from **0 to 9** and click on the guess button!
    * Go easy at first, then try to fool it
    * We also appreciate feedback! :)
    """
    )

    
    # Specify canvas parameters in application
    drawing_mode = "freedraw" # limit user to one option
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=18, # implicitly setting to 18
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True, # update just sends the canvas image to streamlit
        height=400,
        width=400,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )
<<<<<<< HEAD
    return canvas_result
=======

    if st.button(label="AI Guess \U0001F914", type = "primary"):
        ConvoNet.load_state_dict(torch.load('pyTeenConvo9882.pth'))
        # st.image(canvas_result.image_data)
        cv.imwrite("canvas.png",  canvas_result.image_data)

        # this section of the code does the image manipulation
        # before passing it to our NN for guessing
        img = cv.imread("canvas.png",cv.IMREAD_GRAYSCALE)
        scaling = cv.resize(img,(28,28))
        my_img_processed = cv.bitwise_not(scaling) # inverting image (background needs to be black)
        cv.imwrite('new.png',my_img_processed)
        new_img = Image.open("new.png")
        # st.image(new_img)
        # my_img_transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
        input_img = SimpleConvoModel.my_transform(my_img_processed).unsqueeze(0)
        guessed_digit = ConvoNet.predict(input_img)
        ans = str(guessed_digit.item())
        st.subheader(f"The AI guessed that it was a {ans}\n")
>>>>>>> upstream/main


    st.markdown(""" 

**Was the AI spot on?**
(Uncheck the box after you give the feedback)
                     
                """)
    
    user_feedback_Y = st.checkbox("Hell Yeah!",key="checkYes")
    user_feedback_N = st.checkbox("It still needs work!",key="checkNO")

    if user_feedback_Y:
        st.balloons()


if __name__ == "__main__":

    st.set_page_config(
        page_title="Guessing Number Demo", page_icon=":pencil2:",
        initial_sidebar_state="collapsed"
    )
    st.title("AI trying its best to guess a number!")
    canvas_result = full_app()
    main(canvas_result)