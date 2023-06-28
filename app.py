import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
# Load the pre-trained model
model = load_model('handwritten4.model')  # Replace 'model.h5' with the path to your saved model file

# Define the Streamlit app
def main():
    st.title("Digit Recognition")
    st.write("Draw a digit and let the model recognize it!")

    # Create a canvas using st_canvas
    canvas = st_canvas(
        fill_color="#FFFFFF",  # Black color for drawing
        stroke_width=20,
        stroke_color="#000000",  # White color for stroke
        background_color="#FFFFFF",  # Black color for background
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Function to preprocess the drawn digit
    def preprocess_image(image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = np.array([img])
        img = img.reshape((1, 28, 28, 1))
        return img

    # Function to recognize the drawn digit
    def recognize_digit(image):
        img = preprocess_image(image)
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        return digit

    # Recognize the digit when the user clicks the "Recognize" button
    if st.button("Recognize"):
        # Get the drawn image from the canvas
        drawn_image = canvas.image_data.astype("uint8")

        # Preprocess and recognize the digit
        digit = recognize_digit(drawn_image)

        # Display the recognized digit
        st.write(f"The drawn digit is: {digit}")

if __name__ == "__main__":
    main()
