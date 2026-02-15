import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64

st.title("CGAN Image Generator")

# Load the CGAN model (which contains the generator) from the .pkl file
with open('cgan_model.pkl', 'rb') as file:
    loaded_cgan_model = pickle.load(file)

# Extract the generator model from the loaded CGAN model
generator = loaded_cgan_model.get_layer('generator')

st.write("Generator model loaded successfully!")

latent_dim = 100

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.sidebar.header("Image Generation Controls")
selected_label = st.sidebar.selectbox(
    "Select a class to generate:",
    options=list(range(len(class_names))),
    format_func=lambda x: class_names[x]
)

# Consolidate all image generation and display logic within a single button block
if st.sidebar.button("Generate Image", key='generate_button'):
    st.write(f"Generating image for class: {class_names[selected_label]}")
    # Generate random noise
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Prepare label for the generator
    label_input = np.array([selected_label])

    # Generate image
    generated_image = generator.predict([noise, label_input], verbose=0)[0]

    # Rescale image from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) / 2.0

    st.image(generated_image, caption=f"Generated {class_names[selected_label]} Image", use_column_width=True)
    st.success(f"Successfully generated an image for class: {class_names[selected_label]}")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("data:image/jpeg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """ % base64.b64encode(open("/content/background.jpg", "rb").read()).decode(),
    unsafe_allow_html=True
)
