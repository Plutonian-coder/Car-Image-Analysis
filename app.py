import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import os
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain


# Set the Google API key from user data: This will be set in Streamlit Cloud secrets.
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

print("GOOGLE_API_KEY is set:", 'GOOGLE_API_KEY' in os.environ)

# Define the Pydantic model for vehicle information
class Vehicle(BaseModel):
    Type: str = Field(
        ...,
        examples=["Car", "Truck", "Motorcycle", 'Bus', 'Van'],
        description="The type of the vehicle.",
    )
    License: str = Field(
        ...,
        description="The license plate number of the vehicle. A continuous sequence of characters without dots, dashes, or spaces.",
    )
    Make: str = Field(
        ...,
        examples=["Toyota", "Honda", "Ford", "Suzuki"],
        description="The Make of the vehicle.",
    )
    Model: str = Field(
        ...,
        examples=["Corolla", "Civic", "F-150"],
        description="The Model of the vehicle.",
    )
    Color: str = Field(
        ...,
        examples=["Red", "Blue", "Black", "White"],
        description="Return the color of the vehicle.",
    )

# Initialize the JsonOutputParser
parser = JsonOutputParser(pydantic_object=Vehicle)
instructions = parser.get_format_instructions()

# Define the image encoding function
def image_encoding(image_bytes):
    print("Encoding image, original bytes length:", len(image_bytes))
    return {"image": base64.b64encode(image_bytes).decode("utf-8")}

# Define the prompt chain
@chain
def prompt_chain(inputs):
    print("Prompt chain invoked, inputs keys:", list(inputs.keys()))
    prompt = [
        SystemMessage(content="""You are an AI assistant whose job is to inspect an image and provide the desired information from the image. If the desired field is not clear or not well detected, return None for this field. Do not try to guess."""),
        HumanMessage(
            content=[
                {"type": "text", "text": """Examine the main vehicle type, license plate number, make, model and color."""},
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}", "detail": "low"}}]
        )
    ]
    return prompt

# Define the MLLM response function
@chain
def MLLM_response(inputs):
    print("MLLM response invoked")
    model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
        api_key='AIzaSyDKG_J-TMbaQ42C2GJVJxLmtQHlrF_ssqc' # Use st.secrets in Streamlit Cloud
    )
    print("Invoking Gemini model...")
    output = model.invoke(inputs)
    print("Model response received, content length:", len(output.content))
    print("Model response preview:", output.content[:100])
    return output.content

# Build the processing pipeline
pipeline = image_encoding | prompt_chain | MLLM_response | parser

st.title("Car Image Analysis App")
st.write("Upload or capture car images to extract vehicle type, license plate, make, model, and color.")

# Session state for storing car data
if 'car_data' not in st.session_state:
    st.session_state.car_data = pd.DataFrame(columns=["Type", "License", "Make", "Model", "Color", "Image"])

# Input method selection
input_method = st.radio("Choose input method:", ("Upload Image", "Take Photo"))

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])
elif input_method == "Take Photo":
    st.write("Note: Camera access requires running the app in a web browser with camera permissions enabled.")
    uploaded_file = st.camera_input("Take a photo of the car")

if uploaded_file is not None:
    print("Uploaded file detected, processing...")
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Extracting information...")

    try:
        print("Starting pipeline invoke")
        # Invoke the pipeline
        output = pipeline.invoke(image_bytes)
        print("Pipeline output:", output)

        # Add 'Image' field to store the uploaded image bytes for later display if needed
        output['Image'] = image_bytes # Store bytes or path if you want to display it later

        # Add the extracted data to the session state DataFrame
        new_car_df = pd.DataFrame([output])
        st.session_state.car_data = pd.concat([st.session_state.car_data, new_car_df], ignore_index=True)
        st.success("Information extracted successfully!")

    except Exception as e:
        print("Exception during extraction:", e)
        st.error(f"Error extracting information: {e}")


st.header("Collected Car Data")

if not st.session_state.car_data.empty:
    # Display editable DataFrame
    edited_df = st.data_editor(
        st.session_state.car_data.drop(columns=['Image'], errors='ignore'),
        num_rows="dynamic",
        key="car_data_editor"
    )
    st.session_state.car_data = edited_df

    if len(st.session_state.car_data) >= 3:
        st.header("Analytics Dashboard")

        # Total Cars
        st.metric("Total Cars Analyzed", len(st.session_state.car_data))

        # Color Distribution
        st.subheader("Color Distribution")
        color_counts = st.session_state.car_data['Color'].value_counts()
        fig_color = px.bar(color_counts, x=color_counts.index, y=color_counts.values, title='Distribution of Car Colors', color=color_counts.index)
        st.plotly_chart(fig_color)

        # License Plate Presence
        st.subheader("License Plate Presence")
        license_present = st.session_state.car_data['License'].apply(lambda x: x is not None and x != '' and x != 'None').sum()
        license_absent = len(st.session_state.car_data) - license_present
        license_data = pd.DataFrame({
            'Category': ['License Present', 'License Absent'],
            'Count': [license_present, license_absent]
        })
        fig_license = px.pie(license_data, values='Count', names='Category', title='License Plate Detection Status')
        st.plotly_chart(fig_license)

        # Type vs Color Heatmap
        st.subheader("Type vs Color Heatmap")
        type_color = pd.crosstab(st.session_state.car_data['Type'], st.session_state.car_data['Color'])
        fig_heatmap = px.imshow(type_color, text_auto=True, title='Vehicle Type vs Color Distribution')
        st.plotly_chart(fig_heatmap)

else:
    st.info("Upload images to start collecting car data and view analytics.")

