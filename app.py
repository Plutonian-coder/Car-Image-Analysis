import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    return {"image": base64.b64encode(image_bytes).decode("utf-8")}

# Define the prompt chain
@chain
def prompt_chain(inputs):
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
    model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
        google_api_key=os.environ.get('GOOGLE_API_KEY') # Use st.secrets in Streamlit Cloud
    )
    output = model.invoke(inputs)
    return output.content

# Build the processing pipeline
pipeline = image_encoding | prompt_chain | MLLM_response | parser

st.title("🚗 Car Image Analysis App")
st.write("Upload car images to extract vehicle type, license plate, make, model, and color.")

# Session state for storing car data
if 'car_data' not in st.session_state:
    st.session_state.car_data = pd.DataFrame(columns=["Type", "License", "Make", "Model", "Color", "Image"])

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Extracting information...")

    try:
        # Invoke the pipeline
        output = pipeline.invoke(image_bytes)

        # Add 'Image' field to store the uploaded image bytes for later display if needed
        output['Image'] = image_bytes # Store bytes or path if you want to display it later

        # Add the extracted data to the session state DataFrame
        new_car_df = pd.DataFrame([output])
        st.session_state.car_data = pd.concat([st.session_state.car_data, new_car_df], ignore_index=True)
        st.success("Information extracted successfully!")

    except Exception as e:
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
        fig_color, ax_color = plt.subplots(figsize=(8, 6))
        color_counts.plot(kind='bar', ax=ax_color, color='skyblue')
        ax_color.set_title('Distribution of Car Colors')
        ax_color.set_xlabel('Color')
        ax_color.set_ylabel('Number of Cars')
        plt.xticks(rotation=45)
        st.pyplot(fig_color)

        # License Plate Presence
        st.subheader("License Plate Presence")
        license_present = st.session_state.car_data['License'].apply(lambda x: x is not None and x != '' and x != 'None').sum()
        license_absent = len(st.session_state.car_data) - license_present
        license_data = pd.DataFrame({
            'Category': ['License Present', 'License Absent'],
            'Count': [license_present, license_absent]
        })
        fig_license, ax_license = plt.subplots(figsize=(6, 6))
        ax_license.pie(license_data['Count'], labels=license_data['Category'], autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        ax_license.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_license.set_title('License Plate Detection Status')
        st.pyplot(fig_license)

else:
    st.info("Upload images to start collecting car data and view analytics.")

