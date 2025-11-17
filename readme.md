#  Car Image Analysis Streamlit App

This Streamlit application allows users to upload car images, extract key vehicle information (Type, License Plate, Make, Model, Color) using a multimodal AI model, display the collected data in an editable table, and view an analytics dashboard.

##  Features

-   **Image Upload**: Easily upload car images for analysis.
-   **AI-Powered Data Extraction**: Utilizes `gemini-2.5-flash` (via LangChain) to identify vehicle type, license plate, make, model, and color.
-   **Editable Data Table**: View and modify the extracted car data in an interactive DataFrame.
-   **Analytics Dashboard**: Once three or more car entries are collected, an analytics dashboard provides insights into:
    -   Total cars analyzed.
    -   Distribution of car colors.
    -   Presence/absence of license plates.

## Installation

1.  **Clone the repository (or save the files)**: Ensure `requirements.txt` and `app.py` are in the same directory.

2.  **Install dependencies**: Navigate to the project directory in your terminal and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

##  Google API Key Setup

The application requires a Google API Key to access the Gemini model. Follow these steps to set it up:

1.  **Obtain an API Key**: If you don't have one, create a key in [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  **Set as Environment Variable**: Before running the app, set your Google API Key as an environment variable in your terminal:

    ```bash
    export GOOGLE_API_KEY='YOUR_API_KEY'
    ```
    Replace `'YOUR_API_KEY'` with your actual key.

    *   **For Streamlit Cloud Deployment**: If you plan to deploy on Streamlit Cloud, you would add your `GOOGLE_API_KEY` as a secret in the Streamlit Cloud settings rather than an environment variable.

##  How to Run the Application

1.  **Ensure dependencies are installed** and your **API key is set**.
2.  **Run the Streamlit app** from your terminal in the project directory:

    ```bash
    streamlit run app.py
    ```

3.  **Access the application**: Streamlit will launch a new tab in your web browser with the application. If it doesn't open automatically, you can navigate to the URL displayed in your terminal (usually `http://localhost:8501`).

##  Technologies Used

-   [Streamlit](https://streamlit.io/)
-   [LangChain](https://www.langchain.com/)
-   [Google Gemini API](https://ai.google.dev/)
-   [Pydantic](https://docs.pydantic.dev/)
-   [Pandas](https://pandas.pydata.org/)
-   [Matplotlib](https://matplotlib.org/)
-   [Pillow](https://python-pillow.org/)
