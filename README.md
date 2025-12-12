# Student Performance Prediction App

This is a Streamlit application that predicts student exam scores based on various factors like study hours, sleep, attendance, and previous scores.

## Prerequisites

1.  **Python**: Ensure you have Python installed (recommended version 3.8+).
2.  **VS Code**: (Optional) Recommended code editor.

## Setup Instructions

1.  **Unzip the project** to a folder.
2.  Open a terminal/command prompt in that folder.
3.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    ```
4.  **Activate the virtual environment**:
    *   Windows: `.\venv\Scripts\activate`
    *   Mac/Linux: `source venv/bin/activate`
5.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Run the following command in the terminal:
```bash
streamlit run app.py
```

## Features

*   **Student Login**: Students can view their own predictions and risk levels.
*   **Admin Login**: Admins (User: `admin`, Pass: `admin123`) can enter new student data and train the model.
*   **Prediction**: Predicts Exam Score and categorizes into High/Medium/Low Risk.

## File Structure

*   `app.py`: The main application code.
*   `model_training.py`: Script to retrain the machine learning model.
*   `StudentPerformanceFactors.csv`: The dataset used for training.
*   `student_database.csv`: Stores the predicted student records.
*   `*.pkl`: Saved model and metadata files.
