# PromptShield 🛡️

PromptShield is a real-time, 4-class classification system using a fine-tuned DistilBERT model to secure technical Large Language Models (LLMs) against adversarial prompts. It categorizes incoming prompts into `safe`, `unsafe`, `suspicious`, or `jailbreak` classes, enabling a nuanced, tiered security response.

## Description

Large Language Models (LLMs) are increasingly used as technical co-pilots for tasks like code generation and infrastructure management. However, standard binary safety filters (safe/unsafe) are often insufficient, failing to distinguish between benign developer queries and sophisticated adversarial attacks like reconnaissance (`suspicious`) or direct persona manipulation (`jailbreak`). PromptShield addresses this gap by providing a real-time middleware that classifies prompts into four distinct categories, enabling a more intelligent and granular security response (Allow, Warn, Block).

## Features

* **4-Class Prompt Classification:** Accurately categorizes prompts into `safe`, `unsafe`, `suspicious`, or `jailbreak`.
* **Real-time Filtering:** Acts as a fast middleware layer before the LLM.
* **Tiered Security Response:** Allows safe queries, warns users of suspicious activity, and blocks unsafe/jailbreak attempts.
* **Efficient Model:** Built using DistilBERT for an optimal balance of speed and accuracy.
* **Domain-Specific Training:** Trained on a novel, large-scale, programmatically generated dataset focused on technical/developer prompts.

## Getting Started

### Prerequisites

* Python (version used, e.g., 3.10+)
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Pradeep-2901/PromptShield.git](https://github.com/Pradeep-2901/PromptShield.git)
    cd PromptShield
    ```
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On Mac/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    *(You may need to create a `requirements.txt` file first using `pip freeze > requirements.txt` if your teammate didn't include one)*
    ```bash
    pip install -r requirements.txt
    ```
    *(If no `requirements.txt` exists, install manually: `pip install streamlit requests transformers torch python-dotenv`)*
4.  **Set up Gemini API Key:**
    * Create a `.env` file in the root `PromptShield` directory.
    * Add the following line to the `.env` file, replacing `YOUR_API_KEY` with your actual key:
        ```env
        GEMINI_API_KEY="YOUR_API_KEY"
        ```
    * The `app.py` script will automatically load this key. Ensure `.env` is listed in your `.gitignore` file.

### Running the App

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application from the root `PromptShield` directory:
    ```bash
    python -m streamlit run app.py
    ```
3.  The application should open in your default web browser.

## Model & Dataset

* **Model:** Fine-tuned `distilbert-base-uncased`. Model files are located in the `promptshield_model/` directory but are not tracked by Git due to size limits. (You would typically download these separately or use Git LFS if needed for deployment).
* **Dataset:** The `prompts.csv` file contains the dataset used for training and evaluation, consisting of ~12,000 prompts across 4 balanced categories.

## Results

The model achieved an overall accuracy of **97.89%** on the validation set, with high F1-scores across all categories:
* `safe`: 0.98
* `unsafe`: 0.99
* `jailbreak`: 0.99
* `suspicious`: 0.95

The slight confusion between `safe` and `suspicious` highlights the necessity of the 4-class model and the "Warn & Log" approach for ambiguous prompts.

## Team Members

* Pradeep Behera
* T Aadhithya
* Rakesh Kumar S
