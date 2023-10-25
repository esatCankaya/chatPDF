# chatPDF

ChatPDF is a Python-based application that allows you to extract text from PDF documents and perform various operations on the extracted text using natural language processing techniques. It provides a chat interface where you can interact with the extracted text and perform tasks such as searching for specific keywords, summarizing the content, and generating insights.

## Features
- Extract text from PDF documents
- Perform keyword search within the extracted text
- Summarize the content of the PDF document
- Generate insights and analysis based on the extracted text
- User-friendly chat interface for easy interaction

## Technologies Used
List of the technologies and libraries used in the project
- Python, LangChain, OpenAI, FAISS, Streamlit

## Getting Started
  To use this chatbot you should do the followings:
  
  1. Set Up Your Development Environment:
    - Make sure you have Python installed on your system. You can download it from the official Python website.
    - Create a virtual environment to manage project-specific dependencies. You can use venv or conda for this purpose.
  2. Install Required Dependencies
    - Install the necessary Python libraries and packages by running the following command in your project directory. This command will use the dependencies listed in         the requirements.txt file:
       `pip install -r requirements.txt`
  3. Creating environmental variables
    - You'll need an OpenAI API key to use their services. Obtain an API key from the OpenAI platform and save it in an .env file in the project directory.
    - Create an .env file and add your OpenAI API key like this:
       `OPENAI_API_KEY=your_api_key_here`
  3. Create your knowladge base
    - Use the db_creator.py script to create a vector store from your documents.
    - Customize the script to point to the location of your documents and set other relevant parameters. Then run the script to generate the vector store.
  4. Using app.py
    - First you should change store_name variable according to your vectorstore name.
    - Now you can change the query inputs to your like and run the app.
  5. Using streamlit_app.py
    - If you want to provide a graphical user interface (GUI) for your chatPDF, you can work on the streamlit_example.py file. Customize this file as needed to create the user interface.
    - Now you can run the app using:
       `streamlit run ./streamlit_app.py`
