# project-4_group-3
# News Verifier Flask Application

This Flask application classifies news headlines as real or fake, and also checks if the content of a webpage is true or false. The project includes two primary components:

- **appv2.py**: The main Flask application file (located in the project folder).
- **templatesv2/index.html**: The HTML template used to render the frontend.

## How to Run the Project

1. **Open Terminal**
   - On Windows, you can use Git Bash (or any other preferred terminal) and navigate to the project folder.

2. **Start the Application**
   - Type the following command in your terminal:
     ```
     python appv2.py
     ```
   - This will start the Flask server. You should see output similar to:
     ```
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
     ```

3. **Access the Application**
   - Open your web browser and enter the IP address provided by the terminal (usually http://127.0.0.1:5000/).

## Project Structure

```
├── appv2.py                   # Main Flask application
└── templatesv2
    └── index.html             # HTML template
```

## Models

Two pretrained models are used:

- **news_fake_real_model.pkl**: Model to predict if a news headline is real or fake.
- **webpage_truth_model.pkl**: Model to assess if the content of a webpage is true or false.

Make sure these model files are in the project directory before running the application.

## Usage

- **News Verifier**: The application allows you to enter a news headline or webpage content and then shows the prediction (Real/Fake for headlines, True/False for webpage) along with the confidence percentage and extra context.

## Troubleshooting

- Ensure all required model files are present in the project directory.
- If the Flask server does not start, check that you're running the command in the correct folder and that Python is installed and accessible from your terminal.

## License

MIT

