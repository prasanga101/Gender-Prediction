# Image Classifier

This model captures an image from your device camera and can predict gender.

## How to Run the Model?

To run the model, follow these steps:

1. **Create a Virtual Environment:**
   - Set up a virtual environment using a tool like `virtualenv` or `conda` to isolate project dependencies.

2. **Prepare the Dataset:**
   - Create two folders named `Training` and `Validation`.
   - Inside each folder, create two subfolders named `male` and `female`.
   - Upload your dataset into these subfolders.

3. **Install Dependencies:**
   - Install the required Python dependencies by running:
     ```
     pip install -r requirements.txt
     ```

4. **Run the Application:**
   - In the terminal, execute the following command:
     ```
     python app.py
     ```

## Additional Notes:

- **Dataset:** Make sure your dataset is well-prepared and organized within the `Training` and `Validation` folders as described above.
- **Model Training:** If you need to train the model, ensure your dataset is properly labeled and split into training and validation sets before running the training script.

Feel free to reach out if you have any questions or encounter any issues!
