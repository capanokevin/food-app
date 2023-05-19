# Server App Deployment Guide

This guide provides instructions for deploying the server app on your own machine. Follow the steps below to set up the necessary environment and run the application.

## Prerequisites

Before proceeding with the deployment, ensure that you have the following prerequisites installed:

1. Python 3.7 or later
2. pip package manager
3. Git

## Installation

1. Clone the repository by running the following command in your terminal or command prompt:
   ```
   git clone https://github.com/capanokevin/food-app.git
   ```

2. Change your current directory to the cloned repository:
   ```
   cd food-app
   ```

3. Install the required Python packages by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open the Python script file named `food.py` in a text editor.

2. Locate the following lines in the script:

   ```python
   line 27: image_dir = '\images'
   line 28: checkpoint_dir = "\checkpoints\model"
   line 29: recipe_emb_path = "\data\recipe_embeddings\recipe_embeddings_feats_test.pkl" 
   line 30: recipe_dict_path = "\data\recipe_dict\test.pkl"
   
   line 133: '..\dashboard_image\dashboard_image.jpg'
   line 149: '..\collected_images\'
   ```

   Replace these paths with the appropriate paths on your own machine where the required files and directories are located.
   
   Open the Python script file named `recipe_snap.py` and go to line 205. Replace the following paths:
   ```python
   line 205: "..\dashboard_image"
   line 42:  "best_model_4channel_prova2_c.pt"
   ```
   with the appropriate paths on your own machine, where the folder and model weights are located.

3. Save the changes to the script.

4. Run the server app by executing the following command in the terminal or command prompt:
   ```
   streamlit run 'path where food.py is stored'
   ```

5. The server app should now be running locally on your machine.

6. Use the file uploader or camera input to upload or take a picture of an image containing food.

7. The app will process the image and display the uploaded image along with the predicted recipe and nutritional facts.

8. Click on the recipe buttons to view the ingredients and procedure for each recipe.

9. The nutritional facts will be displayed, showing the estimated values for calories, carbohydrates, proteins, fats, and mass (in grams) based on the uploaded image.

10. Enjoy using the server app to analyze food images and explore recipe and nutritional information!

Note: Make sure to keep the terminal or command prompt open while running the server app. If you want to stop the app, press `Ctrl + C` in the terminal or command prompt.
