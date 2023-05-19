import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import streamlit as st
import plotly.express as px
from PIL import Image


from recipe_snap import *
from HydraNet import * 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import shutil
from tqdm import tqdm
import cv2
import random 
import warnings
warnings.filterwarnings("ignore")
import contextlib




# for recipe-nutrient estimation
image_dir = r'C:\Users\kevin\Desktop\Università\DataScience\Stage\RecipeSnap-a-lightweight-image-to-recipe-model-master\images'
checkpoint_dir = r"C:\Users\kevin\Desktop\Università\DataScience\Stage\RecipeSnap-a-lightweight-image-to-recipe-model-master\checkpoints\model"
recipe_emb_path = r"C:\Users\kevin\Desktop\Università\DataScience\Stage\RecipeSnap-a-lightweight-image-to-recipe-model-master\data\recipe_embeddings\recipe_embeddings_feats_test.pkl" 
recipe_dict_path = r"C:\Users\kevin\Desktop\Università\DataScience\Stage\RecipeSnap-a-lightweight-image-to-recipe-model-master\data\recipe_dict\test.pkl"
#hydra_weights_path = r"/datasets/data5/recipe_snap/RecipeSnap-a-lightweight-image-to-recipe-model-master/Calories Estimation/best_model.pt"
rs = RecipeSnap(checkpoint_dir=checkpoint_dir)
rs.load_image_encoder()
rs.load_recipe_lib(recipe_emb_path = recipe_emb_path, recipe_dict_path = recipe_dict_path)



def predict_vars(img, food_rda):
	# estimate recipe
	food_vars, recipe = rs.predict_dashboard(image=img)
	food_vars = torch.tensor(food_vars).float()

	# return recipe and food variables
	return food_vars, recipe


def click(choice):
	if choice is None:
			choice = 0

	with col2:
		title = recipe['dashboard_image.jpg'][choice]['title']
		st.title(title)
		col2_1, col2_2 = st.columns([1,1])
					# view recipe
					# st.text(recipe)
		with col2_1:
			st.header('Ingredients')
			for i in range(len(recipe['dashboard_image.jpg'][choice]['ingredients'])):
				st.markdown(recipe['dashboard_image.jpg'][choice]['ingredients'][i])
		with col2_2:
						# view procedure
			st.header('Procedure')
			procedure = recipe['dashboard_image.jpg'][choice]['instructions']
			for step in procedure:
				st.markdown(step)

	with col3:
				#if img_pil is not None:
		st.header('Nutritional Facts')
				# create metric objects
		for cur_var, cur_score, cur_delta in zip(food_var_names, food_vars, delta):
			st.metric(cur_var, f'{cur_score:.2f}', delta=f'{cur_delta*100:.2f}% RDA', delta_color='normal' if cur_delta<0.5 else 'inverse')
	
	return None 

	

def buttons():
	choice = None
	with contextlib.suppress(Exception):

		first = st.button(recipe['dashboard_image.jpg'][0]['title'], key = 0)
		second = st.button(recipe['dashboard_image.jpg'][1]['title'], key = 1)
		third = st.button(recipe['dashboard_image.jpg'][2]['title'], key = 2)
		fourth = st.button(recipe['dashboard_image.jpg'][3]['title'], key = 3)
		fifth =  st.button(recipe['dashboard_image.jpg'][4]['title'], key =  4)

		if first:
			choice = 0
		elif second:
			choice = 1
		elif third:
			choice = 2
		elif fourth:
			choice = 3
		elif fifth:
			choice = 4

	if choice is not None:
		return choice




if __name__ == '__main__':
	# set page width
	st.set_page_config(layout="wide")
	# define variables
	food_var_names = ['Kcal', 'Carbohydrates ', 'Proteins', 'Fats', 'Mass (gr.)']
	food_rda = [2000, 300, 50, 80, 100]
	# convert to torch
	food_rda = torch.tensor(food_rda).float()
	# initialize
	uploaded_file = None
	img_tensor = None
	
	# define a layout with three columns
	col1, col2, col3 = st.columns([2,3.2,0.8])

	with col1:
		# create file uploader
		
		uploaded_file = st.file_uploader("Upload or Take a picture of an image containing food!", accept_multiple_files=False, key = 'file_uploader')
		#smartphone = st.camera_input('Take a picture with your smartphone!', key = 'smartphone')
		#if smartphone is not None:
		#	uploaded_file = smartphone
		#else: 
		#	pass
		
		if uploaded_file is not None:
					# load image
				img_pil = Image.open(uploaded_file)
					# convert to tensor
				img_tensor_torch = transforms.ToTensor()(img_pil)
				img_tensor_np = np.array(img_pil)
					# show image
				st.image(img_pil, caption=f'Uploaded Image.', use_column_width=True)
					# save image in folder 'C:\Users\kevin\Desktop\Università\DataScience\Stage\dashboard_image'
				img_pil.save(r'C:\Users\kevin\Desktop\Università\DataScience\Stage\dashboard_image\dashboard_image.jpg')
					# predict variables
				food_vars, recipe = predict_vars(img_tensor_np, food_rda)
					# statistics: 'calories_100', 'total_mass', 'fat_100', 'carb_100', 'protein_100'
					#  means :     239.26081883, 443.72362135, 13.40239114, 23.72410244, 5.85989564
					# scales:      171.1492288 , 278.38775641,  17.03006593,  19.64341184, 5.91145122
					# de-scales
				food_vars = food_vars * torch.tensor([171.1492288 , 19.64341184,  5.91145122, 17.03006593, 278.38775641]).float()
					# add mean
				food_vars = food_vars + torch.tensor([239.26081883, 23.72410244, 5.85989564, 13.40239114, 443.72362135]).float()
					# compute delta with rda
				delta = food_vars/food_rda
				image_name = recipe['dashboard_image.jpg'][0]['title'] + '.jpg'
				# replace spaces with underscores
				image_name = image_name.replace(' ', '_')
				# save the image in folder 'C:\Users\kevin\Desktop\Università\DataScience\Stage\collected_images'
				img_pil.save(r'C:\Users\kevin\Desktop\Università\DataScience\Stage\collected_images\{}'.format(image_name))
			


				choice = buttons()
				mem = choice
				
				if choice is None:
					click(0)

				if choice is not None:
					click(choice)
					choice = buttons()