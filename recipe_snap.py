# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import cv2
import numpy as np
import pickle
import os
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from recipe_encoder import * 
from image_encoder import *
from HydraNet import *
from utils import *
#from utils.preprocessing import get_image_loader, get_recipe_loader, recipe_preprocessing 
#from utils.utils import load_checkpoint, count_parameters
import shutil
from torchvision import transforms



class RecipeSnap(object):
    """ a light-weight pretrained model to predict recipe from image

    Parameters
    ----------
    recipe_dict : str
        Path of recipe dictionary file.
    checkpoint_dir : str
        Path of checkpoint folder.
    """

    
    def __init__(self, checkpoint_dir='../checkpoints/model', output_size=1024, image_model='mobilenet_v2'):
        self.checkpoint_dir =  checkpoint_dir
        self.output_size = output_size
        self.image_encoder = ImageEmbedding(output_size=output_size, image_model=image_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Nutritional components prediction
        self.resize_img = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        self.hydra = Hydra()#.to(self.device)
        self.hydra.load_state_dict(torch.load(r'\best_model_4channel_prova2_c.pt',map_location=torch.device('cpu')))
        
        # Recipe prediction for dashboard
        self.preprocess_recipe = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.preprocess_nutrient =  transforms.Compose([transforms.CenterCrop(224),transforms.Normalize(mean=[0.457, 0.532, 0.604, 0.556], std=[0.259, 0.246, 0.233, 0.264])]) #transforms.Resize(256),

        # for depth estimation
        self.midas_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform =  self.midas_transforms.small_transform
        self.midas_small.eval()



    def load_image_encoder(self):
        print(f"Loading checkpoint from ... {self.checkpoint_dir}")
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(self.checkpoint_dir,map_loc=map_loc,suff='best')
        self.image_encoder.load_state_dict(model_dict, strict=False)
        print("Loading checkpoint succeed.")
        print("image encoder", count_parameters(self.image_encoder))

        if self.device != 'cpu' and torch.cuda.device_count() > 1:
            self.image_encoder = torch.nn.DataParallel(self.image_encoder)
        self.image_encoder.to(self.device)
        self.image_encoder.eval()




    def load_recipe_encoder(self):
        print(f"Loading recipe encoder ...")
        self.recipe_encoder = JointEmbedding(output_size=self.output_size, vocab_size=16303)
        map_loc = None if torch.cuda.is_available() else 'cpu'
        model_dict = load_checkpoint(self.checkpoint_dir,map_loc=map_loc,suff='best')       # questo ritorna lo state_dict del modello
        self.recipe_encoder.load_state_dict(model_dict, strict=False)                       # questo prende i parametri per ogni layers
        print("Loading recipe encoder succeed.")
        print("recipe encoder", count_parameters(self.recipe_encoder))

        if self.device != 'cpu' and torch.cuda.device_count() > 1:
            self.recipe_encoder = torch.nn.DataParallel(self.recipe_encoder)
        self.recipe_encoder.to(self.device)
        self.recipe_encoder.eval()




    def compute_image_embedding(self, loader):            
        image_num = len(loader)
        loader = iter(loader)
        img_embeddings = []
        img_names = []
        for _ in range(image_num):
            img, img_name = loader.next()
            img = img.to(self.device)
            with torch.no_grad():
                emb = self.image_encoder(img)                   # qui si computa l'embeddings per ogni immagine
            img_names.extend(img_name)                          # qui si salva il nome dell'immagine
            img_embeddings.append(emb.cpu().detach().numpy())   # qui si salva l'embeddings dell'immagine
        img_embeddings = np.vstack(img_embeddings)              # qui si mettono tutte le embeddings in un unico array in verticale
        return img_embeddings, img_names
    




    def compute_recipe_embedding(self, loader):
        recipe_num = len(loader)
        loader = iter(loader)
        recipe_embeddings = []
        recipe_ids = []
        for _ in range(recipe_num):
            title, ingrs, instrs, ids = loader.next()
            title = title.to(self.device)
            ingrs = ingrs.to(self.device)
            instrs = instrs.to(self.device)
            with torch.no_grad():
                recipe_emb = self.recipe_encoder(title, ingrs, instrs)  # qui si computa l'embeddings per ogni ricetta
            recipe_ids.extend(ids)                                      # qui si salva l'id della ricetta
            recipe_embeddings.append(recipe_emb.cpu().detach().numpy()) # qui si salva l'embeddings della ricetta
        recipe_embeddings = np.vstack(recipe_embeddings)                # qui si mettono tutte le embeddings in un unico array in verticale
        return recipe_embeddings, recipe_ids
        


    def load_image(self, image_dir, batch_size=1, resize=256, im_size=224, augment=True, mode='predict',drop_last=True):
        loader, dataset = get_image_loader(image_dir, resize=resize, im_size=im_size, batch_size=batch_size, 
                                                                augment=augment, mode=mode,drop_last=drop_last)
        print(f"{len(loader)} image loaded")
        return loader, dataset



    def load_recipe(self, recipe_path=None, recipe_dict=None, batch_size=1,drop_last=True):
        loader, dataset = get_recipe_loader(recipe_path=recipe_path, recipe_dict=recipe_dict, batch_size=batch_size, 
                                                drop_last=drop_last)
        print(f"{len(loader)} recipe loaded")
        return loader, dataset

    def load_recipe_lib(self, recipe_emb_path = "../data/recipe_embeddings/recipe_embeddings_feats_test.pkl", 
                    recipe_dict_path="../data/recipe_dict/test.pkl"):
        with open(recipe_emb_path, 'rb') as f:
            self.recipe_embs = pickle.load(f)    # qui si carica il pickle con le embeddings delle ricette
            self.recipe_ids = pickle.load(f)     # qui si carica il pickle con gli id delle ricette
        print("recipe_embs", self.recipe_embs[-2:])
        print("recipe_ids", self.recipe_ids[-2:])
        print(f"Succeed to load recipe embedding from ... {recipe_emb_path}")
        print(f"Recipe embedding (that is a list of vectors) shape: {self.recipe_embs.shape}")
        print(f"Recipe ids (that is a list) length: {len(self.recipe_ids)}")
        # print new line
        print()
        print(f"-------------------------------------------------------")

        with open(recipe_dict_path, 'rb') as f:
            self.recipe_dict = pickle.load(f)
        print(f"Length of recipe dictionary of dish with images: {len(self.recipe_dict)}")
        print()
        print(f"-------------------------------------------------------")

        noimage_file_path = recipe_dict_path[:-4] + "_noimages.pkl"
        if os.path.exists(noimage_file_path):
            with open(noimage_file_path, 'rb') as f:
                self.recipe_dict.update(pickle.load(f))
        print(f"Succeed to load recipe library from ... {recipe_dict_path}")
        print(f"Recipe library size with dish without images {len(self.recipe_dict)}", "recipe library contents: ")
        # print last element of recipe library, key and value
        print(list(self.recipe_dict.items())[-1])



    def predict(self, image_dir,  max_k=1):
        loader, dataset = self.load_image(image_dir)
        img_embs, img_names = self.compute_image_embedding(loader)
        dists = pairwise_distances(img_embs, self.recipe_embs, metric='cosine') 
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k] # retrieve top-k recipes in efficient way
        retrieved_recipes_dict = defaultdict(list)  # è un dizionario flessible, se chiamo un elemento che non c'è lo crea (come lista qua)
        for i, img_name in enumerate(img_names):    # gli passo una cartella di immagini, quindi itera su tutte le immagini
            for rec_id in retrieved_idxs_recs[i]:   # itera su tutte le K ricette trovate per quella immagine
                retrieved_recipes_dict[img_name].append(self.recipe_dict[self.recipe_ids[rec_id]])
                                                        # dentro il dizionario test prendo la ricetta usando
        # Predict the kcal from the image
        nutrients = {}
        
        for idx, image in enumerate(os.listdir(image_dir)):
            dir = image_dir + '/' + image
            img = Image.open(dir)
            img =  self.resize_img(img)[np.newaxis,:,:].to(self.device)
            self.hydra.eval()
            with torch.no_grad():
                kcal_output, carbo_output, protein_output, fat_output, mass_output = self.hydra(img)
            kcal_output = kcal_output.cpu().detach().numpy()
            carbo_output = carbo_output.cpu().detach().numpy()
            protein_output = protein_output.cpu().detach().numpy()
            fat_output = fat_output.cpu().detach().numpy()
            mass_output = mass_output.cpu().detach().numpy()
            nutrients[image] = [kcal_output, carbo_output, protein_output, fat_output, mass_output]

        return img_embs, self.recipe_embs, retrieved_recipes_dict, nutrients



    
    def predict_dashboard(self, image, max_k=5):
        loader, dataset = self.load_image(r'C:\Users\kevin\Desktop\Università\DataScience\Stage\dashboard_image')
        img_embs, img_names = self.compute_image_embedding(loader)
        dists = pairwise_distances(img_embs, self.recipe_embs, metric='cosine') 
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k] # retrieve top-k recipes in efficient way
        retrieved_recipes_dict = defaultdict(list)  # è un dizionario flessible, se chiamo un elemento che non c'è lo crea (come lista qua)
        for i, img_name in enumerate(img_names):    # gli passo una cartella di immagini, quindi itera su tutte le immagini
            for rec_id in retrieved_idxs_recs[i]:   # itera su tutte le K ricette trovate per quella immagine
                retrieved_recipes_dict[img_name].append(self.recipe_dict[self.recipe_ids[rec_id]])
        print(retrieved_recipes_dict)
            
        with torch.no_grad():
            # Predict the Nutrient from the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = self.midas_transform(image)
            depth = self.midas_small(depth)
            depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = depth.cpu().detach().numpy()
            
            image = cv2.resize(image, (256, 256))
            depth = cv2.resize(depth, (256, 256))
                # create a 4 channel imagee, stacking the depth as last channel
            image = torch.from_numpy(np.concatenate((image, depth[:,:,np.newaxis]), axis = 2) ).permute(2, 0, 1) 
            print(image.shape)

            inputs = self.preprocess_nutrient(image)[np.newaxis,:,:,:]
            print(inputs.shape)
            print('predicting nutrients...')
            kcal_output, carbo_output, protein_output, fat_output, mass_output = self.hydra(inputs)
            print('done')
            nutrients = [kcal_output.item(), carbo_output.item(),  protein_output.item(), fat_output.item(), mass_output.item()]
            print(nutrients)
        return nutrients, retrieved_recipes_dict





    def TEST_predict(self, image_dir, key_list, max_k=5):
        loader, dataset = self.load_image(image_dir)
        img_embs, img_names = self.compute_image_embedding(loader)

        # creo la key-list
        key_list = os.listdir(image_dir)

        # Normalization of retrieval vectors
        img_embs_norm = []
        rec_embs_norm = []
        for img_emb in img_embs:
            img_embs_norm.append(img_emb/np.linalg.norm(img_emb))
        for rec_emb in self.recipe_embs:
            rec_embs_norm.append(rec_emb/np.linalg.norm(rec_emb))

        dists = pairwise_distances(img_embs_norm, rec_embs_norm, metric='cosine')
        retrieved_idxs_recs = np.argpartition(dists, range(max_k), axis=-1)[:,:max_k] # retrieve top-k recipes in efficient way
        retrieved_recipes_dict = defaultdict(list)  # è un dizionario flessible, se chiamo un elemento che non c'è lo crea (come lista qua)
        for i, key in enumerate(key_list):    # gli passo una cartella di immagini, quindi itera su tutte le immagini
            for rec_id in retrieved_idxs_recs[i]:   # itera su tutte le K ricette trovate per quella immagine
                retrieved_recipes_dict[key].append(self.recipe_dict[self.recipe_ids[rec_id]])
                                                        # dentro il dizionario test prendo la ricetta usando
        return img_embs, self.recipe_embs, retrieved_recipes_dict




    def update_recipe_lib(self, new_recipes):
        print("Updating recipe lib ...")
        print(f"Before update, there are {len(self.recipe_dict)} recipes in library")
        new_recipe_dict = recipe_preprocessing(new_recipes)
        print(new_recipe_dict)
        loader, dataset = self.load_recipe(recipe_dict = new_recipe_dict)
        new_recipe_embs, new_recipe_ids = self.compute_recipe_embedding(loader)
        self.recipe_embs = np.concatenate((self.recipe_embs, new_recipe_embs))
        self.recipe_ids.extend(new_recipe_ids)
        self.recipe_dict.update(new_recipe_dict)
        print(f"After update, there are {len(self.recipe_dict)} recipes in library")



    def TEST_update_recipe_lib(self, image_dir, key_list, nutrition):
        print("Updating recipe lib with an image embedding ...")
        print(f"Before update, there are {len(self.recipe_dict)} recipes in library")

        loader, dataset = self.load_image(image_dir)
        img_embs, img_names = self.compute_image_embedding(loader)
        img_embs_norm = []
        for img_emb in img_embs:
            img_embs_norm.append(img_emb/np.linalg.norm(img_emb))
        #print("THIS DATA HAS BEEN LOADED: ", img_embs_norm, key_list)

        # Aggiorno sia la lista di embeddings sia la lista di id
        self.recipe_embs = np.concatenate((self.recipe_embs, img_embs_norm)) # metto l'embedding dell'im negli embeddings delle ricette
        self.recipe_ids.extend(key_list)    # metto il nome dell'immagine dentro gli id delle ricette (che sono tutti hash)
        self.recipe_dict.update(nutrition)

        print(f"the new recipe dict is: {nutrition}")
        # print last 5 element of dictionary
        print("LAST 2 ELEMENTS OF RECIPE DICT: ", list(self.recipe_dict.items())[-2:])
        print("LAST 2 ELEMENTS OF RECIPE IDS: ", self.recipe_ids[-2:])
        print(f"After update, there are {len(self.recipe_dict)} recipes in library and {len(self.recipe_ids)} ids and {len(self.recipe_embs)} embeddings")



    def TEST_STACK_update_recipe_lib(self, image_dir, key_list, nutrition):
        print("Updating recipe lib with an image embedding ...")
        print(f"Before update, there are {len(self.recipe_dict)} recipes in library")

        img_embs_norm = []
        list_dir = os.listdir(image_dir)
        for idx, dish in enumerate(list_dir):
            dish_folder = os.path.join(image_dir, dish)
            loader, dataset = self.load_image(dish_folder)
            img_embs, img_names = self.compute_image_embedding(loader)
            centroid = np.mean(img_embs, axis=0)
            img_embs_norm.append(centroid/np.linalg.norm(centroid))
        

        # Aggiorno sia la lista di embeddings sia la lista di id
        self.recipe_embs = np.concatenate((self.recipe_embs, img_embs_norm)) # metto l'embedding dell'im negli embeddings delle ricette
        self.recipe_ids.extend(key_list)    # metto il nome dell'immagine dentro gli id delle ricette (che sono tutti hash)
        self.recipe_dict.update(nutrition)

        print(f"the new recipe dict is: {nutrition}")
        # print last 5 element of dictionary
        print("LAST 2 ELEMENTS OF RECIPE DICT: ", list(self.recipe_dict.items())[-2:])
        print("LAST 2 ELEMENTS OF RECIPE IDS: ", self.recipe_ids[-2:])
        print(f"After update, there are {len(self.recipe_dict)} recipes in library and {len(self.recipe_ids)} ids and {len(self.recipe_embs)} embeddings")






    def save_recipe_lib(self, new_recipe_emb_path, new_recipe_dict_path):
        with open(new_recipe_emb_path, 'wb') as f:
            pickle.dump(self.recipe_embs, f)
            pickle.dump(self.recipe_ids, f)

        with open(new_recipe_dict_path, 'wb') as f:
            pickle.dump(self.recipe_dict, f)
        

    def get_recipe(self, num_recipe):
        # return recipe items from 0 to num_recipe of self.recipe_dict
        return list(self.recipe_dict.items())[:num_recipe]

    def get_recipe_ids(self, key):
        # return true if key is in self.recipe_ids
        return key in self.recipe_ids
