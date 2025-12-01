
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

import timm

import torch.nn.functional as F
import os
import time
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
        
class SimilarityChecker():
    def __init__(self, angles=[0, 45, 90, 135, 180, 225, 270],
                 landmark_images_path=[],
                 kalman_gain = 0.65,
                 similarity_max_windows_size = 35, 
                 score_bias = 0.4, score_gain = 3.,   
                 rbf_sigma = 7.0, rbf_gain = 2.0,
                 device='cpu') -> None:
        
        # Define the standard preprocessing pipeline for DINO models
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale and keep 3 channels
            transforms.Lambda(lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE)),  # Apply sharpening filter
            # transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)),  # Apply sharpening filter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
        ])
        
        self._angles = angles
        self._device = device
        self._sim_scores_buff     = []
        self._sim_scores_max_buff = []
        self._bias = score_bias
        self._gain = score_gain
        self._rbf_sigma = rbf_sigma
        self._rbf_gain = rbf_gain
        
        
    
        self._kalman_gain = kalman_gain
        self._similarity_max_windows_size = similarity_max_windows_size
    
        self._feature_extractor_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
        self._feature_extractor_model.eval().to(device)
        
        self.set_landmarks(landmark_images_path)
        

    def generate_rotated_images(self, image):
        """Generates rotated versions of an image."""
        rotated_images = []
        for angle in self._angles:
            rotated_image = transforms.functional.rotate(image, angle)
            rotated_images.append(self._preprocess(rotated_image).unsqueeze(0))
        
        return rotated_images

    def get_average_embedding(self, image):
        """Returns the average embedding of an image across multiple rotations."""
        rotated_images = self.generate_rotated_images(image)
        rotated_images = torch.cat(rotated_images).to(device)  # Concatenate and send to device
    
        with torch.no_grad():
            embeddings = self._feature_extractor_model(rotated_images)
        
        average_embedding = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, embedding_dim]
    
        return average_embedding
    
    def set_landmarks(self,landmark_images_path):
        """Sets the landmarks for the similarity checker."""
        landmark_embeddings = []
        
        if len(landmark_images_path)>0:
            for landmark_image_path in landmark_images_path:
                landmark_image = Image.open(landmark_image_path) #.convert("RGB")
                landmark_embedding = self.get_average_embedding(landmark_image)
                # normalize 
                landmark_embedding = F.normalize(landmark_embedding)

                landmark_embeddings.append(landmark_embedding)    

        self._landmark_embeddings = landmark_embeddings
        return landmark_embeddings
  

    def compute_similarity_with_landmarks(self, frame, bias=0.4, gain=3.0, RBF_sigma=7.0, RBF_gain=2.0):
        """Computes the similarity of a frame with the landmarks."""            
        input_embedding = self.get_average_embedding(frame)
        similarities = []
        for landmark_embedding in self._landmark_embeddings:
            
            input_embedding = F.normalize(input_embedding)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(input_embedding, landmark_embedding).item()**2
            similarity_dist = torch.linalg.norm(input_embedding-landmark_embedding).cpu().numpy()
            similarity_exp = RBF_gain*np.exp(-RBF_sigma*similarity_dist**2)
            
            similarities.append(gain*(similarity-bias)+similarity_exp)  # Convert tensor to scalar
            
        return similarities
    
    def reset_scores(self):
        self._sim_scores_buff     = []
        self._sim_scores_max_buff = []
        self._sim_scores_raw_buff = []

        
    def process(self, frame):
        
        if len(self._landmark_embeddings) == 0:
            self._sim_scores_buff = np.array([0,0]).reshape(1,2)
            self._sim_scores_max_buff = np.array([0,0]).reshape(1,2)
            self._sim_scores_raw_buff = np.array([0,0]).reshape(1,2)
            return np.array([0,0]).reshape(1,2)
        
        # Compute similarity scores
        similarity_scores = np.array(self.compute_similarity_with_landmarks(frame,self._bias,self._gain,self._rbf_sigma,self._rbf_gain))
        
        if len(self._sim_scores_buff) == 0:
            self._sim_scores_buff = similarity_scores.reshape(1,len(self._landmark_embeddings))
            self._sim_scores_max_buff = similarity_scores.reshape(1,len(self._landmark_embeddings))
            self._sim_scores_raw_buff = similarity_scores.reshape(1,len(self._landmark_embeddings))
            return similarity_scores 

        else:
            if self._sim_scores_buff.shape[0]>1:
                new_ss = self._kalman_gain*self._sim_scores_buff[-1] + (1-self._kalman_gain)*similarity_scores
                
                if self._sim_scores_max_buff.shape[0]<self._similarity_max_windows_size:
                    self._sim_scores_max_buff = np.vstack([self._sim_scores_max_buff,
                                            np.array([np.maximum(np.max(self._sim_scores_buff[:,j]), new_ss[j]) for j in range(len(self._landmark_embeddings))])])
                else:
                    self._sim_scores_max_buff = np.vstack([self._sim_scores_max_buff,
                                            np.array([np.maximum(np.max(self._sim_scores_buff[-self._similarity_max_windows_size:,j]), new_ss[j]) for j in range(len(self._landmark_embeddings))])])
                
                self._sim_scores_buff = np.vstack([self._sim_scores_buff,new_ss])
                self._sim_scores_raw_buff = np.vstack([self._sim_scores_raw_buff,similarity_scores])
                

            else:
                try:
                    # if len(similarity_scores.shape) == 1:
                    #     similarity_scores = similarity_scores.reshape(1,len(self._landmark_embeddings))
                    self._sim_scores_buff = np.vstack([self._sim_scores_buff,similarity_scores])
                    self._sim_scores_max_buff = np.vstack([self._sim_scores_max_buff,similarity_scores])
                    self._sim_scores_raw_buff = np.vstack([self._sim_scores_raw_buff,similarity_scores])
                    return similarity_scores #.reshape(1,len(self._landmark_embeddings))
                except Exception as e:
                    print (e)
                    print ("err", similarity_scores.shape,self._sim_scores_buff.shape)
                    # print (self._sim_scores_buff)
                    # print (self._sim_scores_max_buff)
                    # print (self._sim_scores_raw_buff)
                    return similarity_scores
                
        
        
        return self._sim_scores_max_buff[-1,:] #new_ss.reshape(1,len(self._landmark_embeddings))
