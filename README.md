The code is prepared such that it should work if the user has all the installed frameworks. 

# Build the models: prob and evi
The subfolder "model_arch" stands for the models (probabilistic and evidential) and involved libraries (belief theory libraries):

# Training of the Model 
Once models are built, the script "run_model_name.ipynb" can be run to do the training. This particular scrip initially loads the input data as well as the needed evidential libraries.
Thus, data can be visualized, batch, etc. based on created dictionaries. 

For the training, the model is also loading some pretrained weights from a camera model with 400 epochs. And then model can be re-trained or fine-tuned. 

# Evaluation Prob and Evi
Once the model is trained, the weights will be saved in output and the evaluation folder can be checked. 

In the evaluation folder, there are two scripts, for both probabilistic and evidential formulation. In this way, the evidential model can take the weights from probabilistic and apply a particular decision-making 
based on DbI, where the extra class it is added. 


