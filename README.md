# Fusion_DNN_OPG_EHR

Here we developed a multimodal deep learning model to predict systemic diseases from oral condition. We used both OPG and EHR data to train our model. 


![Drawing4](https://user-images.githubusercontent.com/96053939/163914797-adeac54e-3ca5-47a7-b889-8d163c175758.png)


# Dual-loss autoencoder
We used a dual-loss autoencoder to extract periodontal-related features from OPG images. The code of our autoecoder can be found in [autoencoder_model.py](https://github.com/ClinicalAI/Fusion_DNN_OPG_EHR/blob/main/autoencoder_model.py)

# DNN fusion model
We fused the OPG features with EHR features to train a DNN  model to predict systemic diseases. The code of the model can be found in [fusion_model.py](https://github.com/ClinicalAI/Fusion_DNN_OPG_EHR/blob/main/fusion_model.py)


# Run the pipeline
To run the whole pipeline:

`python main.py`

# Training dataset
The raw datasets from the Prince Philip Dental Hospital (PPDH) and Queen Mary Hospital (QMH) cannot be made available due to hospital regulation restrictions and patient privacy concerns. However, we uplpaded few samples as NPZ files to run the models.


