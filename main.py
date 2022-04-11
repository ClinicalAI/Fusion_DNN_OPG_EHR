import autoencoder_model 
import fusion_model

if __name__ == "__main__":
    
    img_npz_path = './dataset/images_npz/allOPG.npz'
    ehr_path = './dataset/clinical_data/EHR.csv'

    train_data, valid_data, train_label_cat, valid_label_cat, ids = autoencoder_model.loading_data(img_npz_path, ehr_path)

    Autoencoder_Model = autoencoder_model.Autoencoder()
    Autoencoder_Model.summary()

    autoencoder_model.trianing_ae_model(Autoencoder_Model, train_data, valid_data, train_label_cat, valid_label_cat, epochs=10)
    la_train_1, la_valid_1 = autoencoder_model.laten_space_save(Autoencoder_Model, train_data, valid_data, train_label_cat, valid_label_cat, ids)
    autoencoder_model.pca_vis(la_train_1, la_valid_1, train_label_cat, valid_label_cat)

    
    latent_dic_path = './latent_space/latent_dic.npz'
    labels_path = './dataset/clinical_data/labels_chapter.csv'

    fusion_model.training_fusion_model(latent_dic_path, labels_path, ehr_path)
