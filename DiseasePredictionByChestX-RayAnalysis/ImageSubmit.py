from DataPreparation.DataNormalizationFromImage import normalizeImage
import pickle

def submit(image):
    # normalze the image
    image_data = normalizeImage(image)
    # Predict the condition of the chest by uploaded image
    #As the prediction returns the 2d array, we need the first and only element
    predicted_chest_condition = predict([image_data])[0]
    print(predicted_chest_condition)
    #before returnning, converting numeric list into diease name list
    return convert_numerics_inTo_texual_chest_condition(predicted_chest_condition)

def convert_numerics_inTo_texual_chest_condition(predicted_chest_condition):
    #All 15 disease labels
    chest_condition_label = ['Atelectasis',
                            'Consolidation',	
                            'Infiltration',	
                            'Pneumothorax',
                            'Edema',
                            'Emphysema',	
                            'Fibrosis',	
                            'Effusion',
                            'Pneumonia',
                            'Pleural_Thickening',	
                            'Cardiomegaly',
                            'Nodule',	
                            'Mass',
                            'Hernia',	
                            'No Finding'] 
    
    predicted_chest_condition_names = []
    for i in range(0, len(predicted_chest_condition)):
        # In numeric list, which are 1's,
        # Those disease are marked
        if predicted_chest_condition[i]==1:
            predicted_chest_condition_names.append(chest_condition_label[i])
    return predicted_chest_condition_names;


def predict(image_data):
    model = load_model()
    # feed the image into the model to get result
    predicted_data = model.predict(image_data)
    return predicted_data

def load_model():
    # Path where trained model is kept/stored
    trained_model_dir = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\ModelPreparation'
    trained_model_file = trained_model_dir + "\\trained_model_random_forest_45.sav"
    # Load the Model using pickle package of python
    trained_model = pickle.load(open(trained_model_file,'rb'))
    return trained_model