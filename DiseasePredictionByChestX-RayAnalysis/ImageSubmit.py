from DataPreparation.DataNormalizationFromImage import normalizeImage
import pickle

def submit(image):
    # normalze the image
    image_data = normalizeImage(image)
    # Predict the condition of the chest by uploaded image
    predicted_chest_condition = predict([image_data])[0]
    print(predicted_chest_condition)
    return convert_numerics_inTo_texual_chest_condition(predicted_chest_condition)

def convert_numerics_inTo_texual_chest_condition(predicted_chest_condition):
    chest_condition_label = ['Nodule',
                            'No Finding',	
                            'Infiltration',	
                            'Pneumonia',
                            'Edema',
                            'Cardiomegaly',	
                            'Pleural_Thickening',	
                            'Emphysema',
                            'Fibrosis',
                            'Pneumothorax',	
                            'Mass',
                            'Effusion',	
                            'Hernia',
                            'Atelectasis',	
                            'Consolidation'] 
    
    predicted_chest_condition_names = []
    for i in range(0, len(predicted_chest_condition)):
        if predicted_chest_condition[i]==1:
            predicted_chest_condition_names.append(chest_condition_label[i])
    return predicted_chest_condition_names;


def predict(image_data):
    model = load_model()
    # feed the image into the model to get result
    predicted_data = model.predict(image_data)
    return predicted_data

def load_model():
    # Model Path
    trained_model_dir = 'C:\\Users\\SMAJUMDAR\\AI_ML_HACKATHON_2024_1\\Hackathon2024_1\\DiseasePredictionByChestX-RayAnalysis\\ModelPreparation'
    trained_model_file = trained_model_dir + "\\trained_model.sav"
    # Load the Model using pickle
    trained_model = pickle.load(open(trained_model_file,'rb'))
    return trained_model