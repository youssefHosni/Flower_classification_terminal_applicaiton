import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json
import numpy as np

def main(model_name,image_path,top_k,class_names):
    saved_keras_model_filepath= model_name
    model = tf.keras.models.load_model(saved_keras_model_filepath,custom_objects={'KerasLayer': hub.KerasLayer})
    model.summary()
    top_k_props,top_k_classes=predict(image_path,model,top_k)   
    with open(class_names, 'r') as f:
        class_names = json.load(f)
    top_k_classes=top_k_classes.astype(int)    
    predicted_k_classes=[class_names[str(classe+1)] for classe in top_k_classes[0]]
    #predicted_k_classes=class_names[str(top_k_classes[0][0]+1)]
    return top_k_props, top_k_classes, predicted_k_classes
        
def process_image(input_image):
    input_image= tf. convert_to_tensor(input_image)
    print("the dimension of the test image is ", np.shape(input_image))
    input_image=tf.image.resize(input_image,[224,224])
    input_image=input_image/225
    output_image=input_image.numpy()
    return output_image

def predict(image_path,model,top_k):
    top_k_props=[]
    top_k_classes=[]
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image=process_image(image)
    processed_image=np.expand_dims(processed_image,axis=0)
    props=model(processed_image)
    [top_k_props, top_k_classes] = tf.math.top_k(props, k=top_k, sorted=True)
    top_k_props=top_k_props.numpy()
    top_k_classes=top_k_classes.numpy() 
    return top_k_props,top_k_classes

