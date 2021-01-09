import argparse
import tensorflow as tf
import tensorflow_hub as hub
from prediction import main

parser = argparse.ArgumentParser(description='image classifier app')
#parser.add_argument('i',dest='image_path',  action="store", type= str,required=True)
#parser.add_argument('m',dest='model_name', action="store",type= str,required=True)

parser.add_argument('image_path',  action="store", type= str)
parser.add_argument('model_name', action="store",type= str)
parser.add_argument('--top_k',dest='top_k', action="store",type= int,required=False,default=1)
parser.add_argument('--category_names',dest='class_names', action="store",type= str,required=False,default='label_map.json')
args = parser.parse_args()
image_path=args.image_path
model_name=args.model_name
top_k=args.top_k
class_names=args.class_names
top_k_props, top_k_classes, predicted_k_classes=main(model_name,image_path,top_k,class_names)
print('top ' +str(top_k) +  ' classes ', predicted_k_classes)
print('top ' +str(top_k) + ' propabilites', top_k_props)

    
