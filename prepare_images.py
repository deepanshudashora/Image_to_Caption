import cv2

def extract_image_features(images,model):
    images_features = {}
    for i in images:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        
        img = img.reshape(1,224,224,3)
        pred = model.predict(img).reshape(2048,)
            
        img_name = i.split('/')[-1]
        
        images_features[img_name] = pred      
    return images_features
