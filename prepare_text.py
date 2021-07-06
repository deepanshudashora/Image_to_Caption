def clean_text_data(captions,image_features):
    error_count = 0
    captions_dict = {}
    for i in captions:
        img_name = i.split(',')[0] 
        try:
            caption = i.split(',')[1]
            if img_name in image_features:
                if img_name not in captions_dict:
                    captions_dict[img_name] = [caption]                    
                else:
                    captions_dict[img_name].append(caption)
        except:
            error_count+=1

    return captions_dict,error_count

def preprocessed(txt):
    modified = txt.lower()
    modified = 'startofseq ' + modified + ' endofseq'
    return modified