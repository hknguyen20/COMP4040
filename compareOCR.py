
import os
import cv2
import pytesseract
from PIL import Image
import easyocr
import json

reader_dict = {
    'ch_sim': easyocr.Reader(['ch_sim', 'en']),
    'ms': easyocr.Reader(['ms', 'en']),
    'ta': easyocr.Reader(['ta', 'en'])
}

def pyTexxeract(image):
    config = "-l eng+chi_sim+chi_tra+tam+msa --psm 4 --oem 1"
    text = pytesseract.image_to_string(image, config=config)
    return ''.join(text).replace('\n','').replace('\f','')

def eazyOCR(image_path):
    results_list = []
    text_list = []
    for _,reader in reader_dict.items():
        results_list.append(reader.readtext(image_path))
        text_list.append(reader.readtext(image_path, detail = 0, paragraph=True))

    best_confidence = 0
    
    for i in range(3):
        avg_confidence = sum([result[2] for result in results_list[i]]) / len(results_list[i]) if results_list[i] else 0
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_text = text_list[i]
    return ' '.join(best_text)

if __name__ == "__main__":    
    imgs_list = ['0125','0169','0271','0272','0277','0295','0297','0298','0371','0386','0409','0481','0551','0564','0611','0814','0851','0879','1110','1191']
    print(len(imgs_list))
    store = {}
    for img in imgs_list:
        filepath=f'/home/ubuntu/shared/ospc/img/{img}.png'
        im = cv2.imread(filepath)
        value = {}
        print('Comparing for imgage id', img)
        value['A'] = pyTexxeract(im) 
        value['B'] = eazyOCR(filepath)
        store[img] = value
    with open('compareOCR_output.jsonl', 'w', encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False, indent=4)
    