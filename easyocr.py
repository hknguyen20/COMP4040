import easyocr
import json
import os

reader_dict = {
    'ch_sim': easyocr.Reader(['ch_sim', 'en']),
    'ms': easyocr.Reader(['ms', 'en']),
    'ta': easyocr.Reader(['ta', 'en'])
}


def get_best_ocr_result(image_path):
    results_list = []
    text_list = []
    bbox_list = []
    for _,reader in reader_dict.items():
        results_list.append(reader.readtext(image_path))
        text_list.append(reader.readtext(image_path, detail = 0, paragraph=True))

    best_confidence = 0
    best_result = None
    
    for i in range(3):
        avg_confidence = sum([result[2] for result in results_list[i]]) / len(results_list[i]) if results_list[i] else 0
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_text = text_list[i]
            best_result = results_list[i]
    
    for coord, _,_ in best_result:
        xmin = min([p[0] for p in coord])
        xmax = max([p[0] for p in coord])
        ymin = min([p[1] for p in coord])
        ymax = max([p[1] for p in coord])
        bbox_list.append((xmin, ymin, xmax, ymax))
    return ' '.join(best_text), bbox_list

if __name__ == '__main__':
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    folder_dir = '/home/ubuntu/shared/ospc'
    json_file = '/home/ubuntu/shared/ospc/train.jsonl'
    bbox = []
    metadata = []
    with open(json_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        image_path = os.path.join(folder_dir, data['img'])
        
        best_language, best_ocr_results, best_text = get_best_ocr_result(image_path)
        data['text'] = best_text
        metadata.append(data)
        bbox_data_serializable = [([(int(x), int(y)) for x, y in point_list], label, confidence) for point_list, label, confidence in best_ocr_results]
        
        bbox.append({'id': data['id'], 'bbox': bbox_data_serializable})

        print(f'Best detected language: {best_language}')
        print('Best OCR Results:', best_ocr_results)
        print('Best text :', best_text)
        
    with open('bbox.json', 'w', encoding="utf-8") as f:
        for item in bbox:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    with open('train.jsonl', 'w', encoding="utf-8") as f:
        for item in metadata:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')