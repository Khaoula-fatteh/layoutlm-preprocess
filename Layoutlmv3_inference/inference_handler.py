##inference handler 


import os
import json
from PIL import Image
import torch
import logging
from .utils import load_model, load_processor, normalize_box, compare_boxes, adjacent
from .annotate_image import get_flattened_output, annotate_image

logger = logging.getLogger(__name__)

class ModelHandler(object):
    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, model_dir):
        logger.info("Loading transformer model")
        self.model_dir = model_dir
        self.model = self.load(self.model_dir)
        self.initialized = True

    def preprocess(self, batch):
        inference_dict = batch
        self._raw_input_data = inference_dict
        processor = load_processor()
        images = [Image.open(path).convert("RGB") for path in inference_dict['image_path']]
        self._images_size = [img.size for img in images]
        words = inference_dict['words']
        boxes = [[normalize_box(box, images[i].size[0], images[i].size[1]) for box in doc] for i, doc in enumerate(inference_dict['bboxes'])]
        encoded_inputs = processor(images, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        self._processed_data = encoded_inputs
        return encoded_inputs

    def load(self, model_dir):
        model = load_model(model_dir)
        return model

    def inference(self, model_input):
        with torch.no_grad():
            inference_outputs = self.model(**model_input)
            predictions = inference_outputs.logits.argmax(-1).tolist()
        results = []
        for i in range(len(predictions)):
            tmp = dict()
            tmp[f'output_{i}'] = predictions[i]
            results.append(tmp)
        return [results]

    def postprocess(self, inference_output):
        docs = []
        k = 0
        for page, doc_words in enumerate(self._raw_input_data['words']):
            doc_list = []
            width, height = self._images_size[page]
            for i, doc_word in enumerate(doc_words, start=0):
                word_tagging = None
                word_labels = []
                word = dict()
                word['id'] = k
                k += 1
                word['text'] = doc_word
                word['pageNum'] = page + 1
                word['box'] = self._raw_input_data['bboxes'][page][i]
                _normalized_box = normalize_box(self._raw_input_data['bboxes'][page][i], width, height)
                for j, box in enumerate(self._processed_data['bbox'].tolist()[page]):
                    if compare_boxes(box, _normalized_box):
                        if self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]] != 'O':
                            word_labels.append(self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]][2:])
                        else:
                            word_labels.append('other')
                if word_labels != []:
                    word_tagging = word_labels[0] if word_labels[0] != 'other' else word_labels[-1]
                else:
                    word_tagging = 'other'
                word['label'] = word_tagging
                word['pageSize'] = {'width': width, 'height': height}
                if word['label'] != 'other':
                    doc_list.append(word)
            spans = []
            def adjacents(entity): return [adj for adj in doc_list if adjacent(entity, adj)]
            output_test_tmp = doc_list[:]
            for entity in doc_list:
                if adjacents(entity) == []:
                    spans.append([entity])
                    output_test_tmp.remove(entity)

            while output_test_tmp != []:
                span = [output_test_tmp[0]]
                output_test_tmp = output_test_tmp[1:]
                while output_test_tmp != [] and adjacent(span[-1], output_test_tmp[0]):
                    span.append(output_test_tmp[0])
                    output_test_tmp.remove(output_test_tmp[0])
                spans.append(span)

            output_spans = []
            for span in spans:
                if len(span) == 1:
                    output_span = {"text": span[0]['text'], "label": span[0]['label'], "words": [{'id': span[0]['id'], 'box': span[0]['box'], 'text': span[0]['text']}]}
                else:
                    output_span = {"text": ' '.join([entity['text'] for entity in span]), "label": span[0]['label'], "words": [{'id': entity['id'], 'box': entity['box'], 'text': entity['text']} for entity in span]}
                output_spans.append(output_span)
            docs.append({f'output': output_spans})
        return docs

    def handle(self, data, output_dir):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        inference_out = self.postprocess(model_out)
        
        for i, inference_doc in enumerate(inference_out):
            image_path = data['image_path'][i]
            image_name = os.path.basename(image_path)
            json_output_path = os.path.join(output_dir, f'{image_name}_inference.json')
            with open(json_output_path, 'w') as json_file:
                json.dump(inference_doc, json_file, ensure_ascii=False, indent=4)
                
            flattened_output = get_flattened_output([inference_doc])
            annotate_image(image_path, flattened_output[0], output_dir)

_service = ModelHandler()

def handle(data, model_dir, output_dir):
    if not _service.initialized:
        _service.initialize(model_dir)

    if data is None:
        return None

    return _service.handle(data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--images_path", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSON and annotated images")
    args = parser.parse_args()

    images_dir = args.images_path
    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    words = []  # Your logic to obtain words from images
    bboxes = []  # Your logic to obtain bounding boxes from images
    
    data = {
        "image_path": images,
        "words": words,
        "bboxes": bboxes
    }

    handle(data, args.model_path, args.output_dir)
