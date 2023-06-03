from email.mime import image
from torchvision import transforms
import json
from utils import *
from PIL import Image, ImageDraw, ImageFont
from pprint import PrettyPrinter
import argparse
import cv2

pp = PrettyPrinter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
model_path = "./models/"
checkpoint = model_path + 'm6.pth.tar'
checkpoint = torch.load(checkpoint, map_location =device)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, test_object, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    images=list()
    boxes = torch.FloatTensor(test_object['boxes']) # (n_objects, 4)
    labels = torch.LongTensor(test_object['labels'])  # (n_objects)
    difficulties = torch.ByteTensor(test_object['difficulties'])  # (n_objects)
    
    # Transform
    image = normalize(to_tensor(resize(original_image)))
    #print(labels)
    # Move to default device
    #image, boxes, labels, difficulties= transform(original_image, boxes, labels, difficulties, split='TEST')
    images.append(image)
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes_batch = det_boxes_batch[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes_batch = det_boxes_batch * original_dims

    # Decode class integer labels
    #det_labels_batch = [rev_label_map[l] for l in det_labels_batch[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels_batch == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font= ImageFont.load_default()
    #font = ImageFont.truetype("./calibril.ttf", 15)
    #test_object=json.loads(test_object)
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
   
    #image_path = torch.from_numpy(np.asarray(image_path))
    #image_path = image_path.to(device)
    
    det_boxes.append(det_boxes_batch)
    det_labels.append(det_labels_batch)
    det_scores.append(det_scores_batch)
    true_boxes.append(boxes)
    true_labels.append(labels)
    true_difficulties.append(difficulties)

    print((det_boxes),(det_labels_batch),(det_scores_batch),(true_boxes),(true_labels))

    APs, mAP = calculate_mAP(det_boxes, det_labels_batch, det_scores_batch, true_boxes, true_labels, true_difficulties)
    pp.pprint(APs)
    # Suppress specific classes, if needed
    det_labels_batch = [rev_label_map[l] for l in det_labels_batch[0].to('cpu').tolist()]
    for i in range(det_boxes_batch.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        #boxes = [b.to(device) for b in boxes]
        #labels = [l.to(device) for l in labels]
        #difficulties = [d.to(device) for d in difficulties]
        # Boxes
        box_location = det_boxes_batch[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels_batch[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels_batch[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels_batch[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels_batch[i]])
        draw.text(xy=text_location, text=det_labels_batch[i].upper()+":  " +str(round(det_scores_batch[0][i].item(),2)), fill='white',
                  font=font)
    del draw

    return annotated_image

def detect_batch(paths):
    for pt in paths:
        print('processed path::'+ str(pt))
        image_name = pt.split('/')[-1]
        original_image = Image.open(pt, mode='r')
        original_image = original_image.convert('RGB')
        #img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg'
        #original_image = Image.open(img_path, mode='r')
        #original_image = original_image.convert('RGB')
        annotated_image=detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
        annotated_image = np.asarray(annotated_image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        #cv2.imshow('Annotated image', annotated_image)
        cv2.waitKey(0)
        cv2.imwrite(f"outputs/{image_name}", annotated_image)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', 
                        default='./COCO/images/test/000000574739.jpg',
                        help='path to the test data')
    args = vars(parser.parse_args())
    img_path = args['input']
    with open(os.path.join('./output', 'TEST_images.json'), 'r') as j:
        test_images = json.load(j)
    with open(os.path.join('./output', 'TEST_objects.json'), 'r') as j:
            test_objects = json.load(j)
    assert len(test_images) == len(test_objects)
    for t,o in zip(test_images,test_objects):
        image_name = t.split('/')[-1]
        print('image path'+str(image_name)+str(o))
        original_image = Image.open(t, mode='r')
        original_image = original_image.convert('RGB')
        #img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg'
        #original_image = Image.open(img_path, mode='r')
        #original_image = original_image.convert('RGB')
        annotated_image=detect(original_image,o, min_score=0.2, max_overlap=0.5, top_k=200)
        annotated_image = np.asarray(annotated_image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        #cv2.imshow('Annotated image', annotated_image)
        # cv2.waitKey(0)
        cv2.imwrite(f"./output/images/{image_name}", annotated_image)
        
