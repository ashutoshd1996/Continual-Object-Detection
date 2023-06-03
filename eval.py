from utils import *
from datasets import PascalVOCDataset, COCODataset
from tqdm import tqdm
import torch.utils.data
from pprint import PrettyPrinter
from detect import detect_batch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './output/'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 6
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models/"
checkpoint = model_path + 'm6.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint, map_location=device)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = COCODataset(data_folder, split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluateTest(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]
            #image_path = torch.from_numpy(np.asarray(image_path))
            #image_path = image_path.to(device)
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            print(len(det_boxes),len(det_labels),len(det_scores),len(true_boxes),(true_labels),len(true_difficulties))
        # Calculate mAP
        APs, mAP, Prec, Rec = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.7)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    print('\nPrecision (P): %.3f' % Prec)
    print('\nRecall (R): %.3f \n' % Rec)

if __name__ == '__main__':
    # evaluateTest(test_loader, model)
    evaluate(test_loader,model)
