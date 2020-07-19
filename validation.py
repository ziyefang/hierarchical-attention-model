from dataset import TextDataset, collate_fn
from utils import *
import os, sys
from utils import MetricTracker
import webbrowser
import os.path

class Validate:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataset = TextDataset(config.validation_files, config.label_file, config.vocab_file)
        print(self.dataset.data)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False,
                                                      collate_fn=collate_fn)

    def eval(self):
        y_true = []
        predictions_all = []
        self.model.eval()
        with torch.no_grad():
            for (docs, labels, doc_lengths, sent_lengths) in self.dataloader:
                docs = docs.to(self.device)
                labels = labels.to(self.device)
                doc_lengths = doc_lengths.to(self.device)
                sent_lengths = sent_lengths.to(self.device)
                scores, word_att_weights, sentence_att_weights = self.model(docs, doc_lengths, sent_lengths)
                predictions = scores.max(dim=1)[1]
                print(predictions)
                print(labels)
                predictions_all += [p.item() for p in predictions]  # y_hat_class.squeeze()
                y_true += [y.item() for y in labels]
            print(y_true)
            print(predictions_all)
            metrics_results = MetricTracker.print_metrics_binary(y_true, predictions_all, logging)
            return metrics_results


if __name__ == "__main__":
    if not os.path.exists("best_model/model.pth.tar"):
        print("Visualization requires pretrained model to be saved under ./best_model.\n")
        print("Please run 'python train.py <args>'")
        sys.exit()

    checkpoint = torch.load("best_model/model.pth.tar")
    model = checkpoint['model']
    model.eval()

    train_files, validation_files, test_files = train_validation_test_split("C:/Users/ziyefang96/PycharmProjects/machinelearning/bin\HAN/testcombine")
    print(os.getcwd())
    print(os.path.isfile("test.py"))
    dataset = TextDataset(validation_files, "result.csv", "vectors.txt")
    #4 or 5 patients from the validation
    doc = "13 patient / test information : indication : coronary artery disease"

    result = visualize(model, dataset, doc)

    with open('result.html', 'w') as f:
        f.write(result)

    webbrowser.open_new('file://'+os.getcwd()+'/result.html')