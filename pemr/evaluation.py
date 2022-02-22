import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from pemr.data import ContrastiveDataset
import torchaudio

def evaluate(
    args, encoder, finetuned_head, test_dataset, dataset_name: str, segment_length: int, device
) -> dict:
    est_array = []
    gt_array = []

    to_db = torchaudio.transforms.AmplitudeToDB()
    encoder = encoder.to(device)
    encoder.eval()

    if finetuned_head is not None:
        finetuned_head = finetuned_head.to(device)
        finetuned_head.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            _, label = test_dataset[idx]
            batch = test_dataset.concat_clip(idx, segment_length)
            batch = torchaudio.transforms.MelSpectrogram(
                hop_length=args.hop_size, n_mels=args.n_mels, n_fft=args.n_fft)(batch.cpu())
            batch = batch.to(device).squeeze()
            batch = to_db(batch)
            output = encoder(batch)
            if finetuned_head:
                output = finetuned_head(output)

            # we always return logits, so we need a sigmoid here for multi-label classification
            if dataset_name in ["magnatagatune", "msd"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            track_prediction = output.mean(dim=0)
            est_array.append(track_prediction)
            gt_array.append(label)

    if dataset_name in ["magnatagatune"]:
        est_array = torch.stack(est_array, dim=0).cpu().numpy()
        gt_array = torch.stack(gt_array, dim=0).cpu().numpy()
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        return {
            "PR-AUC": pr_aucs,
            "ROC-AUC": roc_aucs,
        }

    accuracy = metrics.accuracy_score(gt_array, est_array)
    return {"Accuracy": accuracy}
