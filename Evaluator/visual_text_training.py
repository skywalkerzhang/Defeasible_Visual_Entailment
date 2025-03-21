import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import wandb
import pandas as pd
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import random
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
import math


class ClassificationHead(nn.Module):
    """Head for classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob=0.1, num_labels=1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class AttentionHead(nn.Module):
    def __init__(self, hidden_size, num_labels=1):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(hidden_states.size(-1))
        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, value)
        attention_output = self.dropout(attention_output)
        output = self.out_proj(attention_output)
        return output


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def process_single_df(df, image_dir):
    image_paths = df['SNLIPairId'].apply(lambda x: os.path.join(image_dir, x.split('#')[0])).tolist()
    hypotheses = df['Hypothesis'].astype(str).tolist()
    if 'Premise' in df.columns:
        captions = df['Premise'].astype(str).tolist()
    else:
        captions = None
    updates = df['Update'].astype(str).tolist()
    update_types = df['UpdateType'].apply(lambda x: 1 if x == 'strengthener' else -1).tolist()  # 增强为 1，减弱为 -1
    return image_paths, hypotheses, captions, updates, update_types


class VisualTextDataset(Dataset):
    def __init__(self, image_paths, hypotheses, premises, updates, update_types, tokenizer, image_transform, max_length=128):
        self.image_paths = image_paths
        self.hypotheses = hypotheses
        self.premises = premises
        self.updates = updates
        self.update_types = update_types
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)

            hypothesis = self.hypotheses[idx]
            update = self.updates[idx]
            update_type = self.update_types[idx]

            # 将 hypothesis 和 update 组合在一起进行编码
            inputs_hypothesis_update = self.tokenizer(
                hypothesis,
                update,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

            # 如果 premises 存在，将 hypothesis 和 premise 组合在一起进行编码
            if self.premises is not None:
                premise = self.premises[idx]
                inputs_hypothesis_premise = self.tokenizer(
                    hypothesis,
                    premise,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                inputs_hypothesis_premise = {key: val.squeeze(0) for key, val in inputs_hypothesis_premise.items()}
            else:
                # 直接赋值为空字符串
                inputs_hypothesis_premise = {"input_ids": torch.tensor([], dtype=torch.long),
                                             "attention_mask": torch.tensor([], dtype=torch.long)}

            inputs_hypothesis_update = {key: val.squeeze(0) for key, val in inputs_hypothesis_update.items()}

            return image, inputs_hypothesis_premise, inputs_hypothesis_update, update_type
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None

    def __len__(self):
        return len(self.hypotheses)


class CustomLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomLoss, self).__init__()
        self.reduction = reduction

    def forward(self, score_hypo_premise, score_hypo_update, labels):
        outputs = (score_hypo_update - score_hypo_premise).view(-1)
        loss = - torch.mean(torch.log(torch.sigmoid(outputs * labels.view(-1))))
        return loss


class VisualEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(VisualEncoder, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉最后的分类层

    def forward(self, images):
        features = self.model(images)
        features = features.view(features.size(0), -1)  # 展平为二维
        return features


class BERTModelModule(nn.Module):
    def __init__(self, model_name: str, use_classification_head=False, classification_weight=0.9):
        super(BERTModelModule, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.use_classification_head = use_classification_head
        self.classification_weight = classification_weight
        self.visual_encoder = VisualEncoder()
        self.custom_loss_fn = CustomLoss()

        if self.use_classification_head:
            self.classifier = nn.Linear(self.bert_model.config.hidden_size + 2048, 2)
            self.ce_loss_fn = nn.CrossEntropyLoss()

        self.regressor = nn.Linear(self.bert_model.config.hidden_size + 2048, 1)

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.visual_encoder(images)
        text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output  # 获取 [CLS] token 的输出
        combined_features = torch.cat((pooled_output, visual_features), dim=1)
        if self.use_classification_head:
            logits = self.classifier(combined_features)
        else:
            logits = None
        score = self.regressor(combined_features)
        return logits, score

    def compute_loss_and_scores(self, batch):
        weight = self.classification_weight
        images, hypo_premise_inputs, hypo_update_inputs, update_types = batch

        if hypo_premise_inputs and hypo_premise_inputs['input_ids'].numel() > 0:
            # 获取文本和图像的特征并计算分类和回归分数
            logits_hypo_premise, score_hypo_premise = self.forward(images, hypo_premise_inputs['input_ids'],
                                                                   hypo_premise_inputs['attention_mask'])
        else:
            logits_hypo_premise = None
            score_hypo_premise = None

        # 获取文本和图像的特征并计算分类和回归分数
        logits_hypo_update, score_hypo_update = self.forward(images, hypo_update_inputs['input_ids'],
                                                             hypo_update_inputs['attention_mask'])

        # 计算自定义损失
        if hypo_premise_inputs and hypo_premise_inputs['input_ids'].numel() > 0:
            custom_loss = self.custom_loss_fn(score_hypo_premise, score_hypo_update, update_types)
        else:
            custom_loss = None

        if self.use_classification_head:
            if logits_hypo_update is not None:
                # 计算分类损失
                update_types_classification = (update_types + 1) // 2  # 将 -1, 1 转换为 0, 1
                ce_loss = self.ce_loss_fn(logits_hypo_update, update_types_classification.long())
            else:
                ce_loss = None
            # 总损失
            if custom_loss is not None and ce_loss is not None:
                loss = weight * ce_loss + (1 - weight) * custom_loss
            elif ce_loss is not None:
                loss = ce_loss
            else:
                loss = custom_loss
        else:
            loss = custom_loss

        return loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', type=str, default='../Data/DVE_train.csv ',
                        help='CSV file containing image paths, captions, and targets for training.')
    parser.add_argument('--val_csv_file', type=str, default='../Data/DVE_dev.csv',
                        help='CSV file containing image paths, captions, and targets for validation.')
    parser.add_argument('--test_csv_file', type=str, default='../Data/DVE_test.csv',
                        help='CSV file containing image paths, captions, and targets for testing.')
    parser.add_argument('--image_dir', type=str, default='/home/yxz230014/Defeasible_Visual_Entailment_agent_test/Data/flickr30k_images',
                        help='Directory containing images.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for fine-tuning.')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate for fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--wandb_project', type=str, default="DI_visual_text", help='wandb project name.')
    parser.add_argument('--output_model', type=str, default="DI_visual_text",
                        help='Base name for the output model file.')
    parser.add_argument('--dataset_type', type=str, choices=['VE', 'DI'], default='DI',
                        help='Dataset type: VE or DI.')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for inference.')
    parser.add_argument('--model_path', type=str, help='Path to the fine-tuned model checkpoint.')
    parser.add_argument('--output_file', type=str, default='test_results.csv',
                        help='File to save the test results.')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--classification_weight', type=float, help='The weight for classification loss.')
    parser.add_argument('--use_classification_head', action='store_true', help='Whether to use classification head.')
    return parser.parse_args()


def generate_filename(base_name, lr, accuracy, batch_size, ext):
    return f"{base_name}_lr{lr}_acc{accuracy:.4f}_bs{batch_size}.{ext}"


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, args):
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images, hypo_premise_inputs, hypo_update_inputs, update_types = batch
            images = images.to(device)
            hypo_premise_inputs = {k: v.to(device) for k, v in hypo_premise_inputs.items()}
            hypo_update_inputs = {k: v.to(device) for k, v in hypo_update_inputs.items()}
            update_types = torch.tensor(update_types, dtype=torch.float32).clone().detach().to(device)
            optimizer.zero_grad()

            loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update = model.compute_loss_and_scores(
                (images, hypo_premise_inputs, hypo_update_inputs, update_types))

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            wandb.log({"step_loss": loss.item()})

        avg_train_loss = np.mean(train_losses)
        wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        model.eval()
        val_losses = []
        val_correct_predictions = 0
        val_total_predictions = 0
        val_classification_correct_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                images, hypo_premise_inputs, hypo_update_inputs, update_types = batch
                images = images.to(device)
                hypo_premise_inputs = {k: v.to(device) for k, v in hypo_premise_inputs.items()}
                hypo_update_inputs = {k: v.to(device) for k, v in hypo_update_inputs.items()}
                update_types = torch.tensor(update_types, dtype=torch.float32).clone().detach().to(device)

                val_loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update = model.compute_loss_and_scores(
                    (images, hypo_premise_inputs, hypo_update_inputs, update_types))

                val_losses.append(val_loss.item())

                # 计算自定义的准确率
                for score_hp, score_hu, update_type in zip(score_hypo_premise, score_hypo_update, update_types):
                    if (update_type == 1 and score_hu > score_hp) or (update_type == -1 and score_hu < score_hp):
                        val_correct_predictions += 1
                    val_total_predictions += 1

                # 计算分类准确率
                if args.use_classification_head:
                    preds = torch.argmax(logits_hypo_update, dim=1)
                    update_types_classification = (update_types + 1) // 2  # 将 -1, 1 转换为 0, 1
                    val_classification_correct_predictions += (preds == update_types_classification.long()).sum().item()

        avg_val_loss = np.mean(val_losses)
        val_accuracy = val_correct_predictions / val_total_predictions if val_total_predictions > 0 else 0
        val_classification_accuracy = val_classification_correct_predictions / val_total_predictions if val_total_predictions > 0 else 0

        wandb.log({
            "epoch_val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_classification_accuracy": val_classification_accuracy,
            "epoch": epoch + 1
        })

        print(
            f"Epoch {epoch + 1}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}, Val Classification Accuracy: {val_classification_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_filename = generate_filename(args.output_model, args.lr, val_accuracy, args.batch_size, "pth")
            torch.save(model.state_dict(), model_filename)
            print(f"Best model saved with accuracy: {val_accuracy}")

        scheduler.step()


def test(model, test_loader, device, args=None, calculate_accuracy=True, return_df=False):
    model.eval()
    results = []
    correct_predictions = 0
    total_predictions = 0
    classification_correct_predictions = 0
    premise_valid = False

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, hypo_premise_inputs, hypo_update_inputs, targets = batch
            images = images.to(device)
            hypo_update_inputs = {k: v.to(device) for k, v in hypo_update_inputs.items()}
            targets = targets.to(device)

            if hypo_premise_inputs['input_ids'].size(0) > 0 and hypo_premise_inputs['attention_mask'].size(0) > 0:
                hypo_premise_inputs = {k: v.to(device) for k, v in hypo_premise_inputs.items()}
                premise_valid = True

            loss, logits_hypo_premise, logits_hypo_update, score_hypo_premise, score_hypo_update = model.compute_loss_and_scores(
                (images, hypo_premise_inputs if premise_valid else None, hypo_update_inputs, targets))

            # 计算自定义的准确率
            if calculate_accuracy:
                if premise_valid:
                    for score_hp, score_hu, target in zip(score_hypo_premise, score_hypo_update, targets):
                        if (target == 1 and score_hu > score_hp) or (target == -1 and score_hu < score_hp):
                            correct_predictions += 1
                        total_predictions += 1
                        results.append((score_hu.item(), score_hp.item(), target.item()))
                else:
                    for score_hu, target in zip(score_hypo_update, targets):
                        total_predictions += 1
                        results.append((score_hu.item(), None, target.item()))
            else:
                for score_hu, target in zip(score_hypo_update, targets):
                    results.append((score_hu.item(), None, target.item()))

            # 计算分类准确率
            if calculate_accuracy and args.use_classification_head:
                preds = torch.argmax(logits_hypo_update, dim=1)
                update_types_classification = (targets + 1) // 2  # 将 -1, 1 转换为 0, 1
                classification_correct_predictions += (preds == update_types_classification.long()).sum().item()

    if calculate_accuracy:
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        classification_accuracy = classification_correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Accuracy: {accuracy * 100:.2f}%, Classification Accuracy: {classification_accuracy * 100:.2f}%")

    results_df = pd.DataFrame(results, columns=['Score_Update', 'Score_Premise', 'UpdateType'])
    if not premise_valid or not calculate_accuracy:
        results_df = results_df.drop(columns=['Score_Premise'])

    if return_df:
        return results_df
    else:
        if calculate_accuracy:
            csv_filename = generate_filename(args.output_file.replace(".csv", ""), args.lr, accuracy,
                                             args.test_batch_size, "csv")
        else:
            csv_filename = generate_filename(args.output_file.replace(".csv", ""), args.lr, 0, args.test_batch_size,
                                             "csv")
        results_df.to_csv(csv_filename, index=False)
        print(f"Test results saved to {csv_filename}")
        return None


def main():
    args = parse_args()

    wandb.init(project=args.wandb_project)

    train_df = pd.read_csv(args.train_csv_file)
    val_df = pd.read_csv(args.val_csv_file)
    test_df = pd.read_csv(args.test_csv_file)

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    train_image_paths, train_hypotheses, train_premises, train_updates, train_update_types = process_single_df(train_df, args.image_dir)
    val_image_paths, val_hypotheses, val_premises, val_updates, val_update_types = process_single_df(val_df, args.image_dir)
    test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types = process_single_df(test_df, args.image_dir)

    image_transform = Compose([
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model_name = "bert-large-uncased"
    model = BERTModelModule(model_name=model_name, use_classification_head=args.use_classification_head).to(device)

    train_dataset = VisualTextDataset(train_image_paths, train_hypotheses, train_premises, train_updates, train_update_types, tokenizer, image_transform)
    val_dataset = VisualTextDataset(val_image_paths, val_hypotheses, val_premises, val_updates, val_update_types, tokenizer, image_transform)
    test_dataset = VisualTextDataset(test_image_paths, test_hypotheses, test_premises, test_updates, test_update_types, tokenizer, image_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / args.epochs)

    # 训练模型
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, args)

    # 测试模型
    test_accuracy = test(model, test_loader, device, args)
    wandb.log({"test_accuracy": test_accuracy})

    wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    main()
