import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
import torchmetrics

class MyModule(nn.Module):
    def __init__(self, num_inp=768, num_units=18):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(num_inp, num_units)

    def forward(self, X, **kwargs):
        return self.dense0(X)

class DiagModule(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = MyModule(model_hparams['num_inp'], model_hparams['num_units'])
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

        # Initialize dictionary to store classwise accuracy
        self.classwise_acc = {}
        # Initialize confusion matrix
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=model_hparams['num_units'], normalize="true")
        self.final_confusion_matrix = None 
        # Initialize variable to store predictions
        self.predictions = []
    
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x)
        loss = self.loss_module(preds, targets)

        acc = (preds.argmax(dim=-1) == targets).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)

        acc = (targets == preds).float().mean()
        self.log('val_acc', acc)
        # Calculate classwise accuracy
        class_acc = torchmetrics.functional.accuracy(preds, targets, task='multiclass', num_classes=self.hparams.model_hparams['num_units'], average=None).cpu().numpy()

        # Update classwise accuracy in the log dictionary
        for i, acc in enumerate(class_acc):
            self.classwise_acc[f'class_{i}'] = acc
        
        # Calculate confusion matrix
        self.confmat(preds, targets)
        
    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)
        self.predictions.append(preds)

        acc = (targets == preds).float().mean()
        self.log('test_acc', acc)

        # Calculate classwise accuracy
        class_acc = torchmetrics.functional.accuracy(preds, targets, task='multiclass', num_classes=self.hparams.model_hparams['num_units'], average=None).cpu().numpy()

        # Update classwise accuracy in the log dictionary
        for i, acc in enumerate(class_acc):
            self.classwise_acc[f'class_{i}'] = acc
        
        # Calculate confusion matrix
        self.confmat(preds, targets)

    def on_validation_epoch_end(self):
        # Log classwise accuracy at the end of each epoch
        self.log_dict(self.classwise_acc, on_epoch=True, prog_bar=True)
        self.classwise_acc = {}

        # Compute and log confusion matrix
        self.final_confusion_matrix = self.confmat.compute().cpu().numpy()
        self.confmat.reset()

    def on_test_epoch_end(self):
        # Log classwise accuracy at the end of testing
        self.log_dict(self.classwise_acc, on_epoch=True, prog_bar=True)
        self.classwise_acc = {}

        # Compute and log confusion matrix
        self.final_confusion_matrix = self.confmat.compute().cpu().numpy()
        self.confmat.reset()

        # Save predictions
        self.predictions = torch.cat(self.predictions, dim=0).cpu().numpy()
