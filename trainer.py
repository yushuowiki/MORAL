import os
import csv
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from math import ceil
from my_utils.train.trainer import Trainer, BestRecorder

class MyTrainer(Trainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.interval = 25
        self.dataloader = DataLoader(self.dataset, batch_size=self.args['batch_size'], shuffle=True)
        if self.args['animator_output'] == True:
            self.animator.set_legend(['train loss', 'test loss', 'test acc', 'val acc'])
        self.best_recorder.initial(['train loss', 'test loss', 'test acc', 'val acc'], ['min', 'min', 'max', 'max'])

    def train(self):
        for epoch in range(self.args['epochs']):
            self.model.train()
            loss = 0
            for batch in self.dataloader:
                l = self.train_batch(batch)
            loss += l

            if epoch % 10 == 0:
                test_loss, test_acc = self.test()
                val_acc = self.valid()
            data = (loss, test_loss, test_acc, val_acc)
            self.best_recorder.cal_best(epoch, data)

            if self.args['animator_output'] == True:
                self.animator.step(epoch + 1, data)
            message = f'epoch:{epoch}, train_loss:{loss:.4f}, test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}, val_acc:{val_acc:.4f}'
            self.logger.log(message, data)
            self.checkpoint(epoch)

            if self.early_stop(test_acc, 'max'):
                break
    
    def train_batch(self, data):
        self.optimizer.zero_grad()
        data.to(self.device)
        self.criterion.to(self.device)

        feature_info = data.x
        structure_info = data.structure_info
        edge_index = data.edge_index
        labels = data.y
        
        feature_info, structure_info, edge_index = feature_info.to(self.device), structure_info.to(self.device), edge_index.to(self.device)
        
        if self.args['model_name'] == 'MORAL':
            y_pred = self.model(feature_info, structure_info, edge_index)
            if isinstance(y_pred, tuple): # MORAL
                s_org, s_rec, y_pred = y_pred
                reconstruct_loss = torch.sqrt(torch.sum(torch.pow(s_org - s_rec, 2), -1)).mean() # L2-norm for (origin - reconstruct)
        elif self.args['model_name'] == 'MORE':
            y_pred = self.model(feature_info, structure_info, edge_index)
        elif self.args['model_name'] == 'MLP':
            y_pred = self.model(feature_info)
        else: # other GNN models
            y_pred = self.model(feature_info, edge_index)
        y_pred = y_pred[data.train_mask]
        y_label = labels[data.train_mask]

        if len(data.y.shape) == 2: # multi labels
            y_pred = F.sigmoid(y_pred)
        classify_loss = self.criterion(y_pred, y_label.long())
        #print(y_pred.shape)
        try:
            loss = (1 - self.args['beta']) * classify_loss + self.args['beta'] * reconstruct_loss # for MORAL reconstruct
        except:
            loss = classify_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, model):
        model.eval()

        with torch.no_grad():
            data = self.dataset[0]
            data.to(self.device)
            feature_info = data.x
            structure_info = data.structure_info
            edge_index = data.edge_index
            feature_info, structure_info, edge_index = feature_info.to(self.device), structure_info.to(self.device), edge_index.to(self.device)
            if self.args['model_name'] == 'MORAL':
                y_pred = self.model(feature_info, structure_info, edge_index) # return tuple
            elif self.args['model_name'] == 'MORE':
                y_pred = self.model(feature_info, structure_info, edge_index)
            elif self.args['model_name'] == 'MLP':
                y_pred = model(feature_info)
            else: # other GCNs
                y_pred = model(feature_info, edge_index)
        
        if len(data.y.shape) == 2: # multi labels
            y_pred = (y_pred[0], y_pred[1], F.sigmoid(y_pred)) if isinstance(y_pred, tuple) else F.sigmoid(y_pred)
        
        
        return y_pred
    
    def test(self):
        data = self.dataset[0]
        data.to(self.device)
        labels = data.y
        y_pred = self.predict(self.model)
        if isinstance(y_pred, tuple): # MORAL
            s_org, s_rec, y_pred = y_pred
            test_reconstruct_loss = torch.sqrt(torch.sum(torch.pow(s_org - s_rec, 2), -1)).mean()

        y_pred = y_pred[data.test_mask]
        y_label = labels[data.test_mask]
        test_classify_loss = self.criterion(y_pred, y_label.long())
        try:
            test_loss = (1 - self.args['beta']) * test_classify_loss + self.args['beta'] * test_reconstruct_loss
        except:
            test_loss = test_classify_loss
        test_loss = test_loss.item()
        if len(y_label.shape) == 1:
            y_pred = F.softmax(y_pred, dim=-1).argmax(dim=-1)
            test_acc = (y_pred == y_label).sum() / len(y_label)
        else: # len = 2, multi labels
            test_acc = sum(row.all().int() for row in (y_pred.ge(0.5) == y_label.gt(0.0))) / len(y_label)
        test_acc = test_acc.item()

        return test_loss, test_acc

    def valid(self):
        data = self.dataset[0]
        data.to(self.device)
        labels = data.y
        y_pred = self.predict(self.model)
        if isinstance(y_pred, tuple):
            _, _, y_pred = y_pred

        y_pred = y_pred[data.val_mask]
        y_label = labels[data.val_mask]
        if len(y_label.shape) == 1:
            y_pred = F.softmax(y_pred, dim=-1).argmax(dim=-1)
            val_acc = (y_pred == y_label).sum() / len(y_label)
        else: # len = 2, multi labels
            val_acc = sum(row.all().int() for row in (y_pred.ge(0.5) == y_label.gt(0.0))) / len(y_label)
        val_acc = val_acc.item()

        return val_acc
    
    def cal_f1_score(self):
        ''' get best model by test acc'''
        record = self.best_recorder.get_best()[2] # test acc
        epoch = record[2]
        epoch = ceil(epoch / self.interval) * self.interval
        if epoch > self.args['epochs'] - 1:
            epoch = self.args['epochs'] - 1
        best_model = torch.load(f'{self.dir_path}/{self.topic}_epoch{epoch}.pt')

        data = self.dataset[0]
        data.to(self.device)
        labels = data.y
        y_pred = self.predict(best_model)
        if isinstance(y_pred, tuple):
            _, _, y_pred = y_pred

        y_pred = y_pred[data.test_mask]
        y_label = labels[data.test_mask]
        if len(y_label.shape) == 1:
            y_pred = F.softmax(y_pred, dim=-1).argmax(dim=-1)
        else: # len = 2, multi labels
            y_pred = F.sigmoid(y_pred)
            y_pred = y_pred.ge(0.5).int()
            y_label = y_pred.ge(0.0).int()

        f1 = f1_score(y_label.cpu(), y_pred.cpu(), average='macro')
        return f1
    
    def end_custom(self):
        data = self.dataset[0]
        data.to(self.device)
        test_best = self.best_recorder.get_best()[2][1]
        val_best = self.best_recorder.get_best()[3][1]
        f1_score = self.cal_f1_score()
        y_pred = self.predict(self.model)
        feature_info = y_pred[0]
        labels = data.y
        if isinstance(y_pred, tuple): # MORAL
            s_org, s_rec, y_pred = y_pred
            test_reconstruct_loss = torch.sqrt(torch.sum(torch.pow(s_org - s_rec, 2), -1)).mean()

        y_label = labels
        test_classify_loss = self.criterion(y_pred, y_label.long())
        try:
            test_loss = (1 - self.args['beta']) * test_classify_loss + self.args['beta'] * test_reconstruct_loss
        except:
            test_loss = test_classify_loss
        test_loss = test_loss.item()
        
        if len(y_label.shape) == 1:
            feature_info = y_pred
            y_pred = F.softmax(y_pred, dim=-1).argmax(dim=-1)
            
        else: # len = 2, multi labels
            y_pred == y_pred.ge(0.5)


        if not os.path.exists('records'):
            os.mkdir('records')
        
        try:
            file = open(self.args['record_path'], 'a')
        except:
            file = open('./records/record.csv', 'a')

        writer = csv.writer(file)
        records = [test_best, val_best, f1_score] + list(self.args.values())
        writer.writerow(records)
        file.close()
        return test_best,y_pred,feature_info