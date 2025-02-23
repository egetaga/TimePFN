from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')


class Exp_Sub_Channel_Training(Exp_Basic):
    def __init__(self, args):
        super(Exp_Sub_Channel_Training, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optimizer')
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                channel_size = batch_x.size(2)
                max_channel_size = self.args.max_channel
                number_of_iter = channel_size // max_channel_size + 1

                pred=None
                true=None

                for j in range(number_of_iter):
                    batch_x_subset = batch_x[:, :, j * max_channel_size:(j + 1) * max_channel_size]
                    batch_y_subset = batch_y[:, :, j * max_channel_size:(j + 1) * max_channel_size]

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y_subset[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_subset[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_subset = batch_y_subset[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if pred is None:
                        pred = outputs.detach().cpu()
                        true = batch_y_subset.detach().cpu()
                    else:
                        pred = torch.cat((pred, outputs.detach().cpu()), dim=2)
                        true = torch.cat((true, batch_y_subset.detach().cpu()), dim=2)

                loss = criterion(pred, true)
                total_loss.append(loss)
                preds.append(pred.numpy())
                trues.append(true.numpy())

        total_loss = np.average(total_loss)



        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)


        self.model.train()
        return total_loss, mae, mse, rmse, mape, mspe

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #load the model from the path
        if self.args.load:
            self.model.load_state_dict(torch.load(self.args.load_path))
            #get the test and vali loss
            criterion = self._select_criterion()
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                0, 0, 0, vali_loss, test_loss))
            f = open("result_long_term_forecast.txt", 'a')
            f.write("zero-shot-" + setting + "  \n")
            f.write('mse:{}, mae:{}'.format(test_loss, test_mae))
            f.write('\n')
            f.write('\n')
            f.close()

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                channel_size = batch_x.size(2)
                max_channel_size = self.args.max_channel
                number_of_iter = channel_size // max_channel_size + 1
                for j in range(number_of_iter):
                    batch_x_subset = batch_x[:, :, j * max_channel_size:(j + 1) * max_channel_size]
                    batch_y_subset = batch_y[:, :, j * max_channel_size:(j + 1) * max_channel_size]
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y_subset[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_subset[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            if self.args.subset_prediction == -1:
                                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y_subset = batch_y_subset[:, -self.args.pred_len:, f_dim:].to(self.device)
                            else:
                                outputs = outputs[:, -self.args.pred_len:-self.args.pred_len + self.args.subset_prediction, f_dim:]
                                batch_y_subset = batch_y_subset[:, -self.args.pred_len:-self.args.pred_len + self.args.subset_prediction, f_dim:].to(self.device)

                            loss = criterion(outputs, batch_y_subset)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_subset = batch_y_subset[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y_subset)
                        train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe  = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            wandb.log({"train_loss": train_loss , "vali_loss": vali_loss, "test_loss": test_loss , "vali_mae": vali_mae, "vali_mse": vali_mse, "vali_rmse": vali_rmse, "vali_mape": vali_mape, "vali_mspe": vali_mspe, "test_mae": test_mae, "test_mse": test_mse, "test_rmse": test_rmse, "test_mape": test_mape, "test_mspe": test_mspe})
            print(f"train_loss: {train_loss} vali_loss: {vali_loss} test_loss: {test_loss} vali_mae: {vali_mae} vali_mse: {vali_mse} vali_rmse: {vali_rmse} vali_mape: {vali_mape} vali_mspe: {vali_mspe} test_mae: {test_mae} test_mse: {test_mse} test_rmse: {test_rmse} test_mape: {test_mape} test_mspe: {test_mspe}")

            if early_stopping.early_stop:
                print("Early stopping")
                break

            #No scheduler
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)
  
            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if self.args.load:
                self.model.load_state_dict(torch.load(self.args.load_path))
                print('model loaded from the path {}'.format(self.args.load_path))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
                print('model loaded from the path {}'.format(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                channel_size = batch_x.size(2)
                max_channel_size = self.args.max_channel
                number_of_iter = channel_size // max_channel_size + 1

                pred = None
                true = None

                for j in range(number_of_iter):
                    batch_x_subset = batch_x[:, :, j * max_channel_size:(j + 1) * max_channel_size]
                    batch_y_subset = batch_y[:, :, j * max_channel_size:(j + 1) * max_channel_size]

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y_subset[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_subset[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_subset = batch_y_subset[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y_subset = batch_y_subset.detach().cpu().numpy()

                    if pred is None:
                        pred = outputs
                        true = batch_y_subset
                    else:
                        pred = np.concatenate((pred, outputs), axis=2)
                        true = np.concatenate((true, batch_y_subset), axis=2)
                
                
                
                
                #true = batch_y.detach().cpu().numpy()
                if true.shape != pred.shape:
                    print(true.shape, pred.shape)
                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                    true = test_data.inverse_transform(true.squeeze(0)).reshape(shape)

                
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        if self.args.load:
            f.write("fine-tuned-" + setting + "  \n")
        else:
            f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return