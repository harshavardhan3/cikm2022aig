import os
import torch
from copy import deepcopy
import numpy as np
from federatedscope.gfl.loss.vat import VATLoss
from federatedscope.core.trainers import GeneralTorchTrainer


class FLITTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
    def register_default_hooks_train(self):
        super(FLITTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=record_initialization_global,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_global,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FLITTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=record_initialization_global,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_global,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        ctx.global_model.to(ctx.device)
        predG = ctx.global_model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLIT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss = lossLocalLabel + torch.relu(lossLocalLabel -
                                                 lossGlobalLabel.detach())
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()
        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )
    def save_prediction(self, path, client_id, task_type):
        y_inds, y_probs = self.ctx.test_y_inds, self.ctx.test_y_prob
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = np.argmax(y_probs, axis=-1) if 'classification' in task_type.lower() else y_probs

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in task_type.lower():
                    line = [client_id, y_ind] + [y_pred]
                else:
                    line = [client_id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')

class FLITPlusTrainer(FLITTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
    def _hook_on_batch_forward(self, ctx):
        # LDS should be calculated before the forward for cross entropy
        batch = ctx.data_batch.to(ctx.device)
        ctx.global_model.to(ctx.device)
        if ctx.cur_mode == 'test':
            lossLocalVAT, lossGlobalVAT = torch.tensor(0.), torch.tensor(0.)
        else:
            vat_loss = VATLoss()  # xi, and eps
            lossLocalVAT = vat_loss(deepcopy(ctx.model), batch,
                                    deepcopy(ctx.criterion))
            lossGlobalVAT = vat_loss(deepcopy(ctx.global_model), batch,
                                     deepcopy(ctx.criterion))

        pred = ctx.model(batch)
        predG = ctx.global_model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLITPLUS trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        lossGlobalLabel = ctx.criterion(predG, label)
        lossLocalLabel = ctx.criterion(pred, label)

        weightloss_loss = lossLocalLabel + torch.relu(lossLocalLabel -
                                                      lossGlobalLabel.detach())
        weightloss_vat = (lossLocalVAT +
                          torch.relu(lossLocalVAT - lossGlobalVAT.detach()))
        weightloss = self.cfg.flitplus.lambdavat * \
            weightloss_vat + weightloss_loss
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (
                    lossLocalLabel +
                    self.cfg.flitplus.weightReg * lossLocalVAT)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )
    def save_prediction(self, path, client_id, task_type):
        y_inds, y_probs = self.ctx.test_y_inds, self.ctx.test_y_prob
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = np.argmax(y_probs, axis=-1) if 'classification' in task_type.lower() else y_probs

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in task_type.lower():
                    line = [client_id, y_ind] + [y_pred]
                else:
                    line = [client_id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')

class FedFocalTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
    def register_default_hooks_train(self):
        super(FedFocalTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FedFocalTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FLIT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        lossLocalLabel = ctx.criterion(pred, label)
        weightloss = lossLocalLabel
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (lossLocalLabel)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )


class FedVATTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
    def register_default_hooks_train(self):
        super(FedVATTrainer, self).register_default_hooks_train()
        self.register_hook_in_train(new_hook=record_initialization_local,
                                    trigger='on_fit_start',
                                    insert_pos=-1)
        self.register_hook_in_train(new_hook=del_initialization_local,
                                    trigger='on_fit_end',
                                    insert_pos=-1)

    def register_default_hooks_eval(self):
        super(FedVATTrainer, self).register_default_hooks_eval()
        self.register_hook_in_eval(new_hook=record_initialization_local,
                                   trigger='on_fit_start',
                                   insert_pos=-1)
        self.register_hook_in_eval(new_hook=del_initialization_local,
                                   trigger='on_fit_end',
                                   insert_pos=-1)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        if ctx.cur_mode == 'test':
            lossLocalVAT = torch.tensor(0.)
        else:
            vat_loss = VATLoss()  # xi, and eps
            lossLocalVAT = vat_loss(deepcopy(ctx.model), batch,
                                    deepcopy(ctx.criterion))

        pred = ctx.model(batch)
        if ctx.criterion._get_name() == 'CrossEntropyLoss':
            label = batch.y.squeeze(-1).long()
        elif ctx.criterion._get_name() == 'MSELoss':
            label = batch.y.float()
        else:
            raise ValueError(
                f'FedVAT trainer not support {ctx.criterion._get_name()}.')
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        lossLocalLabel = ctx.criterion(pred, label)
        weightloss = lossLocalLabel + self.cfg.flitplus.lambdavat * \
            lossLocalVAT
        if ctx.weight_denomaitor is None:
            ctx.weight_denomaitor = weightloss.mean(dim=0,
                                                    keepdim=True).detach()
        else:
            ctx.weight_denomaitor = self.cfg.flitplus.factor_ema * \
                                    ctx.weight_denomaitor + (
                                            -self.cfg.flitplus.factor_ema +
                                            1) * weightloss.mean(
                                            keepdim=True, dim=0).detach()

        loss = (1 - torch.exp(-weightloss / (ctx.weight_denomaitor + 1e-7)) +
                1e-7)**self.cfg.flitplus.tmpFed * (
                    lossLocalLabel +
                    self.cfg.flitplus.weightReg * lossLocalVAT)
        ctx.loss_batch = loss.mean()

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )
    def save_prediction(self, path, client_id, task_type):
        y_inds, y_probs = self.ctx.test_y_inds, self.ctx.test_y_prob
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        y_preds = np.argmax(y_probs, axis=-1) if 'classification' in task_type.lower() else y_probs

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in task_type.lower():
                    line = [client_id, y_ind] + [y_pred]
                else:
                    line = [client_id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')

def record_initialization_local(ctx):
    """Record weight denomaitor to cpu

    """
    ctx.weight_denomaitor = None


def del_initialization_local(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.weight_denomaitor = None


def record_initialization_global(ctx):
    """Record the shared global model to cpu

    """
    ctx.global_model = deepcopy(ctx.model)
    ctx.global_model.to(torch.device("cpu"))


def del_initialization_global(ctx):
    """Clear the variable to avoid memory leakage

    """
    ctx.global_model = None
