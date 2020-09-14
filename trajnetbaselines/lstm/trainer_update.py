"""Command line tool to train an LSTM model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle
import torch
import numpy as np


import trajnetplusplustools

from .. import augmentation
from .loss import PredictionLoss, L2Loss
from .lstm_update import LSTM, LSTMPredictor, drop_distant
from .gridbased_pooling import GridBasedPooling
from .non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from .non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from .more_non_gridbased_pooling import NMMP
from .modules import plot_error

from .. import __version__ as VERSION

from .utils import center_scene, random_rotation


class Trainer(object):
    def __init__(self, goal_model, traj_model, goal_optimizer=None, goal_lr_scheduler=None,
                 traj_optimizer=None, traj_lr_scheduler=None,
                 criterion='L2', device=None, batch_size=32, obs_length=9, pred_length=12, augment=False,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False):

        # setting for goal_model
        self.goal_model = goal_model if goal_model is not None else LSTM()
        self.goal_optimizer = goal_optimizer if goal_optimizer is not None else torch.optim.SGD(
            self.goal_model.parameters(), lr=3e-2, momentum=0.9)
        self.goal_lr_scheduler = (goal_lr_scheduler
                                  if goal_lr_scheduler is not None
                                  else torch.optim.lr_scheduler.StepLR(self.goal_optimizer, 15))

        # setting for traj_model
        self.traj_model = traj_model if traj_model is not None else LSTM()
        self.traj_optimizer = traj_optimizer if traj_optimizer is not None else torch.optim.SGD(
            self.traj_model.parameters(), lr=3e-4, momentum=0.9)
        self.traj_lr_scheduler = (traj_lr_scheduler
                                  if traj_lr_scheduler is not None
                                  else torch.optim.lr_scheduler.StepLR(self.traj_optimizer, 15))

        if criterion == 'L2':
            self.criterion = L2Loss()
            self.loss_multiplier = 100
        else:
            self.criterion = PredictionLoss()
            self.loss_multiplier = 1

        self.device = device if device is not None else torch.device('cpu')
        self.goal_model = self.goal_model.to(self.device)
        self.traj_model = self.traj_model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length + self.pred_length

        self.augment = augment
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

    def loop(self, train_scenes, val_scenes, train_goals, val_goals, out, epochs=35, train_goals_epochs=3, \
             start_epoch=0, model_flag='use_goals_pred_goals'):

        # if model_flag == 'use_goals_pred_goals', i.e., need to train the goal model and trajectory model,
        # then first train the goal model, and use the pred_goals to train the trajectory model.
        # Otherwise only train the trajectory model with goals ground truth or zero value（"not_use_goals").
        epoch_losses = []
        if model_flag == 'use_goals_pred_goals':
            # if need to pred_goals, first train the goal model which is indicated by pred_flag
            for epoch in range(start_epoch, start_epoch + train_goals_epochs):
                self.train(train_scenes, train_goals, epoch, model_flag, pred_flag="pred_goals")
        # train the trajectory models and save the model state at each epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'traj_state_dict': self.traj_model.state_dict(),
                         'traj_optimizer': self.traj_optimizer.state_dict(),
                         'traj_scheduler': self.traj_lr_scheduler.state_dict()}
                LSTMPredictor(self.traj_model).save(state, out + '.epoch{}'.format(epoch))  #
            epoch_loss = self.train(train_scenes, train_goals, epoch, model_flag, pred_flag="pred_trajs")
            # self.val(val_scenes, val_goals, epoch)
            epoch_losses.append(epoch_loss)

        # save the final trajectory model
        state = {'epoch': epoch + 1, 'state_dict': self.traj_model.state_dict(),
                 'optimizer': self.traj_optimizer.state_dict(),
                 'scheduler': self.traj_lr_scheduler.state_dict()}
        LSTMPredictor(self.traj_model).save(state, out + '.epoch{}'.format(epoch + 1))
        LSTMPredictor(self.traj_model).save(state, out)

        # only return trajectory training loss
        return epoch_losses

    def get_lr(self, pred_flag):
        model_optimizer = self.traj_optimizer if pred_flag == "pred_trajs" else self.goal_optimizer
        for param_group in model_optimizer.param_groups:
            return param_group['lr']

    def train(self, scenes, goals, epoch, model_flag, pred_flag):
        start_time = time.time()

        print('epoch', epoch)
        self.goal_lr_scheduler.step()
        self.traj_lr_scheduler.step()

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.goal_model.train()
        self.traj_model.train()
        self.goal_optimizer.zero_grad()
        self.traj_optimizer.zero_grad()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
                # scene = augmentation.add_noise(scene, thresh=0.01)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)

                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                ## Train Batch
                loss = self.train_goal_batch(batch_scene, batch_scene_goal, batch_split, model_flag, pred_flag)

                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

            if (scene_i + 1) % (10 * self.batch_size) == 0:
                self.log.info({
                    'model': model_flag,
                    'type': 'train_pred_goal' if pred_flag == "pred_goals" else "train_pred_trajs",
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(pred_flag),
                    'loss': round(loss, 3),
                })

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })

        return round(epoch_loss / (len(scenes)), 5)

    def val(self, scenes, goals, epoch):
        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.train()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)

                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                loss_val_batch, loss_test_batch = self.val_batch(batch_scene, batch_scene_goal, batch_split)
                val_loss += loss_val_batch
                test_loss += loss_test_batch

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            'test_loss': round(test_loss / len(scenes), 3),
            'time': round(eval_time, 1),
        })

    def train_goal_batch(self, batch_scene, batch_scene_goal, batch_split, model_flag, pred_flag):
        """Training of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene
        model_flag: String ["use_goals_pred_goals, "not_use_goals", "use_goals_pred_trajs"]
            String defining the type of model structure.
            Inherited from the main function
        pred_flag: String ["pred_goals", "pred_trajs"]
            String defining the type of submodel to train
            "pred_goals" and "pred_trajs" represent the training of goal and trajectory models respectively

        Returns
        -------
        loss : scalar
            Training loss of the batch
        """

        ## If observation dropout active
        if self.obs_dropout:
            self.start_length = random.randint(0, self.obs_length - 2)

        observed = batch_scene[self.start_length:self.obs_length].clone()
        prediction_truth = batch_scene[self.obs_length:self.seq_length - 1].clone()
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length - 1:self.seq_length - 1]

        # if need to train both pred_goals and pred_trajs models
        if model_flag == "use_goals_pred_goals":
            # predict goals coordinates of main pedestrian of each track
            pred_goals = self.goal_model(observed, batch_split, "pred_goals")
            pred_goals = pred_goals.requires_grad_()
            batch_scene_goal = batch_scene_goal.requires_grad_()

            # L2 Loss between predicted goals and real goals
            pred_goal_loss = torch.nn.MSELoss()
            pred_goal_loss = pred_goal_loss(pred_goals, batch_scene_goal)

            # first only train the pred_goal model and update it
            if pred_flag == "pred_goals":
                self.goal_optimizer.zero_grad()
                pred_goal_loss.backward()
                self.goal_optimizer.step()
                return pred_goal_loss.item()

            # train the pred_trajs model after finish training the pred_goals model
            elif pred_flag == "pred_trajs":
                # Input predicted goals to the pred_trajs model to help forecasting
                rel_outputs, outputs = self.traj_model(observed, batch_split, "pred_outputs", pred_goals,
                                                       prediction_truth)
                # L2 Loss between predicted trajectories and real trajectories
                pred_outputs_loss = self.criterion(rel_outputs[-self.pred_length:], targets, batch_split) \
                                    * self.batch_size * self.loss_multiplier
                loss = pred_outputs_loss

        # if only need to train pred_trajs models
        if model_flag == "not_use_goals" or "use_goal_pred_trajs":
            # "not_use_goals": train the trajectory model without using goals but a zero Tensor instead
            # "use_goal_pred_trajs": train the trajectory model with goals coordinates ground truth 'batch_scene_goal'
            rel_outputs, outputs = self.traj_model(observed, batch_split, "pred_outputs", batch_scene_goal,
                                                   prediction_truth)
            # trajectory loss
            loss = self.criterion(rel_outputs[-self.pred_length:], targets,
                                  batch_split) * self.batch_size * self.loss_multiplier

        # train the trajectory model and update it
        self.traj_optimizer.zero_grad()
        loss.backward()
        self.traj_optimizer.step()

        # return the pred_trajs loss of the batch
        return loss.item()

    def val_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Validation of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Validation loss of the batch when groundtruth of neighbours
            is provided
        loss_test : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided (evaluation scenario)
        """

        if self.obs_dropout:
            self.start_length = 0

        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:self.seq_length - 1].clone()  ## CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length - 1:self.seq_length - 1]
        observed_test = observed.clone()

        with torch.no_grad():
            ## groundtruth of neighbours provided (Better validation curve to monitor model)
            rel_outputs, _ = self.model(observed, batch_scene_goal, batch_split, prediction_truth)
            loss = self.criterion(rel_outputs[-self.pred_length:], targets,
                                  batch_split) * self.batch_size * self.loss_multiplier

            ## groundtruth of neighbours not provided
            rel_outputs_test, _ = self.model(observed_test, batch_scene_goal, batch_split, n_predict=self.pred_length)
            loss_test = self.criterion(rel_outputs_test[-self.pred_length:], targets,
                                       batch_split) * self.batch_size * self.loss_multiplier

        return loss.item(), loss_test.item()


def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file + '.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None


def main(model_flag, epochs=4):
    """

    Parameters
    ----------
    model_flag: String ["use_goals_pred_goals", "not_use_goals", "use_goals_pred_trajs"]
        "use_goals_pred_goals": generate data with goals coordinates but only used for training the pred_goal model
        "not_use_goals": generate data without goals, i.e., adding zero tensor as goals for training the pred_trajs model
        "use_goals_pred_trajs": generate data with real goals coordinates to train the pred_trajs model
    epochs: Int [0, inf)
        Determines the number of epochs of training pred_trajs models

    Returns
    -------
    loss_history: List
        List of losses at every epoch
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--step_size', default=10, type=int,
                        help='step_size of lr scheduler')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--goal_lr', default=1e-2, type=float,
                        help='initial goal model learning rate')
    parser.add_argument('--train_goals_epochs', default=3, type=int,
                        help='number of epochs of training the pred_goals model')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp'),
                        help='type of interaction encoder')
    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize the scene along direction of movement')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes (rotation augmentation)')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of oservation')
    parser.add_argument('--path', default='synth_data',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2', choices=('L2', 'pred'),
                        help='loss objective to train the model')
    parser.add_argument('--goals', action='store_true',
                        help='flag to use goals')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ##Pretrain Pooling AE
    pretrain.add_argument('--load_pretrained_pool_path', default=None,
                          help='load a pickled model state dictionary of pool AE before training')
    pretrain.add_argument('--pretrained_pool_arch', default='onelayer',
                          help='architecture of pool representation')
    pretrain.add_argument('--downscale', type=int, default=4,
                          help='downscale factor of pooling grid')
    pretrain.add_argument('--finetune', type=int, default=0,
                          help='finetune factor of pretrained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims (for gridbased pooling)')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of pooling/interaction vector')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attentionmlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attentionmlp vel dimension')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background value of gridbased pooling')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider velocity in nn')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='number of neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iters in NMMP')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout (regularization)')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='start length during obs dropout')
    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    args.output = 'OUTPUT_BLOCK/{}/lstm_{}_{}.pkl'.format(args.path, model_flag, args.output)


    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    # loading a previously saved model
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    ## pretrained pool model (if any)
    pretrained_pool = None

    # create interaction/pooling modules
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                     mlp_dim_vel=args.vel_dim)
    elif args.type == 'nmmp':
        pool = NMMP(hidden_dim=args.hidden_dim, out_dim=args.pool_dim, k=args.mp_iters)
    elif args.type == 'attentionmlp':
        pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                   mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
    elif args.type == 'directionalmlp':
        pool = DirectionalMLPPooling(out_dim=args.pool_dim)
    elif args.type == 'nn':
        pool = NN_Pooling(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
    elif args.type == 'nn_lstm':
        pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 'traj_pool':
        pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att_fast':
        pool = SAttention_fast(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type != 'vanilla':
        pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
                                cell_side=args.cell_side, n=args.n, front=args.front,
                                out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
                                constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
                                norm=args.norm, layer_dims=args.layer_dims)

    args.path = 'DATA_BLOCK/' + args.path
    ## Prepare data
    # without using goal coordinates
    if model_flag == "not_use_goals":
        train_scenes, train_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
    # with using goal coordinates (for pred_goals model or pred_trajs model)
    else:
        train_scenes, train_goals = prepare_data(args.path, subset='/train/', sample=args.sample)
    val_scenes, val_goals = prepare_data(args.path, subset='/val/', sample=args.sample, goals=args.goals)

    # create goal forecasting model: pred_goals model
    goal_model = LSTM(pool=pool,
                      embedding_dim=args.coordinate_embedding_dim,
                      hidden_dim=args.hidden_dim,
                      goal_dim=args.goal_dim)

    # create trajectory forecasting model: pred_trajs model
    traj_model = LSTM(pool=pool,
                      embedding_dim=args.coordinate_embedding_dim,
                      hidden_dim=args.hidden_dim,
                      goal_flag=True,
                      goal_dim=args.goal_dim)

    # optimizers and schedulars for two models
    goal_optimizer = torch.optim.Adam(goal_model.parameters(), lr=args.goal_lr, weight_decay=1e-4)
    traj_optimizer = torch.optim.Adam(traj_model.parameters(), lr=args.lr, weight_decay=1e-4)
    goal_lr_scheduler, traj_lr_scheduler = None, None
    if args.step_size is not None:
        goal_lr_scheduler = torch.optim.lr_scheduler.StepLR(goal_optimizer, args.step_size)
        traj_lr_scheduler = torch.optim.lr_scheduler.StepLR(traj_optimizer, args.step_size)

    start_epoch = 0

    # train
    if args.load_state:
        # load pretrained model.
        # useful for tranfer learning
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        traj_model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        if args.load_full_state:
            # load optimizers from last training
            # useful to continue model training
            print("Loading Optimizer Dict")
            optimizer = torch.optim.Adam(traj_model.parameters(), lr=args.lr)  # , weight_decay=1e-4
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    # trainer
    trainer = Trainer(goal_model=goal_model, traj_model=traj_model, goal_optimizer=goal_optimizer,
                      goal_lr_scheduler=goal_lr_scheduler, traj_optimizer=traj_optimizer,
                      traj_lr_scheduler=traj_lr_scheduler, criterion=args.loss, device=args.device,
                      batch_size=args.batch_size, obs_length=args.obs_length, pred_length=args.pred_length,
                      augment=args.augment, normalize_scene=args.normalize_scene, save_every=args.save_every,
                      start_length=args.start_length, obs_dropout=args.obs_dropout)

    # record the train loss
    loss_history = trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, args.epochs,
                                args.train_goals_epochs, start_epoch, model_flag)

    np.save('loss_record/' + model_flag + '_loss.npy', np.array(loss_history))

    return loss_history


if __name__ == '__main__':
    # generate data with goals to train the pred_goal model (M2)
    use_goals_pred_goal_loss = main("use_goals_pred_goals")
    # generate data without goals and train the pred_traj model (M1)
    not_use_goals_loss = main("not_use_goals")
    # generate data with goals to train the pred_traj model (M0)
    use_goals_pred_trajs_loss = main("use_goals_pred_trajs")

    # plot the training loss of two models
    plot_error(use_goals_pred_goal_loss, not_use_goals_loss, use_goals_pred_trajs_loss, 'E2: use_goals_pred_goals',
               'E1: not_use_goals', 'E0: use_goals_pred_trajs')
