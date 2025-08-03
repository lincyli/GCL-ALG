import os
import random
import sys
import argparse
import time
import logging
import shutil
import glob
from torch_geometric.loader import DataLoader
from unsupervised.us_gin import Encoder
from unsupervised.us_evaluate_embedding import evaluate_embedding
from unsupervised.us_model import *

sys.path.append(os.path.abspath(os.path.join('..')))
from datasets import get_dataset

from view_generator import ViewGenerator,GIN_NodeWeightEncoder,mlp_edge_encoder


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--dataset', help='Dataset', default='IMDB-BINARY')
    parser.add_argument('--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
    # parser.add_argument('--decay', dest='lr decay', type=float, default=0, help='Learning rate.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128, help='')
    parser.add_argument('--mlp_dim', dest='mlp_dim', type=int, default=128, help='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=str, default='cl_exp', help='')
    parser.add_argument('--save', type=str, default='debug', help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        os.mkdir(os.path.join(path, 'model'))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class simclr(nn.Module):
    def __init__(self, dataset, hidden_dim, num_gc_layers, prior, alpha=0.5, beta=1., gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers

        self.encoder = Encoder(dataset.num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)  
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean() 

    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss


def train_cl(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device):
    loss_all = 0
    model.train()
    total_graphs = 0
    for data in data_loader:

        view_optimizer.zero_grad()
        optimizer.zero_grad()
        data = data.to(device)

        sample1, view1 = view_gen1(data, True)
        sample2, view2 = view_gen2(data, True)

        out1 = model(view1)
        out2 = model(view2)

        loss = loss_cl(out1, out2)

        loss_all += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

        loss.backward()
        optimizer.step()
        view_optimizer.step()

    loss_all /= total_graphs

    return loss_all


def eval_acc(model, data_loader, device):
    model.eval()
    emb, y = model.encoder.get_embeddings(data_loader, device)  
    acc, std = evaluate_embedding(emb, y)  
    return acc, std


def cl_exp(args):
    set_seed(args.seed)

    joint_log_name = 'joint_log_{}.txt'.format(args.save)
    save_name = args.save
    args.save = '{}-{}-{}-{}'.format(args.dataset, args.seed, args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('unsupervised_exp', args.exp, save_name, args.dataset, args.save)
    create_exp_dir(args.save, glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.info(args)

    device_id = 'cuda:%d' % (args.gpu)
    device = torch.device(device_id if torch.cuda.is_available() else 'cpu')


    epochs = args.epochs
    log_interval = 10
    batch_size = args.batch_size
    lr = args.lr

    dataset = get_dataset(args.dataset, sparse=True, feat_str='deg+odeg100', root='../data')
    dataset = dataset.shuffle()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_eval_loader = DataLoader(dataset, batch_size=batch_size)

    model = simclr(dataset, args.hidden_dim, args.num_gc_layers, args.prior)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    view_gen1 = ViewGenerator(dataset, args.hidden_dim, args.mlp_dim, GIN_NodeWeightEncoder, mlp_edge_encoder)
    view_gen2 = ViewGenerator(dataset, args.hidden_dim, args.mlp_dim, GIN_NodeWeightEncoder, mlp_edge_encoder)


    view_gen1 = view_gen1.to(device)
    view_gen2 = view_gen2.to(device)

    view_optimizer = optim.Adam([{'params': view_gen1.parameters()},
                                 {'params': view_gen2.parameters()}], lr=args.lr
                                , weight_decay=0)

    logger.info('================')
    logger.info('lr: {}'.format(lr))
    logger.info('num_features: {}'.format(dataset.num_features))
    logger.info('hidden_dim: {}'.format(args.hidden_dim))
    logger.info('num_gc_layers: {}'.format(args.num_gc_layers))
    logger.info('================')

    best_test_acc = 0
    best_test_std = 0
    test_accs = []
    best_epoch = 0



    for epoch in range(1, epochs + 1):

        train_loss = train_cl(view_gen1, view_gen2, view_optimizer, model, optimizer, data_loader, device)
        logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))
        if epoch % log_interval == 0:
            test_acc, test_std = eval_acc(model, data_eval_loader, device)
            test_accs.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_std = test_std
                best_epoch = epoch
            logger.info("*" * 50)
            logger.info("Evaluating embedding...")
            logger.info('Epoch: {}, Test Acc: {:.2f} ± {:.2f}'.format(epoch, test_acc * 100, test_std * 100))

        torch.cuda.empty_cache()

    logger.info('Best Test Acc: {:.2f} ± {:.2f}'.format(best_test_acc * 100, best_test_std * 100))
    joint_log_dir = os.path.join('unsupervised_exp', args.exp, save_name)
    joint_log_path = os.path.join(joint_log_dir, joint_log_name)
    print(best_epoch)
    with open(joint_log_path, 'a+') as f:
        f.write('{},{},{:.2f},{:.2f}\n'.format(args.dataset, args.seed, best_test_acc * 100, best_test_std * 100))


if __name__ == '__main__':
    args = arg_parse()
    cl_exp(args)

