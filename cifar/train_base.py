from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging

import models
from data import *
from sklearn.metrics import precision_score, recall_score, f1_score     # to find out evaluation matric like precision, f1 etc
from thop import profile


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH', default='cifar10_resnet_110',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_110)')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=64000, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    # parser.add_argument('--iters', default=64000, type=int,
    #                     help='number of total iterations (default: 64,000)')
    # parser.add_argument('--start-iter', default=0, type=int,
    #                     help='manual iter number (useful on restarts)')

    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='SE',
                    help='Manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=170, type=int, metavar='N',
                    help='Number of total epochs to run')
    args = parser.parse_args()
    return args

def compute_metrics(target, output):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    target = target.cpu().numpy()

    precision = precision_score(target, pred, average='macro', zero_division=1)
    recall = recall_score(target, pred, average='macro', zero_division=1)
    f1 = f1_score(target, pred, average='macro', zero_division=1)

    return precision, recall, f1
def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)


    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    #model = torch.nn.DataParallel(model).cuda()
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0])


    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    import time

    total_train_time = 0
    end = time.time()
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):

        model.train()
        adjust_learning_rate(args, optimizer, epoch)

        data_time.reset()
        batch_time.reset()
        best_prec1 = 0


        for i, (input, target) in enumerate(train_loader):
        # measuring data loading time
            data_time.update(time.time() - end)

            # target = target.squeeze().long().cuda(non_blocking=True)
            target = target.squeeze().long().to(device, non_blocking=False)

            input_var = Variable(input).to(device)
            target_var = Variable(target).to(device)

        # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # print log
            if i % args.print_freq == 0:
                logging.info("Epoch: [{0}][{1}/{2}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                top1=top1)
                )

        epoch_train_time = time.time() - end
        total_train_time += epoch_train_time
    # After each epoch, you can perform validation
        prec1 = validate(args, test_loader, model, criterion)
        # print(prec1)
    # Update the best precision and save checkpoint
        is_best = prec1 > best_prec1
    
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = os.path.join(args.save_path, 'model_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path, 'checkpoint_latest.pth'))

    print("Total training time for {} epochs: {:.2f} minutes".format(args.epochs, total_train_time))

    #     from thop import profile
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
# Profile the model to calculate FLOPs
    flops, params = profile(model, inputs=(input_tensor,))
    total_flops = flops / 10 ** 6  
    total_params = params / 10 ** 6
    print(f"Total FLOPs: {total_flops:.2f} MFLOPs")
    print(f'Total parameters: {total_params:.2f} Params(M)')




    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")       
    # input_tensor = torch.randn(1, 3, 32, 32).to(device)

    # def profile(model, inputs=(input_tensor,)):
    #     flops = 0
    #     params = 0
    #     with torch.no_grad():
    #       for name, param in model.named_parameters():
    #         if param.requires_grad:
    #           flops += torch.numel(param) * param.grad.numel()
    #           params += torch.numel(param)
    #     return flops, params




def validate(args, test_loader, model, criterion):
  device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  flops_meter = AverageMeter()

  # switch to evaluation mode
  model.eval()
  end = time.time()

  for i, (input, target) in enumerate(test_loader):
    target = target.squeeze().long().to(device, non_blocking=False)
    with torch.no_grad():
      input_var = input.to(device)
      target_var = target.to(device)

      # compute output
      output = model(input_var)
      loss = criterion(output, target_var)

      # measure accuracy and record loss
      prec1, = accuracy(output.data, target, topk=(1,))
      top1.update(prec1, input.size(0))
      losses.update(loss.item(), input.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
        logging.info(
            'Test: [{}/{}]\t'
            'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
            'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
            'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                i, len(test_loader), batch_time=batch_time,
                loss=losses, top1=top1, flops=flops_meter  # print FLOPs
                # add flops_meter here
            )
        )

  # Compute FLOPs after all iterations
  



  # Profile the model
#   flops, params = profile(model, inputs=(input_tensor,))

#   # Convert the FLOPs and parameters to MFLOPs and MParams, respectively
#   total_flops = flops / 10 ** 6
#   total_params = params / 10 ** 6

  # Print the FLOP results
#   logging.info('Total FLOPs: %f MFLOPs', total_flops)
#   logging.info('Total parameters: %f MParams', total_params)

  logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
  precision, recall, f1 = compute_metrics(target, output)
  logging.info("Precision: %0.4f, Recall: %0.4f, F1 Score: %0.4f" % (precision, recall, f1))
  return top1.avg




def test_model(args):
    # create model
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = models.__dict__[args.arch](args.pretrained)
    # model = torch.nn.DataParallel(model).to(device)
    model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

        # Load the filtered state_dict into your model
            model.load_state_dict(filtered_state_dict)

            # model.load_state_dict(checkpoint['state_dict'])
            # logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
            #     args.resume, checkpoint['epoch']
            # ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().to(device)

    validate(args, test_loader, model, criterion)
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
# Profile the model to calculate FLOPs
    flops, params = profile(model, inputs=(input_tensor,))
    total_flops = flops / 10 ** 6  
    total_params = params / 10 ** 6
    print(f"Total FLOPs: {total_flops:.2f} MFLOPs")
    print(f'Total parameters: {total_params:.2f} Params(M)')



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter <= 170):
        lr = args.lr * (args.step_ratio ** 2)
    elif 32000 <= _iter < 48000:
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= 48000:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()