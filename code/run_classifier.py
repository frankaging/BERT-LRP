import argparse

from util.train_helper import *
from torch.nn import CrossEntropyLoss, BCELoss, Sigmoid

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main(args):

    device, n_gpu, output_log_file= system_setups(args)

    # data loader, we load the model and corresponding training and testing sets
    model, optimizer, train_dataloader, test_dataloader = \
        data_and_model_loader(device, n_gpu, args)

    # main training step    
    global_step = 0
    global_best_acc = -1
    epoch=0
    evaluate_interval = 5
    # training epoch to eval
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        # train a teacher model solving this task
        global_step, global_best_acc = \
            step_train(train_dataloader, test_dataloader, model, optimizer, 
                        device, n_gpu, evaluate_interval, global_step, 
                        output_log_file, epoch, global_best_acc, args)
        if global_step == -1:
            logger.info("***** Early Stop with patient count *****")
            break
        epoch += 1

    logger.info("***** Global best performance *****")
    logger.info("accuracy on dev set: " + str(global_best_acc))
    
if __name__ == "__main__":

    from util.args_parser import parser
    args = parser.parse_args()
    main(args)