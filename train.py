import os
import time
import datetime
import argparse
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from data import SeismicData
from model import HIWaveletDeepONet
from utils import load_yaml_config, log_config, print_log, log_timer, visual, calculate_l2_error


# set_seed(0)
# np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='FWI problem')
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/fwi_fvb.yaml")
    parser.add_argument("--work_name", type=str, default="fvb-f")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--iter", type=int, default=0, help="iteration of checkpoint")
    input_args = parser.parse_args()
    return input_args


@log_timer 
def train(input_args):
    '''train and evaluate the network'''

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    batch_size = data_params['batch_size']
    root_path = data_params['root_dir']
    dataset = data_params['name']
    task = data_params['task']
    data_path = os.path.join(root_path, dataset)

    train_dataset = SeismicData(dataset=dataset, task=task, data_path=data_path)
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                            pin_memory=False, num_workers=0)


    test_input, test_param, test_label = train_dataset.get_testdata()

     model = HIWaveletDeepONet(num_parameter=test_param.shape[-1],
                             level=int(model_params["level"]),
                             size=[72, 1000],
                             wavelet=model_params["wavelet"],
                             width=int(model_params["hidden_channels"]),
                             modes1=int(model_params["modes"]),
                             modes2=int(model_params["modes"]))

    model.cuda()



    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'], weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, float(optimizer_params["gamma"]))
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    ckpt_dir = os.path.join('./results', input_args.work_name, summary_params["ckpt_dir"])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    total_steps = 0
    start_epoch = 0
    objs_to_save={}

    if input_args.load_checkpoint:     
        iters = int(input_args.iter)
        assert(os.path.isdir(ckpt_dir))

        print('Loading checkpoints')
        model = torch.load(os.path.join(ckpt_dir, f"{model_params['model'] + model_params['name']}_{iters}.pth"))

        # load optimizers
        try:
            optim_dict = torch.load(ckpt_dir + f'optim_{iters:06d}.pth')
            optimizer.load_state_dict(optim_dict['optimizer_state_dict'])
            total_steps = optim_dict['total_steps']
            start_epoch = optim_dict['epoch']

        except FileNotFoundError:
            print('Unable to load optimizer checkpoints')

    with tqdm(total=len(dataloader) * optimizer_params["epochs"]) as pbar:
        pbar.update(total_steps)
        train_losses = []

        for epoch in range(start_epoch, optimizer_params["epochs"]):
            local_time_beg = time.time()

            model.train()
            for i, (train_inputs, train_param, train_label) in enumerate(dataloader):
                start_time = time.time()
                
                if isinstance(train_inputs, torch.Tensor):
                    train_inputs = train_inputs.cuda()
                if isinstance(train_label, torch.Tensor):
                    train_label = train_label.cuda()
                if isinstance(train_param, torch.Tensor):
                    train_param = train_param.cuda()

                pred = model((train_inputs, train_param))
                train_loss = criterion(pred, train_label)
                
                train_losses.append([train_loss.item()])

                optimizer.zero_grad()
                train_loss.backward()

                optimizer.step()
                scheduler.step()

                pbar.update(1)
                total_steps += 1

                if not total_steps % summary_params["print_step_interval"]:
                    tqdm.write("Epoch %d,  step %d,  train loss %0.6f,  iteration time %0.3f" % (epoch, total_steps, train_loss.item(), time.time() - start_time))
            

             epoch_seconds = time.time() - local_time_beg
             steps_per_epoch = train_dataset // batch_size
             step_seconds = (epoch_seconds / steps_per_epoch) * 1000
             print_log(f"epoch: {epoch} train loss: {train_loss:.5f} "
                     f"epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}ms")

            # save checkpoint
            if epoch > 0 and epoch % summary_params["save_ckpt_interval"] == 0:
                torch.save(model.state_dict(), 
                           os.path.join(ckpt_dir, f"{model_params['model'] + model_params['name']}_{epoch}.pth"))

                np.savetxt(os.path.join(ckpt_dir, 'train_losses_%06d.txt' % total_steps),
                           np.array(train_losses))
                save_dict = {'epoch': epoch,
                             'total_steps': total_steps,
                             'optimizer_state_dict': optimizer.state_dict()}
                save_dict.update(objs_to_save)
                torch.save(save_dict, os.path.join(ckpt_dir, 'optim_%06d.pth' % total_steps))

    # save model
    torch.save(model.state_dict(), 
                os.path.join(ckpt_dir, f"{model_params['model'] + model_params['name']}_{optimizer_params['epochs']}.pth"))

    np.savetxt(os.path.join(ckpt_dir, 'train_losses_%06d.txt' % total_steps),
                np.array(train_losses))
    save_dict = {'epoch': optimizer_params["epochs"],
                    'total_steps': total_steps,
                    'optimizer_state_dict': optimizer.state_dict()}
    save_dict.update(objs_to_save)
    torch.save(save_dict, os.path.join(ckpt_dir, 'optim_%06d.pth' % total_steps))
    

    save_dir = os.path.join('./results', input_args.work_name)
    visual(save_dir, total_steps, model, test_input, test_param, test_label)

if __name__ == '__main__':
    log_config('./logs', 'fwi-fvb')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())
    args = parse_args()

    train(args)

