"""utils"""
import os
import numpy as np
import torch
import time
import matplotlib
import matplotlib.pyplot as plt
from utils import print_log
from data import minmax_denormalize, log_transform, log_detransform


def _calculate_error(label, prediction):
    """calculate l2-error to evaluate accuracy"""
    rel_error = np.sqrt(np.sum(np.square(label.flatten() - prediction.flatten()))) / \
                (np.sqrt(np.sum(np.square(label.flatten()))))
    return rel_error



def calculate_l2_error(model, inputs, param, label):
    """
    Evaluate the model sequentially.

    Args:
        model (nn.Module): Prediction network.
        inputs (numpy.Array): Input data.             
        param: (numpy.Array): Parameter data.          
        label (numpy.Array): Label data.              
    """
    print_log("================================Start Evaluation================================")
    time_beg = time.time()
         
    bsize = 1   
    n_batches = inputs.shape[0] // bsize

    sample_shape = (bsize, 5, inputs.shape[-2], inputs.shape[-1])

    y_pred =  []
    for i in range(n_batches):
        # test_label = label[i:i + bsize]
        test_batch = inputs[i:i + bsize].reshape(sample_shape)
        test_param = param[i:i + bsize].reshape(bsize, param.shape[-1])
               
        prediction = model((test_batch, test_param,))     # [1, 1, 70, 70]
        prediction_np = prediction.detach().cpu().numpy()
        y_pred.append(prediction_np)

    pred_np = np.concatenate(y_pred, axis=0)
    label_np = label.detach().cpu().numpy()

    rel_rmse_error = _calculate_error(label_np, pred_np)
        
    print_log(f"Test loss: {rel_rmse_error}")
    print_log("=================================End Evaluation=================================")
    print_log("predict total time: {} s".format(time.time() - time_beg))


def visual_static(path, steps, x, yy, yp):
    """ Plot static figures during model training.

    Args:
        path (str): save path of results.
        steps (int): Training step.
        x (Array): Input data.
        yy (Array): Label data.
        yp (Array): Predicted data.
    """
    cmap = matplotlib.colormaps['jet']
    # vmin, vmax = np.min(yy), np.max(yy)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.title(f"Input")
    plt.imshow(x, cmap=cmap, aspect=70/1000)
    plt.axis('off')
    # plt.colorbar()


    plt.subplot(1, 4, 2)
    plt.title(f"Label")
    plt.imshow(yy, cmap=cmap)
    plt.axis('off')
    # plt.colorbar()


    plt.subplot(1, 4, 3)
    plt.title(f"Predict")
    plt.imshow(yp, cmap=cmap)
    plt.axis('off')
    # plt.colorbar()


    plt.subplot(1, 4, 4)
    plt.title(f"Error")
    plt.imshow(np.abs(yy - yp), cmap=cmap)
    plt.axis('off')
    # plt.colorbar()

    plt.tight_layout()

    plt.savefig(os.path.join(path, 'result-' + str(steps) +'.jpg'))
    plt.close()


def visual(path, steps, model, inputs, param, labels):
    """ Infer the model sequentially and visualize the results during model training.

    Args:
        Prediction network.
        path (str): save path of results.
        steps (int): Training step.
        model (torch.nn.Module): Torch model.
        inputs (numpy.array): Input data.               
        param (numpy.array): Parameter data.            
        labels (numpy.array): Label data.               
    """

    image_path = os.path.join(path, 'images')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    test_batch = inputs[0].reshape((1, 5, inputs.shape[-2], inputs.shape[-1]))
    test_param = param[0].reshape((1, param.shape[-1]))
    labels = labels[0].reshape(70, 70)
    
    prediction = model((test_batch, test_param,))
    
    test_batch = test_batch.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    inp = test_batch[0][0].reshape(1000,70)
    pred = prediction_np.reshape(70, 70)
    labels_np = labels_np.reshape(70, 70)
    visual_static(image_path, steps, inp, labels_np, pred)



def plot_loss(x, y, title=None):

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}

    plt.plot(x, y)
    plt.semilogy()
    # plt.grid(True)
    plt.legend(loc="upper right", prop=font)
    plt.xlabel('f', font)
    plt.ylabel('rel. l2 error', font)
    plt.yticks(fontproperties='Times New Roman', size=font["size"])
    plt.xticks(fontproperties='Times New Roman', size=font["size"])
    plt.title(title, font)
    plt.show()





