import matplotlib.pyplot as plt
import numpy as np

def loss_plot(history_dict, epochs=20):
    x = np.arange(1,epochs+1)
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    plt.title('Train loss & validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, train_loss, 'o', label='Train loss')
    plt.plot(x, val_loss, label='Validation loss')
    plt.legend()
    plt.show()
    
def acc_plot(history_dict, epochs=20):
    x = np.arange(1,epochs+1)
    if 'binary_accuracy' in history_dict.keys():
        train_acc = history_dict['binary_accuracy']
    else:
        train_acc = history_dict['accuracy']
    if 'val_binary_accuracy' in history_dict.keys():
        val_acc = history_dict['val_binary_accuracy']
    else:
        val_acc = history_dict['val_accuracy']

    plt.title('Train accuracy & validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuarcy')
    plt.plot(x, train_acc, 'o', label="Train accuracy")
    plt.plot(x, val_acc, label="Validation accuracy")
    plt.legend()
    plt.show()
    


# If GPU ran out of memory in jupyter:
# from keras import backend as K
# K.clear_session()
