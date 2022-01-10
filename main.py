import numpy as np
from sklearn.utils import shuffle
from numpy import save
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow import keras
import pandas as pd
import util
import warnings
warnings.filterwarnings("ignore")
def main():
    inputs,targets = util.load_data()
    a=[2,18,29,36,46,47]
    # Bỏ mẫu 23 hoặc 2
    #Đổi mẫu 21 thành 29 
    inp_test = inputs[a,:]
    out_test = targets[a,:]

    b=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,22,23,24,25,26,27,28,21,30,31,32,33,34,35,40,37,38,39,41,42,43,44,45]
    inp_train = np.zeros((42,4))

    inp_train=inputs[b,:]
    out_train=targets[b,:]
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50, restore_best_weights = True)
    model = Sequential([
    Dense(5, activation = 'tanh',kernel_initializer = 'normal'),
    Dense(5, activation = 'tanh',kernel_initializer = 'normal'),
    Dense(1,kernel_initializer = 'normal', activation = 'sigmoid'),
    ])
    model.compile(optimizer ='adam',
                loss = 'mean_squared_error',
                metrics = ['mape'])

    history = model.fit(inp_train,out_train, epochs=5000, validation_data = (inp_test, out_test), batch_size=42,callbacks=[callback],verbose = 2)
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    results = model.evaluate(inp_test,  out_test, verbose = 0)
    print('test loss, test acc:', results)
    result = model.predict(inp_test)
    saiso  = np.abs((result-out_test)/out_test)*100
    saisotb = np.mean(saiso)
    phuongsai = np.std(saiso)
    print("Gia tri dua vao ",inp_test)
    print("Ket qua thuc te la :",out_test)
    print("Ket qua du doan la :",result)
    print("sai so tuong doi la :",saiso)
    print("Sai so truong binh :",saisotb)
    print("Phuong sai sai so la",phuongsai)
    print("RMSE :",np.sqrt(np.mean((result-out_test)**2)))
    model.save('saved_model/my_model')
if __name__ == '__main__':
    main()




