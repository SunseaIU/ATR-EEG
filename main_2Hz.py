from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import os
from threading import Thread
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error


class LoadDataSet():
    def __init__(self, trial,data_name, type='de_LDS'):
        self.trial = trial
        self.root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), 'data','seed-vig',data_name)
        self.type = type

    def load_chronologically(self):
        feature = loadmat(os.path.join(self.root, 'DE', str(self.trial) + '.mat'))[self.type]
        label = np.squeeze(loadmat(os.path.join(self.root, 'perclos_labels', str(self.trial) + '.mat'))["perclos"])

        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 17, 25)

        sample_num = feature.shape[0]

        train_val_num = int(sample_num * 0.3)
        test_num = sample_num - train_val_num
        train_val_idx = np.random.RandomState(seed=970304).permutation(train_val_num)
        train_num = int(train_val_num * 0.8)
        val_num = train_val_num - train_num
        train_idx = train_val_idx[:train_num]
        val_idx = np.setdiff1d(train_val_idx, train_idx)

        train_feature, val_feature = feature[train_idx, :, :], feature[val_idx, :, :]
        test_feature = feature[-test_num:, :, :]
        train_feature, val_feature, test_feature = np.expand_dims(train_feature, -1), np.expand_dims(val_feature,
                                                                                                     -1), np.expand_dims(
            test_feature, -1),

        train_label, val_label = label[train_idx], label[val_idx]
        test_label = label[-test_num:]
        train_label, val_label, test_label = np.expand_dims(train_label, -1), np.expand_dims(val_label,
                                                                                             -1), np.expand_dims(
            test_label, -1)

        return train_feature, val_feature, test_feature, train_label, val_label, test_label

    def load_stochastically(self):
        # EEG = loadmat(os.path.join(self.root, 'Raw_Data', str(self.trial)+'.mat'))["EEG"]["data"][0][0]
        feature = loadmat(os.path.join(self.root, 'DE', str(self.trial) + '.mat'))[self.type]
        label = np.squeeze(loadmat(os.path.join(self.root, 'perclos_labels', str(self.trial) + '.mat'))["perclos"])

        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 17, 25)

        allIdx = np.random.RandomState(seed=970304).permutation(feature.shape[0])
        amount = int(feature.shape[0] / 5)

        valIdx = allIdx[0:amount]
        trainIdx = np.setdiff1d(allIdx, valIdx)

        train_feature = feature[trainIdx, :, :]
        val_feature = feature[valIdx, :, :]

        train_feature, val_feature = np.expand_dims(train_feature, -1), np.expand_dims(val_feature, -1)

        train_label = label[trainIdx]
        val_label = label[valIdx]
        train_label, val_label = np.expand_dims(train_label, -1), np.expand_dims(val_label, -1)

        return train_feature, train_label, val_feature, val_label

    def load_all_data(self):
        feature = loadmat(os.path.join(self.root, 'DE', str(self.trial) + '.mat'))[self.type]
        label = np.squeeze(loadmat(os.path.join(self.root, 'perclos_labels', str(self.trial) + '.mat'))["perclos"])

        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 17, 25)

        feature, label = np.expand_dims(feature, -1), np.expand_dims(label, -1)
        return feature, label


class Vignet(tf.keras.Model):
    def __init__(self, brain_channels=17):
        tf.keras.backend.set_floatx("float64")
        super(Vignet, self).__init__()

        # 设置损失和metric指标tracker
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_tracker = keras.metrics.MeanSquaredError(name='mse')
        self.mape_tracker = keras.metrics.MeanAbsolutePercentageError(name='mape')

        self.regularizer = tf.keras.regularizers.L1L2(l1=0.05, l2=0.005)
        self.activation = tf.nn.leaky_relu

        # Define convolution layers
        self.conv1 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv2 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv3 = tf.keras.layers.Conv2D(10, (1, 5), kernel_regularizer=self.regularizer, activation=self.activation)
        self.conv4 = tf.keras.layers.Conv2D(20, (17, 1), kernel_regularizer=self.regularizer,
                                            activation=self.activation)

        self.conv_mhrssa1 = tf.keras.layers.Conv2D(brain_channels, (brain_channels, 1),
                                                   kernel_regularizer=self.regularizer, activation=None)
        self.conv_mhrssa2 = tf.keras.layers.Conv2D(brain_channels, (brain_channels, 1),
                                                   kernel_regularizer=self.regularizer, activation=None)
        self.conv_mhrssa3 = tf.keras.layers.Conv2D(brain_channels, (brain_channels, 1),
                                                   kernel_regularizer=self.regularizer, activation=None)

        self.conv_depthwise1 = tf.keras.layers.DepthwiseConv2D((1, 5), kernel_regularizer=self.regularizer,
                                                               activation=None)
        self.conv_depthwise2 = tf.keras.layers.DepthwiseConv2D((1, 5), kernel_regularizer=self.regularizer,
                                                               activation=None)
        self.conv_depthwise3 = tf.keras.layers.DepthwiseConv2D((1, 5), kernel_regularizer=self.regularizer,
                                                               activation=None)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    # Define multi-head residual spectro-spatio attention module
    def MHRSSA(self, x, out_filter, conv_mhrssa, conv_depthwise):
        for i in range(out_filter):
            tmp = conv_mhrssa(x)
            if i == 0:
                MHRSSA = tmp
            else:
                MHRSSA = tf.concat((MHRSSA, tmp), 1)

        MHRSSA = tf.transpose(MHRSSA, perm=[0, 3, 2, 1])

        MHRSSA = conv_depthwise(MHRSSA)
        MHRSSA = tf.keras.activations.softmax(MHRSSA)
        return MHRSSA

    def call(self, x, training=None, mask=None):
        att1 = self.MHRSSA(x, 10, self.conv_mhrssa1, self.conv_depthwise1)
        hidden = self.conv1(x)
        hidden *= att1

        att2 = self.MHRSSA(hidden, 10, self.conv_mhrssa2, self.conv_depthwise2)
        hidden = self.conv2(hidden)
        hidden *= att2

        att3 = self.MHRSSA(hidden, 10, self.conv_mhrssa3, self.conv_depthwise3)
        hidden = self.conv3(hidden)
        hidden *= att3

        hidden = self.conv4(hidden)

        hidden = self.flatten(hidden)
        hidden = self.dense(hidden)

        y_hat = hidden
        return y_hat

    def train_step(self, data):

        x, y = data
        # 1. 创建GradientTape，通过前向传播记录loss相对于可训练变量的计算过程；
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.metrics.MSE(y, y_pred)

        # 2. 计算梯度
        grad = tape.gradient(loss, self.trainable_variables)

        # 3. 使用指定的优化器在模型上应用梯度
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        # 4. 更新每次输出的metrics.
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(y, y_pred)
        self.mape_tracker.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred)

        # 5. 返回metrics, metrics默认含有loss
        return {"loss": self.loss_tracker.result(), "mse": self.mse_tracker.result(),
                "mape": self.mape_tracker.result()}

    def test_step(self, data):
        x, y = data
        # 1. 创建GradientTape，通过前向传播记录loss相对于可训练变量的计算过程；
        with tf.GradientTape() as tape:
            y_pred = self(x, training=False)
            loss = keras.metrics.MSE(y, y_pred)

        # 4. 更新每次输出的metrics.
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(y, y_pred)
        self.mape_tracker.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred)

        # 5. 返回metrics, metrics默认含有loss
        return {"loss": self.loss_tracker.result(), "mse": self.mse_tracker.result(),
                "mape": self.mape_tracker.result()}

    @property
    def metrics(self):
        """
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        :return:
        """
        return [self.loss_tracker, self.mse_tracker, self.mape_tracker]


class Attention(tf.keras.Model):
    def __init__(self, source_num):
        super(Attention, self).__init__()

        self.regularizer = tf.keras.regularizers.L1L2(l1=0.05, l2=0.005)
        self.activation = tf.nn.leaky_relu

        self.atten_conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 11), kernel_regularizer=self.regularizer,
                                                  activation=self.activation)
        # self.atten_conv2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(1, 5), kernel_regularizer=self.regularizer,
        #                                           activation=self.activation)
        self.atten_conv3 = tf.keras.layers.Conv2D(filters=20, kernel_size=(17, 1), kernel_regularizer=self.regularizer,
                                                  activation=self.activation)
        self.atten_flatten = tf.keras.layers.Flatten()
        self.atten_dense1 = tf.keras.layers.Dense(units=32, activation=self.activation,
                                                  kernel_regularizer=self.regularizer)
        self.atten_dense2 = tf.keras.layers.Dense(units=source_num + 1, kernel_regularizer=self.regularizer)

    def call(self, inputs, training=None, mask=None):
        # 训练A2T网络
        hidden = self.atten_conv1(inputs)
        # hidden = self.atten_conv2(hidden)
        hidden = self.atten_conv3(hidden)
        hidden = self.atten_flatten(hidden)
        hidden = self.atten_dense1(hidden)
        hidden = self.atten_dense2(hidden)

        y_atten = tf.keras.activations.softmax(hidden)
        return y_atten


class A2TVignet(tf.keras.Model):
    def __init__(self, source_model_list):
        super(A2TVignet, self).__init__()
        self.base_model = Vignet()
        self.source_model_list = source_model_list
        self.atten_model = Attention(source_num=len(source_model_list))

        self.loss_tracker_base = keras.metrics.Mean(name="loss_base")
        self.loss_tracker_final = keras.metrics.Mean(name="loss_final")
        self.mse_tracker_base = keras.metrics.MeanSquaredError(name='mse_base')
        self.mse_tracker_final = keras.metrics.MeanSquaredError(name='mse_final')
        self.mape_tracker_base = keras.metrics.MeanAbsolutePercentageError(name='mape_base')
        self.mape_tracker_final = keras.metrics.MeanAbsolutePercentageError(name='mape_final')

    def call(self, inputs, training=None, mask=None):
        y_hat = self.base_model(inputs, training=False)
        y_atten = self.atten_model(inputs, training=False)
        # combine base and source task outputs
        for i, sm in zip(range(len(self.source_model_list)), self.source_model_list):
            if i == 0:
                y_probs = sm(inputs, training=False)
            else:
                y_probs = tf.concat([y_probs, sm(inputs, training=False)], axis=1)
        y_probs = tf.concat([y_probs, y_hat], axis=1)

        y_final = tf.multiply(y_probs, y_atten)

        y_final = tf.reduce_sum(y_final, axis=1, keepdims=True)
        return y_hat, y_final, y_atten

    def compile(self, base_optimizer, final_optimizer):
        super(A2TVignet, self).compile()
        self.base_optimizer = base_optimizer
        self.final_optimizer = final_optimizer

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_hat = self.base_model(x, training=True)
            loss_base = keras.metrics.MSE(y, y_hat)

        gradient_base = tape.gradient(loss_base, self.base_model.trainable_variables)
        self.base_optimizer.apply_gradients(zip(gradient_base, self.base_model.trainable_variables))

        with tf.GradientTape() as tape:
            y_hat, y_final, y_atten = self(x, training=True)
            loss_final = keras.metrics.MSE(y, y_final)

        gradient_final = tape.gradient(loss_final, self.atten_model.trainable_variables)
        self.final_optimizer.apply_gradients(zip(gradient_final, self.atten_model.trainable_variables))

        self.loss_tracker_base.update_state(loss_base)
        self.loss_tracker_final.update_state(loss_final)

        return {"loss_base": self.loss_tracker_base.result(), "loss_final": self.loss_tracker_final.result()}

    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_hat, y_final, y_atten = self(x, training=False)
            loss_base = keras.metrics.MSE(y, y_hat)
            loss_final = keras.metrics.MSE(y, y_final)

        self.loss_tracker_base.update_state(loss_base)
        self.loss_tracker_final.update_state(loss_final)

        return {"loss_base": self.loss_tracker_base.result(), "loss_final": self.loss_tracker_final.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_base, self.loss_tracker_final]


class Train():
    def __init__(self, trial,data_name):
        self.trial = trial
        self.load_data = LoadDataSet(trial=trial,data_name=data_name)
        self.root = os.path.dirname(os.path.abspath("__file__"))
        self.data_name=data_name

        # parameters for vignet
        self.vignet = Vignet()
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epoch_num = 1000
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

    def train_source_vignet(self):
        # load data
        train_feature, train_label, val_feature, val_label = self.load_data.load_stochastically()

        early_stoping_callback = keras.callbacks.EarlyStopping(monitor='val_mse', patience=15, mode='min')
        model_save_dir = os.path.join(self.root, 'model', 'source',self.data_name, str(self.trial))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_svae_callback = keras.callbacks.ModelCheckpoint(monitor='val_mse', mode='min',
                                                              filepath=os.path.join(model_save_dir, 'weights'),
                                                              save_best_only=True)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=5, verbose=1,
                                                               mode='min')
        self.vignet.compile(optimizer=self.optimizer)
        self.vignet.fit(x=train_feature, y=train_label, batch_size=self.batch_size, epochs=self.epoch_num,
                        validation_data=(val_feature, val_label),
                        callbacks=[early_stoping_callback, model_svae_callback, reduce_lr_callback])

    def train_a2t_vignet(self, sub_num):
        # load data
        train_feature, val_feature, test_feature, train_label, val_label, test_label = self.load_data.load_chronologically()

        # load source_model
        sub_list = list(range(1, sub_num+1))
        sub_list.remove(self.trial)
        source_model_list = []
        for i in sub_list:
            vignet_temp = Vignet()
            vignet_temp.load_weights(
                filepath=os.path.join(self.root, 'model', 'source',self.data_name, str(self.trial), 'weights')).expect_partial()
            source_model_list.append(vignet_temp)

        # parameters for a2t_vignet
        a2t_vignet = A2TVignet(source_model_list=source_model_list)
        base_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        atten_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # train
        early_stoping_callback = keras.callbacks.EarlyStopping(monitor='val_loss_final', patience=15, mode='min')
        log_dir = os.path.join(self.root, 'logs',self.data_name, str(self.trial))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        model_save_dir = os.path.join(self.root, 'model', 'target',self.data_name, str(self.trial))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_svae_callback = keras.callbacks.ModelCheckpoint(monitor='val_loss_final', mode='min',
                                                              filepath=os.path.join(model_save_dir, 'weights'),
                                                              save_best_only=True)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss_final', factor=0.5, patience=5,
                                                               verbose=1,
                                                               mode='min')
        a2t_vignet.compile(base_optimizer=base_optimizer, final_optimizer=atten_optimizer)
        a2t_vignet.fit(x=train_feature, y=train_label, batch_size=self.batch_size, epochs=self.epoch_num,
                       validation_data=(val_feature, val_label),
                       callbacks=[early_stoping_callback, tensorboard_callback, model_svae_callback,
                                  reduce_lr_callback])


class Predict():
    def __init__(self, trial,data_name):
        self.trial = trial
        self.load_data = LoadDataSet(trial=trial,data_name=data_name)
        self.root = os.path.dirname(os.path.abspath("__file__"))
        self.data_name=data_name

    def predict(self, sub_num):
        # load data
        train_feature, val_feature, test_feature, train_label, val_label, test_label = self.load_data.load_chronologically()

        # load model
        sub_list = list(range(1, sub_num+1))
        sub_list.remove(self.trial)
        source_model_list = []
        for i in sub_list:
            vignet_temp = Vignet()
            vignet_temp.load_weights(
                filepath=os.path.join(self.root, 'model', 'source',self.data_name, str(self.trial), 'weights')).expect_partial()
            source_model_list.append(vignet_temp)

        a2t_vignet = A2TVignet(source_model_list=source_model_list)
        a2t_vignet.load_weights(os.path.join(self.root, 'model', 'target',self.data_name, str(self.trial), 'weights')).expect_partial()
        y_pred_base, y_pred_final, y_atten_test = a2t_vignet.predict(x=test_feature)

        rmse_base = np.sqrt(mean_squared_error(y_true=test_label, y_pred=y_pred_base))
        rmse_final = np.sqrt(mean_squared_error(y_true=test_label, y_pred=y_pred_final))
        # load all data
        feature, lable = self.load_data.load_all_data()
        y_atten_all = a2t_vignet.predict(x=feature)[2]

        save_dir = os.path.join(self.root, 'result',self.data_name, str(self.trial)+'.mat')
        save_dict = {}
        save_dict['y_true'] = test_label
        save_dict['y_pred_final'] = y_pred_final
        save_dict['y_pred_base'] = y_pred_base
        save_dict['rmse_base'] = rmse_base
        save_dict['rmse_final'] = rmse_final
        save_dict['atten_test']=y_atten_test
        save_dict['atten_final']=y_atten_all
        savemat(file_name=save_dir, mdict=save_dict)


def train_source_vig(trial,data_name):
    model = Train(trial=trial,data_name=data_name)
    model.train_source_vignet()


def train_target_a2t_vignet(trial, sub_num,data_name):
    model = Train(trial=trial,data_name=data_name)
    model.train_a2t_vignet(sub_num=sub_num)

def preditct_a2t_vignet(trial,sub_num,data_name):
    model=Predict(trial=trial,data_name=data_name)
    model.predict(sub_num=sub_num)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    data_name='EEG_2Hz'

    # 用N个源域数据训练N个base模型
    # for i in range(1, 24):
    #     train_source_vig(i,data_name=data_name)

    # 用目标域的部分训练集数据(30%)训练一个base模型
    # for i in range(1, 24):
    #     train_target_a2t_vignet(i, 23,data_name=data_name)

    for i in range(1,24):
        preditct_a2t_vignet(i,23,data_name=data_name)


# 代码逻辑
# 1.用N个源域数据训练N个base模型
# 2.用目标域的部分训练集数据(30%)训练一个base模型
# 3.用2中的数据训练attention网络，中间会输出N+1个权重a_i(sum(a_i)=1)，将sum(a_i*y_i)作为最终某个样本的预测，以此调整attention网络
