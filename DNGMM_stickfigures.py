import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
import csv
import time
import torch
from tensorflow.keras import backend as K

def log_csv(strToWrite, file_name):
    path = r'log_history/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + file_name + '.csv', 'a+', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(strToWrite)
    f.close()

def get_ACC_NMI(_y, _y_pred):
    y = np.array(_y)
    y_pred = np.array(_y_pred)
    s = np.unique(y_pred)
    t = np.unique(y)

    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(C)
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)
    acc = np.round(1.0 * count / len(y), 5)

    temp = np.array(y_pred)
    for i in range(N):
        y_pred[temp == col[i]] = i
    from sklearn.metrics import normalized_mutual_info_score,mutual_info_score
    nmi = np.round(normalized_mutual_info_score(y, y_pred, average_method="arithmetic"), 5)
    return acc, nmi

def get_all_data(ds_name='stickfigures', dir_path=r'dataset/'):
    dir_path = dir_path + ds_name
    x_all = torch.load(f"{dir_path}/x_all.pt", map_location=torch.device('cpu'))
    y_all = torch.load(f"{dir_path}/y_all.pt", map_location=torch.device('cpu'))
    if y_all.shape[0] != x_all.shape[0]:
        y_all = y_all.T
    # x_all = functional.resize(x_all, size=[56, 56])
    if type(x_all) == torch.Tensor:
        x_all = x_all.detach().cpu().numpy()
    if type(y_all) == torch.Tensor:
        y_all = y_all.detach().cpu().numpy()
    if x_all.shape[1] in [1, 3]:  # channel_last in tensorflow
        x_all = np.moveaxis(x_all, 1, -1)
    # dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))
    return x_all, y_all

def model_conv(load_weights=True):
    # init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    init = 'uniform'
    filters = [32, 128, hidden_units]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        input)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        x)
    # x = layers.Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed')(x)
    x = tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1))
    h = x
    x = layers.Dense(filters[1] * (input_shape[0] // 4 * input_shape[0] // 4), activation='relu')(x)
    x = layers.Reshape((input_shape[0] // 4, input_shape[0] // 4, filters[1]))(x)
    # x = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding=pad3, activation='relu')(x)
    x = layers.Conv2DTranspose(filters[0], kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(input_shape[2], kernel_size=5, strides=2, padding='same')(x)
    output = layers.Concatenate()([h,
                                   layers.Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    if load_weights:
        model.load_weights(f'weight_base_{ds_name}.h5')
        print('model_conv: weights was loaded')
    return model


def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)


def train_base(ds_xx,load_weights):
    model = model_conv(load_weights=load_weights)
    if not load_weights:
        model.compile(optimizer='adam', loss=loss_train_base)
        model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
        model.save_weights(f'weight_base_{ds_name}.h5')


def sorted_eig(X):
    e_vals, e_vecs = np.linalg.eig(X)
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def eig(X):
    e_vals, e_vecs = np.linalg.eig(X)
    idx = np.argsort(-e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def myloss(y_true, y_pred, pre):
    C = np.linalg.cholesky(pre)
    A = np.matmul(y_true,C)
    B = tf.matmul(y_pred,C)
    dis = losses.mse(A, B)
    return dis


def exploss(y_true, y_pred):
    loss = K.sum(K.square(y_pred - y_true), axis=1)
    loss = K.mean(K.exp(loss))
    return loss


def train(x, y):
    log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv(log_str.split(';'), file_name=ds_name)
    model = model_conv()

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    index = 0
    kmeans_n_init = 100
    
    index_array = np.arange(x.shape[0])
    num_views = y.shape[1]
    clusters = []
    for i in range(num_views):
        clusters += [len(np.unique(y[:, i]))]

    print("clusters:")
    print(clusters)
    clusters = np.asarray(clusters)
    
    for nv in range(num_views):
        mdl_coding = []
        n_clusters = clusters[nv]
        assignment = np.array([-1] * len(x))
        weights = np.ones(n_clusters)
        Pre = []
        for i in range(n_clusters):
            Pre.append(np.zeros((hidden_units,hidden_units)))

        ind = 0
        Us = []
        for ite in range(101):
            
            if ite % update_interval == 0:
                with tf.device('cpu'):
                    H = model(x).numpy()[:, :hidden_units]
                    x_hat = model(x).numpy()[:, hidden_units:]
                gm = GaussianMixture(n_components=n_clusters, covariance_type='full', init_params='k-means++').fit(H)
                assignment = gm.predict(H)
                labels = np.unique(assignment)
                U = gm.means_
                Us.append(U)
                C = gm.covariances_
                resp_total = gm.resp  #
                dim = int(H.shape[1] / 2)
                logp = gm.score_samples(H)
                LM = 0
                LD = 0
                for i in range(n_clusters):
                    cluster_i = []
                    cnt = 0.00001
                    for j in range(H.shape[0]):
                        if assignment[j] == labels[i]:
                            cluster_i.append(H[j,:])
                            cnt += 1
                    weights[i] = 1.0 * H.shape[0] / cnt
                    LM += (H.shape[1] ** 2 + 3 * H.shape[1] + 2)/4 * math.log2(cnt)
                    LD += cnt * math.log2(1.0 * H.shape[0] / cnt)

                #full covariance matrix
                LD -= np.sum(logp)
                # l2_loss = losses.mse(layers.Flatten()(x), x_hat)
                mdl_loss = LM+LD
                # mdl_loss += np.sum(l2_loss)
                mdl_coding.append(mdl_loss)
                print(mdl_loss)

                model.save_weights(f'weight_{ds_name}_{nv}_{ind}.h5')
                ind += 1

                Vs = []
                y_true = []
                cluster_diag = []
                for i in range(n_clusters):
                    pi = np.array(assignment) == labels[i]
                    number = np.sum(pi)
                    diag = tf.cast(tf.linalg.diag(pi.astype(int)), dtype=tf.float32)
                    cluster_diag.append(diag)
                    pre = np.linalg.inv(C[i])
                    Pre[i] = pre
                    Evals, V = sorted_eig(pre)
                    Vs.append(V)
                    H_vt = np.matmul(H, V)
                    U_vt = np.matmul(U, V)
                    mcr = []
                    for j in range(1, hidden_units):
                        dim = hidden_units - j
                        z_trun = H_vt[:, :dim]
                        I = np.eye(dim)
                        matrix1 = I + dim / (H.shape[0] * 0.5) * np.matmul(z_trun.T, z_trun)
                        matrix2 = I + dim / (number * 0.5) * np.matmul(np.matmul(z_trun.T, diag), z_trun)
                        count = number / (2 * H.shape[0]) * np.log(np.linalg.det(matrix2))
                        mcr_dim = count / (0.5 * np.log(np.linalg.det(matrix1)))
                        mcr.append(mcr_dim)
                    max_ind = min(np.argmin(mcr), int(np.sqrt(H.shape[1])))
                    for j in range(H.shape[0]):
                        if assignment[j] == labels[i]:
                            H_vt[j, -1 - max_ind:] = U_vt[i, -1 - max_ind:]
                    y_true.append(H_vt)

                loss = np.round(np.mean(loss_value), 5)
                acc, nmi = get_ACC_NMI(y[:,nv], assignment)

                # log
                log_str = f'iter {ite // update_interval}; acc, nmi = {acc, nmi}; loss:' \
                          f'{loss:.5f}; time:{time.time() - time_start:.3f}; mdl loss: {mdl_loss:.5f}'
                print(log_str)
                log_csv(log_str.split(';'), file_name=ds_name)


            #mini batch
            index = 0
            #np.random.shuffle(index_array)
            while(index * batch_size <= x.shape[0]):
                idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)
                    y_pred = model(x[idx])
                    x_hat = y_pred[:, hidden_units:]
                    loss_value = 0
                    for i in range(n_clusters):
                        loss_weights = resp_total[:, i] / tf.reduce_sum(resp_total[:, i])
                        loss_weights = tf.cast(loss_weights, dtype=tf.float32)
                        y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], Vs[i])
                        y_true_i = y_true[i]
                        loss_weights = np.array(loss_weights)[idx]
                        loss_value += tf.multiply(losses.mse(y_true_i[idx], y_pred_cluster), loss_weights)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                index = index + 1


        indd = np.argmin(np.asarray(mdl_coding))
        print(indd)
        model.load_weights(f'weight_{ds_name}_{nv}_{indd}.h5')
        with tf.device('cpu'):
            H = model(x).numpy()[:, :hidden_units]

        U=Us[indd]

        #traditional methods
        M = U.T
        S = np.matmul(M, M.T)
        eigsv, eigs = eig(S)
        A = np.real(eigs[:,:n_clusters-1])
        tmp1 = np.real(np.linalg.inv(np.matmul(A.T, A)))
        tmp2 = np.matmul(A, tmp1)
        tmp3 = np.matmul(tmp2, A.T)
        P = np.eye(H.shape[1]) - tmp3
        H = np.matmul(H, P.T)

        # retrain the encoder to learn this projection
        for epoch1 in range(30):
            #mini batch
            index = 0
            #np.random.shuffle(index_array)
            while(index * batch_size <= x.shape[0]):
                idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)
                    y_pred = model(x[idx])
                    x_hat = y_pred[:, hidden_units:]
                    y_true = H[idx]
                    loss_value = losses.mse(y_true, y_pred[:, :hidden_units])
                    # loss_value += losses.mse(layers.Flatten()(x[idx]), x_hat)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                index = index + 1


if __name__ == '__main__':
    pretrain_epochs = 200
    pretrain_batch_size = 256
    batch_size = 2048
    update_interval = 10
    hidden_units = 8
    tf.random.set_seed(42)

    time_start = time.time()
    ds_name='stickfigures'
    x, y = get_all_data(ds_name)
    input_shape = (x[0].shape[0], x[0].shape[0], 1)
    ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(8000).batch(pretrain_batch_size)
    train_base(ds_xx,load_weights=True)
    train(x, y)
    print(time.time() - time_start)
