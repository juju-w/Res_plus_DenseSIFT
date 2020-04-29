import time
from ops import *
from utils import *
label_dim=2
img_size=48
import tensorflow as tf

class ResNetSIFT(object):

    def __init__(self, sess, args,i,j):
        self.i=i
        self.j=j
        self.model_name = 'ResNetSIFT'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'dsift' :
            self.train_x, self.test_x,self.train_y, self.test_y ,self.dsift_train_x,self.dsift_test_x= load_dsift(self.j)
            self.img_size = img_size
            self.c_dim = 3
            self.label_dim = label_dim
        if self.dataset_name == 'cifar100' :
            self.train_x, self.test_x,self.train_y, self.test_y ,self.dsift_train_x,self.dsift_test_x = load_cifar100_dsift()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100
        if self.dataset_name == 'cifar10' :
            self.train_x, self.test_x,self.train_y, self.test_y ,self.dsift_train_x,self.dsift_test_x = load_cifar10_dsift()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = args.lr

    def network(self,x,dsift,is_training=True,reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 64 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################
            x = residual_block(x, channels=ch * 2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]):
                x = residual_block(x, channels=ch * 2, is_training=is_training, downsample=False,
                                   scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch * 4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]):
                x = residual_block(x, channels=ch * 4, is_training=is_training, downsample=False,
                                   scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch * 8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]):
                x = residual_block(x, channels=ch * 8, is_training=is_training, downsample=False,
                                   scope='resblock_3_' + str(i))

            ########################################################################################################

            x = batch_norm(x, is_training, scope='batch_normx')
            # x=tf.reshape(x,[1,-1])
            x = relu(x)
            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit1')
            #######################################################################################################
            # y = conv(dsift, 64, kernel=3, stride=2, use_bias=True, scope='convy1')
            # y = conv(y, 16, kernel=3, stride=2, use_bias=True, scope='convy2')
            y = batch_norm(dsift, is_training, scope='batch_normy')
            y = relu(y)
            y = global_avg_pooling(y)
            # y = tf.reshape(y, [y.shape[0], 1, 1, y.shape[-1]])
            # y=tf.reshape(y,[1,-1])
            y = fully_conneted(y, units=self.label_dim, scope='logit2')
            #######################################################################################################
            z = tf.concat([x, y], 1)
            z = fully_conneted(z, units=self.label_dim, scope='logit3')

            return x,y,z








            # y = tf.reshape(y, [y.shape[0], 1, 1, y.shape[-1]])
            # # y=batch_norm(y,is_training, scope='batch_normy')
            # y = relu(y)
            # #########################################################################################################
            # z = tf.concat([x, y], 3)
            # z = fully_conneted(z, units=256, scope='logit_siftz')
            # # z=relu(z)
            # z = fully_conneted(z, units=self.label_dim, scope='logit')
            # return z

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim],name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')
        self.dsift_train_inptus=tf.placeholder(tf.float32,[self.batch_size,5,5,280],name='dsift_train_input')

        self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim],name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')
        self.dsift_test_inptus=tf.placeholder(tf.float32,[len(self.dsift_test_x),5,5,280],name='dsift_test_input')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logitsx,self.train_logitsy,self.train_logits = \
            self.network(self.train_inptus,self.dsift_train_inptus)
        self.test_logitsx,self.test_logitsy,self.test_logits = \
            self.network(self.test_inptus,self.dsift_test_inptus, is_training=False, reuse=True)

        self.train_losshole,self.train_lossx,self.train_lossy,self.train_acc,self.train_plab,self.train_tlab = \
            classification_loss3(logit1=self.train_logits,logit2=self.train_logitsx,logit3=self.train_logitsy, label=self.train_labels)
        self.test_losshole,self.test_lossx,self.test_lossy,self.test_acc,self.test_plab,self.test_tlab = \
            classification_loss3(logit1=self.test_logits,logit2=self.test_logitsx,logit3=self.test_logitsy,label=self.test_labels)

        self.train_loss=(self.train_losshole+self.train_lossx+self.train_lossy)*0.3333
        self.test_loss=(self.test_losshole+self.test_lossx+self.test_lossy)*0.3333

        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss

        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_acc)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_acc)
        # self.summary_fea=tf.summary.image(['fea',self.fea])
        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir+str(self.i)+'='+str(self.j), self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75):
                epoch_lr = epoch_lr * 0.1
            fea = []
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_dsift=self.dsift_train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)
                # batch_dsift = data_augmentation(batch_dsift, self.img_size, self.dataset_name)
                train_feed_dict = {
                    self.train_inptus: batch_x,
                    self.train_labels: batch_y,
                    self.dsift_train_inptus : batch_dsift,
                    self.lr: epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus: self.test_x,
                    self.test_labels: self.test_y,
                    self.dsift_test_inptus:self.dsift_test_x
                }

                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run([self.optim, self.train_summary, self.train_loss, self.train_loss],
                    feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss, test_accuracy= self.sess.run([self.test_summary, self.test_loss, self.test_acc], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print(
                    "Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.5f, test_accuracy: %.5f, learning_rate : %.4f,train_loss: %5f" \
                    % (
                    epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr,
                    train_loss))
                # if epoch == self.epoch
                # print(subfea)
                # fea.append(subfea)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}_{}-{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr,self.i,self.j)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: self.test_x,
            self.test_labels: self.test_y,
            self.dsift_test_inptus: self.dsift_test_x
        }

        summary_str, test_loss, test_accuracy,p,t = self.sess.run(
            [self.test_summary, self.test_loss, self.test_acc,self.test_plab,self.test_tlab], feed_dict=test_feed_dict)
        import metrics
        print("test_accuracy: {}".format(test_accuracy))
        with open('resnetsift.txt', 'a') as f:  # 设置文件对象
            f.write(str(self.i) + '-' + str(self.j) + ',' + str(metrics.accuracy(t, p)) + ',' + str(
                metrics.precision(t, p)) + ',' + str(metrics.recall(t, p)) + ',' + str(metrics.f1score(t, p))
                    + ',' + str(metrics.ft(t, p)) + '\n')
    def fea_get(self):
        subtrain = []
        subtest = []
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        start_batch_id = 0
        for idx in range(start_batch_id, self.iteration):
            batch_x = self.train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
            # batch_dsift = self.dsift_train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.train_y[idx * self.batch_size:(idx + 1) * self.batch_size]

            batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

            train_feed_dict = {
                self.train_inptus: batch_x,
                self.train_labels: batch_y,
                # self.dsift_train_inptus: batch_dsift,
            }

            test_feed_dict = {
                self.test_inptus: self.test_x,
                self.test_labels: self.test_y,
                # self.dsift_test_inptus: self.dsift_test_x

            }

            # update network
            subtrain1 = self.sess.run(
                [self.subfea1],
                feed_dict=train_feed_dict)
            subtrain.append(subtrain1)

            # test
        subtest1 = self.sess.run(
            [self.subfea2], feed_dict=test_feed_dict)
        subtest.append(subtest1)
        shape1, shape2 = np.array(subtrain).shape, np.array(subtest).shape
        return np.array(subtrain).reshape(shape1[0] * shape1[2], shape1[-1]), np.array(self.train_y)[:, 0], np.array(
            subtest).reshape(shape2[2], shape2[-1]), np.array(self.test_y)[:, 0]
