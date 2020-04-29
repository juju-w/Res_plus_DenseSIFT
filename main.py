from ResNet import ResNet
# from ResNet1 import ResNet
import argparse
from utils import *
from ResNetSIFT import ResNetSIFT



"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='customize', help='[cifar10, cifar100, mnist, fashion-mnist, tiny, customize,sift')


    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=50, help='18, 34, 50, 101, 152')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    i=0
    j=0
    # tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # cnn = ResNetSIFT(sess, args,i,j)
        cnn = ResNet(sess, args,i,j)
        # build graph
        cnn.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            cnn.train()

            print(" [*] Training finished! \n")

            cnn.test()
            print(" [*] Test finished!")


        if args.phase == 'test' :
            cnn.test()
            print(" [*] Test finished!")

        if args.phase == 'fea' :
            featr,labtr,feate,labte=cnn.fea_get()
            from sklearn import svm
            clf=svm.SVC()
            clf.fit(featr,labtr[0:len(featr)])
            print(clf.score(feate,labte))
            print(" [*] features already get!")
    tf.reset_default_graph()

if __name__ == '__main__':
            main()
