import os
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits


def create_dataset(data_path, batch_size=32, num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    # Resize images to (32, 32)
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = CV.Rescale(rescale, shift)  # rescale images
    # change shape from (height, width, channel) to (channel, height, width) to fit network.
    hwc2chw_op = CV.HWC2CHW()
    # change data type of label to int32 to fit network
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op,
                            num_parallel_workers=num_parallel_workers)

    buffer_size = 10000
    # 10000 as in LeNet train script
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(1)

    return mnist_ds


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Conv layer weight initial."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """Fc layer weight initial."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """Weight initial."""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """Lenet network structure."""
    # define the operator required

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc = fc_with_initialize(120, 84)
        self.fc3 = nn.Dense(84, 10, weight_variable(), weight_variable())
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_net(model, epoch_size, mnist_path, ckpoint_cb):
    """Define the training method."""
    print("============== Starting Training ==============")
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32)
    model.train(epoch_size, ds_train, callbacks=[
        ckpoint_cb, LossMonitor()], dataset_sink_mode=False)


def test_net(network, model, mnist_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    ds_eval = create_dataset(os.path.join(mnist_path, "test"))
    result = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== result:{} ==============".format(result))


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    lr = 0.01
    momentum = 0.9
    epoch_size = 1
    mnist_path = "./MNIST_Data"
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_dataset = create_dataset(os.path.join(mnist_path, "train"), 32)
    val_dataset = create_dataset(os.path.join(mnist_path, "test"))
    network = LeNet5()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(
        save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"loss", 'acc'})

    print("============== Starting Training ==============")

    config_ck = CheckpointConfig(save_checkpoint_steps=1000,
                                 keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(
        prefix="checkpoint_lenet", directory='./ckpt', config=config_ck)
    callbacks = [LossMonitor()]

    train_net(model, epoch_size, mnist_path, ckpoint_cb)
    test_net(network, model, mnist_path)
