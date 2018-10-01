import mxnet as mx
import time
import click


import unet

def score(sym, data_shape, dev, batch_size, num_batches):
    # get mod
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    return (time.time() - tic) / (num_batches*batch_size)


@click.command()
@click.option('--align', default=0)
@click.option('--num_iter', default=10)
@click.option('--num_cycles', default=5)
def run(align, num_iter, num_cycles):
    data_shape = (1, 3, 192, 192)
    batch_size = 1
    num_iter = 100
    sym = unet.unet(alignment=align)
    for f in range(num_cycles):
        print("MXNet time: ", score(sym, [('data', data_shape)], mx.cpu(), batch_size, num_batches=num_iter))
        time.sleep(1)


if __name__ == '__main__':
    run()
