import sys
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
from fuel.transformers import Transformer

### optimization algorithm definition
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from extensions import DumpLog, DumpBest, PrintingTo, DumpVariables
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, PARAMETER

import util


logging.basicConfig()
logger = logging.getLogger(__name__)
floatX = theano.config.floatX

def zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return np.ones(shape, dtype=theano.config.floatX)

def glorot(shape):
    d = np.sqrt(6. / sum(shape))
    return np.random.uniform(-d, +d, size=shape).astype(theano.config.floatX)

def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(theano.config.floatX)

_datasets = None
def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]

def get_stream_(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream

class SampleDrops2(Transformer):
    def __init__(self, data_stream, drop_prob,
                 hidden_dim, is_for_test, permutation,
                 drop_law = "constant",
                 **kwargs):
        super(SampleDrops2, self).__init__(
            data_stream, **kwargs)
        self.drop_prob = drop_prob
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False
        self.permutation = permutation

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        transformed_data.append(
            np.swapaxes(data[0].reshape(data[0].shape[0], -1),
                        0, 1)[self.permutation, :, np.newaxis])
        transformed_data.append(data[1][:, 0])
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            drops = np.ones((T, B, self.hidden_dim)) * self.drop_prob
        else:
            drops = np.random.binomial(n=1, p=self.drop_prob,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(drops.astype(floatX))
        return transformed_data

def get_stream(which_set, batch_size, drop_prob,
               hidden_dim, for_evaluation,
               num_examples=None):
    np.random.seed(seed=1)
    permutation = np.random.randint(0, 784, size=(784,))
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    if which_set == "train":
        ds = SampleDrops2(stream, drop_prob, hidden_dim, for_evaluation, permutation)
    else:
        ds = SampleDrops2(stream, drop_prob, hidden_dim, for_evaluation, permutation)
    ds.sources = ('x', 'y', 'drops')
    return ds



def bn(x, gammas, betas, mean, var, args):
    assert mean.ndim == 1
    assert var.ndim == 1
    assert x.ndim == 2
    if not args.use_population_statistics:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
    #var = T.maximum(var, args.epsilon)
    #var = var + args.epsilon

    if args.baseline:
        y = x + betas
    else:
        var_corrected = var + args.epsilon

        y = theano.tensor.nnet.bn.batch_normalization(
            inputs=x, gamma=gammas, beta=betas,
            mean=T.shape_padleft(mean), std=T.shape_padleft(T.sqrt(var_corrected)),
            mode="high_mem")
    assert mean.ndim == 1
    assert var.ndim == 1
    return y, mean, var

activations = dict(
    tanh=T.tanh,
    identity=lambda x: x,
    relu=lambda x: T.max(0, x))


class Empty(object):
    pass

class LSTM(object):
    def __init__(self, args, nclasses):
        self.nclasses = nclasses
        self.activation = activations[args.activation]

    def allocate_parameters(self, args):
        if hasattr(self, "parameters"):
            return self.parameters

        self.parameters = Empty()

        h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
        c0 = theano.shared(zeros((args.num_hidden,)), name="c0")
        if args.init == "id":
            Wa = theano.shared(np.concatenate([
                np.eye(args.num_hidden),
                orthogonal((args.num_hidden,
                            3 * args.num_hidden)),], axis=1).astype(theano.config.floatX), name="Wa")
        else:
            Wa = theano.shared(orthogonal((args.num_hidden, 4 * args.num_hidden)), name="Wa")
        Wx = theano.shared(orthogonal((1, 4 * args.num_hidden)), name="Wx")
        a_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas")
        b_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas")
        ab_betas = theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas")

        # forget gate bias initialization
        forget_biais = ab_betas.get_value()
        forget_biais[args.num_hidden:2*args.num_hidden] = 1.
        ab_betas.set_value(forget_biais)

        c_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="c_gammas")
        c_betas  = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="c_betas")

        if not args.baseline:
            parameters_list = [h0, c0, Wa, Wx, a_gammas, b_gammas, ab_betas, c_gammas, c_betas]
        else:
            parameters_list = [h0, c0, Wa, Wx, ab_betas, c_betas]
        for parameter in parameters_list:
            print parameter.name
            add_role(parameter, PARAMETER)
            setattr(self.parameters, parameter.name, parameter)

        return self.parameters

    def construct_graph_popstats(self, args, x, drops,
                                 length, popstats=None):
        p = self.allocate_parameters(args)


        def stepfn(x, drops,
                   dummy_h, dummy_c,
                   pop_means_a, pop_means_b, pop_means_c,
                   pop_vars_a, pop_vars_b, pop_vars_c,
                   h, c):

            atilde = T.dot(h, p.Wa)
            btilde = x
            if args.baseline:
                a_normal, a_mean, a_var = bn(atilde, 1.0, p.ab_betas, pop_means_a, pop_vars_a, args)
                b_normal, b_mean, b_var = bn(btilde, 1.0, 0,          pop_means_b, pop_vars_b, args)
            else:
                a_normal, a_mean, a_var = bn(atilde, p.a_gammas, p.ab_betas, pop_means_a, pop_vars_a, args)
                b_normal, b_mean, b_var = bn(btilde, p.b_gammas, 0,          pop_means_b, pop_vars_b, args)
            ab = a_normal + b_normal
            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                          for j, fn in enumerate([self.activation] + 3 * [T.nnet.sigmoid])]

            if args.elephant:
                c_n = dummy_c + f * c + drops * (i * g)
            else:
                c_n = dummy_c + f * c + i * g
            if args.baseline:
                c_normal, c_mean, c_var = bn(c_n, 1.0, p.c_betas, pop_means_c, pop_vars_c, args)
            else:
                c_normal, c_mean, c_var = bn(c_n, p.c_gammas, p.c_betas, pop_means_c, pop_vars_c, args)
            h_n = dummy_h + o * self.activation(c_normal)


            ## Zoneout
            if args.zoneout:
                h = h_n * drops + (1 - drops) * h
                c = c_n * drops + (1 - drops) * c
            else:
                h = h_n
                c = c_n

            return (h, c, atilde, btilde, c_normal,
                   a_mean, b_mean, c_mean,
                    a_var, b_var, c_var)


        xtilde = T.dot(x, p.Wx)
        if args.noise:
            # prime h with white noise
            Trng = MRG_RandomStreams()
            h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
        elif args.summarize:
            # prime h with mean of example
            h_prime = x.mean(axis=[0, 2])[:, None]
        else:
            h_prime = 0

        dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
                            c=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

        if popstats is None:
            popstats = OrderedDict()
            for key, size in zip("abc", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden]):
                for stat, init in zip("mean var".split(), [0, 1]):
                    name = "%s_%s" % (key, stat)
                    popstats[name] = theano.shared(
                        init + np.zeros((length, size,), dtype=theano.config.floatX),
                        name=name)
        popstats_seq = [popstats['a_mean'], popstats['b_mean'], popstats['c_mean'],
                        popstats['a_var'], popstats['b_var'], popstats['c_var']]

        [h, c, atilde, btilde, htilde,
         batch_mean_a, batch_mean_b, batch_mean_c,
         batch_var_a, batch_var_b, batch_var_c ], _ = theano.scan(
             stepfn,
             sequences=[xtilde, drops, dummy_states["h"], dummy_states["c"]] + popstats_seq,
             outputs_info=[T.repeat(p.h0[None, :], xtilde.shape[1], axis=0) + h_prime,
                           T.repeat(p.c0[None, :], xtilde.shape[1], axis=0),
                           None, None, None,
                           None, None, None,
                           None, None, None])

        batchstats = OrderedDict()
        batchstats['a_mean'] = batch_mean_a
        batchstats['b_mean'] = batch_mean_b
        batchstats['c_mean'] = batch_mean_c
        batchstats['a_var'] = batch_var_a
        batchstats['b_var'] = batch_var_b
        batchstats['c_var'] = batch_var_c

        updates = OrderedDict()
        if not args.use_population_statistics:
            alpha = 1e-2
            for key in "abc":
                for stat, init in zip("mean var".split(), [0, 1]):
                    name = "%s_%s" % (key, stat)
                    popstats[name].tag.estimand = batchstats[name]
                    updates[popstats[name]] = (alpha * batchstats[name] +
                                               (1 - alpha) * popstats[name])
        return dict(h=h, c=c,
                    atilde=atilde, btilde=btilde, htilde=htilde), updates, dummy_states, popstats


def construct_common_graph(situation, args, outputs, dummy_states, Wy, by, y):
    ytilde = T.dot(outputs["h"][-1], Wy) + by
    yhat = T.nnet.softmax(ytilde)

    errors = T.neq(y, T.argmax(yhat, axis=1))
    cross_entropies = T.nnet.categorical_crossentropy(yhat, y)

    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")
    graph = ComputationGraph([cost, cross_entropy, error_rate])

    state_grads = dict((k, T.grad(cost, v)) for k, v in dummy_states.items())

    extensions = []
    extensions = [
        DumpVariables("%s_hiddens" % situation, graph.inputs,
                      [v.copy(name="%s%s" % (k, suffix))
                       for suffix, things in [("", outputs), ("_grad", state_grads)]
                       for k, v in things.items()],
                      batch=next(get_stream(which_set="train",
                                            batch_size=args.batch_size,
                                            num_examples=args.batch_size,
                                            drop_prob=args.drop_prob,
                                            for_evaluation=False,
                                            hidden_dim=args.num_hidden)
                                 .get_epoch_iterator(as_dict=True)),
                      before_training=True, every_n_epochs=10)]

    return graph, extensions

def construct_graphs(args, nclasses, length):
    constructor = LSTM #if args.lstm else raise NotImplementedError('we only lstm')

    if args.permuted:
        permutation = np.random.randint(0, length, size=(length,))

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")

    ### graph construction
    inputs = dict(features=T.tensor3("x"), drop=T.tensor3('drops'), targets=T.ivector("y"))
    x, drop, y = inputs["features"], inputs['drop'], inputs["targets"]

    theano.config.compute_test_value = "warn"
    batch = next(get_stream(which_set="train", batch_size=args.batch_size,
                            drop_prob=args.drop_prob, for_evaluation=False,
                            hidden_dim=args.num_hidden).get_epoch_iterator())
    x.tag.test_value = batch[0]
    y.tag.test_value = batch[1]
    drop.tag.test_value = batch[2]


    # x = x.reshape((x.shape[0], length + 0, 1))
    # y = y.flatten(ndim=1)
    # x = x.dimshuffle(1, 0, 2)
    # x = x[0:, :, :]

    # if args.permuted:
    #     x = x[permutation]

    args.use_population_statistics = False
    turd = constructor(args, nclasses)
    (outputs, training_updates, dummy_states, popstats) = turd.construct_graph_popstats(args, x, drop, length)
    training_graph, training_extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y)

    args.use_population_statistics = True
    (inf_outputs, inference_updates, dummy_states, _) = turd.construct_graph_popstats(args, x, drop, length, popstats=popstats)
    inference_graph, inference_extensions = construct_common_graph("inference", args, inf_outputs, dummy_states, Wy, by, y)

    add_role(Wy, PARAMETER)
    add_role(by, PARAMETER)
    args.use_population_statistics = False
    return (dict(training=training_graph,      inference=inference_graph),
            dict(training=training_extensions, inference=inference_extensions),
            dict(training=training_updates,    inference=inference_updates))


if __name__ == "__main__":
    sequence_length = 784
    nclasses = 10

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--zoneout", action="store_true")
    parser.add_argument("--elephant", action="store_true")
    parser.add_argument("--drop-prob", type=float, default=0.85)
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--initial-gamma", type=float, default=0.1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--init", type=str, default="ortho")
    parser.add_argument("--continue-from")
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()

    #assert not (args.noise and args.summarize)
    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed


    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)

    #graphs, extensions, updates = construct_graphs(args, nclasses, sequence_length)
    graph, extension, update = construct_graphs(args, nclasses, sequence_length)

    ### optimization algorithm definition
    step_rule = CompositeRule([
        StepClipping(1.),
        #Momentum(learning_rate=args.learning_rate, momentum=0.9),
        RMSProp(learning_rate=args.learning_rate, decay_rate=0.5),
    ])

    algorithm = GradientDescent(cost=graphs["training"].outputs[0],
                                parameters=graphs["training"].parameters,
                                step_rule=step_rule)
    algorithm.add_updates(updates["training"])
    model = Model(graphs["training"].outputs[0])
    extensions = extensions["training"] + extensions["inference"]


    # step monitor (after epoch to limit the log size)
    step_channels = []
    step_channels.extend([
        algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
        for name, param in model.get_parameter_dict().items()])
    step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
    step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
    step_channels.extend(graphs["training"].outputs)
    logger.warning("constructing training data monitor")
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="iteration", after_batch=False))

    # parameter monitor
    extensions.append(DataStreamMonitoring(
        [param.norm(2).copy(name="parameter.norm:%s" % name)
         for name, param in model.get_parameter_dict().items()],
        data_stream=None, after_epoch=True))

    # performance monitor
    for situation in "training".split(): # add inference
        for which_set in "train valid test".split():
            logger.warning("constructing %s %s monitor" % (which_set, situation))
            channels = list(graphs[situation].outputs)
            extensions.append(DataStreamMonitoring(
                channels,
                prefix="%s_%s" % (which_set, situation), after_epoch=True,
                data_stream=get_stream(which_set=which_set, for_evaluation=True,
                                       batch_size=args.batch_size,
                                       drop_prob=args.drop_prob, hidden_dim=args.num_hidden)))#, num_examples=1000)))
    for situation in "inference".split(): # add inference
        for which_set in "valid test".split():
            logger.warning("constructing %s %s monitor" % (which_set, situation))
            channels = list(graphs[situation].outputs)
            extensions.append(DataStreamMonitoring(
                channels,
                prefix="%s_%s" % (which_set, situation), after_epoch=True,
                data_stream=get_stream(which_set=which_set, for_evaluation=True,
                                       batch_size=args.batch_size,
                                       drop_prob=args.drop_prob, hidden_dim=args.num_hidden)))#, num_examples=1000)))

    extensions.extend([
        TrackTheBest("valid_training_error_rate", "best_valid_training_error_rate"),
        #DumpBest("best_valid_training_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=50),
        Checkpoint("checkpoint.zip", on_interrupt=False, every_n_epochs=1, use_cpickle=True),
        DumpLog("log.pkl", after_epoch=True)])

    if not args.cluster:
        extensions.append(ProgressBar())

    extensions.extend([
        Timing(),
        Printing(),
        PrintingTo("log"),
    ])
    train_stream = get_stream(which_set="train", for_evaluation=False,
                               batch_size=args.batch_size,
                               drop_prob=args.drop_prob, hidden_dim=args.num_hidden)
    main_loop = MainLoop(
        data_stream=train_stream,
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
