import dynet as dy
import random as r


def create_xor_dataset(num_samples=2000):
    # Get the data for xor
    x = []
    y = []
    for _ in range(num_samples):
        x1 = r.randint(0, 1)
        x2 = r.randint(0, 1)
        x.append((x1, x2))
        if x1 == x2:
            y.append(0)
        else:
            y.append(1)
    return x, y

X, y = create_xor_dataset(5)


def create_xor_network(mW, mb, mV, input, expected_output):
    # For each input, create the computational graph and get the loss
    dy.renew_cg()
    W = dy.parameter(mW)
    b = dy.parameter(mb)
    V = dy.parameter(mV)
    x = dy.vecInput(len(input))
    x.set(input)
    y = dy.scalarInput(expected_output)
    graph_output = dy.logistic(V*(dy.tanh(W*x+b)))
    loss = dy.binary_log_loss(graph_output, y)
    return loss


# Define model paramentes
model = dy.Model()
mW = model.add_parameters((8, 2))
mb = model.add_parameters(8)
mV = model.add_parameters((1, 8))
trainer = dy.SimpleSGDTrainer(model)

# Iterate and for each example in the dataset, calculate the loss
seen_instances = 0
total_loss = 0
ITER = 1000
for it in range(ITER):
    for x_, y_ in zip(X, y):
        loss = create_xor_network(mW, mb, mV, x_, y_)
        total_loss += loss.value()
        seen_instances += 1
        loss.backward()
        trainer.update()
        if seen_instances > 1 and seen_instances%100 == 0:
            print("average loss = {}".format(total_loss/seen_instances))
