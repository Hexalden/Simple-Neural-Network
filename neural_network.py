"""
This code create a very simple neural network of custom size, with a retropropagation algorithm to make it learn from the target we want it to reach.

Neural networks produce very funny results while being very simple in theory. I wanted to see how I could implement mine. Like a programming challenge. It is not optimized and stay very simple, but it works.

I do not explain the theory of neural networks, as other skilled people do it pretty well. But the code should be enough commented to be understood by itself.

THe further steps for this program is to implement more customization, especially for the math part.
"""


import numpy as np
import random
import math


class Node(object):
    """A Node is the basic element of the neural network.
    """

    def __init__(self,
                 layer: int,
                 index: int):
        """A Node has:
         - an ID, which is the index of its layer, and the index of the node in this layer;
         - a list of all its input nodes;
         - a list of weight, related to each link described in the list of input nodes;
         - a bias, which is common for all the nodes of the layer;
         - a list of all its output nodes;
         - the current output value of the node;
         - the current delta of the node
        When creating a node, we only need to give it its ID (layer and index in the layer).
        """
        self.id = {"layer": layer, "index": index}
        self.input_node = []
        self.input_node_weight = []
        self.learning_rate = 0.1
        self.output_node = []
        self.output = None
        self.delta = None


    def print_node(self):
        """This function just prints the properties of the current node in a terminal-friendly way.
        """
        print("\t__________")
        print(type(self))
        print("\tNode layer " + str(self.id["layer"]) + " position " + str(self.id["index"]))
        print("\tInputs: ")
        for i in range(len(self.input_node)):
            input_node = self.input_node[i]
            print("\t\tNode layer " + str(input_node.id["layer"]) + " position " + str(input_node.id["index"]) + " with weight " + str(self.input_node_weight[i]))
        print("\tOutputs: ")
        for j in range(len(self.output_node)):
            output_node = self.output_node[j]
            print("\t\tNode layer " + str(output_node.id["layer"]) + " position " + str(output_node.id["index"]))

        print("\tNode output " + str(self.output))
        print("\t__________")

    def add_input_node(self,
                       input_node: 'Node'):
        """Add an Node in the list of inputs of the current Node.
        This function also update the input Node to add in its list of output Node the current Node.
        Finally, the function add a random number [0, 1[ in the list of the weights of the current node. This value correspond to the weight of the link between the two nodes.
        """
        self.input_node.append(input_node)
        input_node.output_node.append(self)
        self.input_node_weight.append(random.random())


    # Functions used for the forward pass in the network.
    def calculate_net_input(self) -> float:
        """This function computes the net sum of all the outputs of its linked input Nodes, taking into account the weight of each link.
        ARGUMENTS: None
        RETURN: the net sum: int
        """
        weighted_inputs = np.array([])
        for i in range(len(self.input_node)):
            input_node = self.input_node[i]
            input_node_value = input_node.output
            input_node_weight = self.input_node_weight[i]
            weighted_input = input_node_value * input_node_weight
            weighted_inputs = np.append(weighted_inputs, weighted_input)
        net_input = np.sum(weighted_inputs)
        return net_input

    def activation_function(self,
                            net_input: float) -> float:
        """Returns a value according to the net sum computed before.
        Especially, this function use the sigmoïd function as activation function.
        """
        return 1 / (1 + math.exp(-net_input))

    def execution(self):
        """Propagates the information in the Node.
        The net input is computed, which is then used in the activation function to give the state of the Node.
        """
        net_input = self.calculate_net_input()
        self.output = self.activation_function(net_input)


    # Functions used for the backpropagation in the network
    def calculate_error(self,
                        target: float) -> float:
        """Computes the error of the Node, regarding its current status and the value it should have had.
        """
        return ((target - self.output) ** 2 ) / 2

    def calculate_delta(self,
                        output_layer_deltas: [float],
                        output_layer_weights: [float]):
        """Computes the delta of the Node.
        TODO
        """
        sum = 0
        for i in range(len(output_layer_deltas)):
            sum += output_layer_deltas[i] * output_layer_weights[i]
        self.delta = sum * self.output * (1 - self.output)

    def update_weights(self,
                       target: float = None):
        """Updates the weights of the Node according to the value it should have had.
        ARGUMENT: the target, to set only if the Node does not have any succedding Node (i.e. is in the last layer), in which case the target is set externally using this argument. If the Node has succedding Nodes, it will fetch the weight of their link and their delta (needed to compute the error) automatically.
        """
        output_layer_deltas = []
        output_layer_weights = []

        if target != None:
            # A target is set, so we are in the last layer of the network.
            # In the last layer, the value (output - target) is equivalent to the delta of a "hypothetical" succedding Node. By setting the related weight to one, we have the same comportment as if we had a succedding layer in the network.
            output_layer_deltas.append(self.output - target)
            output_layer_weights.append(1)

        else:
            # We are not in the last layer of the network.
            # For each node we give our output, we need to gather the weight of the link and the delta of the succedding node.
            for output_node in self.output_node:
                output_layer_deltas.append(output_node.delta)

                # To find the weight of a link, we need first to get the index of the link in the list of input nodes of our succedding node. When found, we use this index to access the correct weight in the list of weights.
                for index, node in enumerate(output_node.input_node):
                    if node.id == self.id:
                        break
                else:
                    print("index not found")
                output_layer_weights.append(output_node.input_node_weight[index])

        # We can then compute the delta of the current node.
        self.calculate_delta(output_layer_deltas, output_layer_weights)

        # Once the delta is computed, we can process to the weight update. We update the weight of each link one after the other.
        # The new weight is computed as follows:
        #   new_weight = current_weight - (bias * (delta * output))
        for i in range(len(self.input_node)):
            input_node = self.input_node[i]
            weight = self.input_node_weight[i]
            weight_error = self.delta * input_node.output
            new_weight = self.input_node_weight[i] - (self.learning_rate * weight_error)
            self.input_node_weight[i] = new_weight



class Input_Node(Node):
    """The Input Node is just a Node which is simplified to act like an input for the network, i.e. having no Node before him but an input provided by the environment.
    The difference is that its execution is much simpler as it only forward its input value.
    As a consequence, inputs values for the network must be provided as input values of the input nodes.
    """

    def __init__(self,
                 layer: int,
                 index: int):
        """A Input Node has:
         - an ID, which is the index of its layer, and the index of the node in this layer;
         - a list of all its input nodes (this should remain empty as the Node is an input of the network);
         - a list of weight, related to each link described in the list of input nodes (this should remain empty as the Node is an input of the network);
         - a bias, which is common for all the nodes of the layer;
         - a list of all its output nodes;
         - the current output value of the node;
         - the current delta of the node
        When creating an input node, we only need to give it its ID (layer (obviously 0 here) and index in the layer).
        """
        self.id = {"layer": layer, "index": index}
        self.input_node = []
        self.input_node_weight = []
        self.output_node = []
        self.output = None
        self.delta = None

    def execution(self):
        """The execution of the input node only forward its input value to the output.
        """
        self.output = self.input



class Network(object):
    """A Network is just an organized set of Nodes.
    The Network is an abstraction of, basically, a list of nodes. Thanks to him, we only need to provide the inputs and launch the procedures to get the outputs of the network.
    """

    def __init__(self,
                 number_layers: int,
                 node_number_distribution: int):
        """To create a network, we only need to provide the number of layers (Int) for the network and the number of nodes for each layer (an array of Int, one for each layer).
        """
        assert(number_layers == len(node_number_distribution))
        self.layer = []
        for i in range(number_layers):
            layer = []
            number_of_node = node_number_distribution[i]

            for j in range(number_of_node):
                if i == 0:
                    node = Input_Node(0, j)
                else:
                    node = Node(i, j)
                    for previous_layer_node in self.layer[i - 1]:
                        node.add_input_node(previous_layer_node)
                layer.append(node)

            # Adding the bias node to the layer. The bias node is like an input node always set to 1, which will provide the next layer with a trainable constant value.
            if i > 0 and i < number_layers - 1:
                bias_node = Input_Node(i, j + 1)
                bias_node.input = 1
                layer.append(bias_node)

            self.layer.append(layer)


    def print_layer(self,
                    index_layer: int):
        """This function just prints the properties of the layer of index i in a terminal-friendly way.
        """
        print("__________")
        print("State of layer " + str(index_layer))
        layer = self.layer[index_layer]
        for node in layer:
            node.print_node()
        print("__________")

    def print_network_state(self):
        """Print the output (state) of all nodes in the network.
        """
        print("\nNetwork state:")
        for index_layer, layer in enumerate(self.layer):
            for node in layer:
                output = str(format(node.output, '.2f'))
                print(output, end='\t')
            print("")
            print("")

    def set_input(self,
                  inputs: [float]):
        """This function puts the array of inputs provided as argument as the input of the network. After this function, guess should be used to process the input through the network.
        """
        assert(len(inputs) == len(self.layer[0]))
        input_layer = self.layer[0]
        for i in range(len(inputs)):
            input_layer[i].input = inputs[i]

    def calculate_global_error(self,
                               target: [float]) -> float:
        """This function computes the global error of the network compared to a target provided as argument.
        """
        assert(len(target) == len(self.layer[-1]))
        global_error = 0
        last_layer = self.layer[-1]
        for i in range(len(last_layer)):
            global_error += last_layer[i].calculate_error(target[i])
        return global_error

    def guess(self,
              input: [float]):
        """This function computes the input through the network. After this function, get_output should be used to extract the output of the network.
        """
        self.set_input(input)
        for layer in self.layer:
            for node in layer:
                node.execution()

    def get_output(self) -> [float]:
        """This function just extract the output of the last layer of the network into an array.
        """
        output_layer = self.layer[-1]
        output = []
        for i in range(len(output_layer)):
            output.append(output_layer[i].output)
        return output

    def update_weights(self,
                       target: [float]):
        """This function do the "learning" part with a backpropagation algorithm. A target should be given as argument to make the network "learn" to reach this target.
        """
        assert(len(target) == len(self.layer[-1]))

        # We update each layer one after the other, starting by the last one.
        for i in reversed(range(len(self.layer))):
            layer = self.layer[i]

            # We update each node of the layer one after the other.
            for j in range(len(layer)):
                # To update the last layer, we need to provide a target to the update_weights function.
                if i == (len(self.layer) - 1):
                    layer[j].update_weights(target = target[j])
                # If we are not in the last layer of the network, we don't need to provide a target to the function, each node will automatically fetch the data it needs to update its weights.
                else:
                    layer[j].update_weights(target = None)


if __name__ == '__main__':
    # We specify the seed to have deterministic random
    random.seed(0)

    # We create a network
    network = Network(10, [10 for i in range(10)])

    # Input of the network
    input = [0.134, 0.5, 0.6, 0.2, 1, 0.1, 0.2, 0.546, 0.844, 0.1234]

    # Target of the network
    target = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # It is time to work!
    for loop in range(100):
        print(str(loop))

        # From the input, compute the results through the network
        network.guess(input)

        # From the target, update the weights inside the network to do a better result next time
        network.update_weights(target)

    # Print the state of the network. The last layer should approximate the target if the network have worked enough time.
    network.print_network_state()
