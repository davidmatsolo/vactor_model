package org.actor.Extras;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActivationFunctions {

    public static INDArray relu(INDArray input) {
        return Transforms.relu(input, true);
    }

    public static INDArray sigmoid(INDArray input) {
        return Transforms.sigmoid(input, true);
    }

    public static INDArray tanh(INDArray input) {
        return Transforms.tanh(input, true);
    }

    public static INDArray softmax(INDArray input) {
        return Transforms.softmax(input.dup(), true); // safe non-inplace variant
    }

    // __define-ocg__: LeakyReLU activation function
    public static INDArray leakyRelu(INDArray input, double alpha) {
        return Transforms.leakyRelu(input, alpha, true); // inplace = true
    }
}
