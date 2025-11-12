package org.actor.Extras;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActivationFunctions {

    public static INDArray relu(INDArray input, boolean derivative) {
        if (derivative) {
            // derivative of ReLU: 1 if x > 0, else 0
            INDArray grad = input.gt(0).castTo(input.dataType());
            return grad;
        }
        return Transforms.relu(input, true);
    }

    public static INDArray sigmoid(INDArray input, boolean derivative) {
        INDArray sig = Transforms.sigmoid(input, true);
        if (derivative) {
            // derivative: σ(x) * (1 - σ(x))
            return sig.mul(sig.rsub(1.0));
        }
        return sig;
    }

    public static INDArray tanh(INDArray input, boolean derivative) {
        INDArray t = Transforms.tanh(input, true);
        if (derivative) {
            // derivative: 1 - tanh(x)^2
            return t.mul(t).rsub(1.0);
        }
        return t;
    }

    public static INDArray softmax(INDArray input, boolean derivative) {
        INDArray sm = Transforms.softmax(input.dup(), true);
        if (derivative) {
            // simplified elementwise gradient approximation: s * (1 - s)
            // (true Jacobian requires outer product)
            return sm.mul(sm.rsub(1.0));
        }
        return sm;
    }

    // __define-ocg__: LeakyReLU activation function
    public static INDArray leakyRelu(INDArray input, double alpha, boolean derivative) {
        if (derivative) {
            // derivative: 1 if x > 0, else alpha
            INDArray grad = input.gt(0).castTo(input.dataType());
            return grad.add(input.lte(0).castTo(input.dataType()).mul(alpha));
        }
        return Transforms.leakyRelu(input, alpha, true);
    }
}
