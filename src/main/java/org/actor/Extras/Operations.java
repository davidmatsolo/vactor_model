package org.actor.Extras;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Operations {

    public static void  clipGradients(INDArray gradients, double clipNorm) {
        double globalNorm = gradients.norm2Number().doubleValue();

        if (globalNorm > clipNorm) {
            double scale = clipNorm / (globalNorm + 1e-6); // epsilon to avoid div by zero
            gradients.muli(scale); // in-place scaling
        }
    }
}
