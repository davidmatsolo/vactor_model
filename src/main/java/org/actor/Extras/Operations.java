package org.actor.Extras;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

public class Operations {

    /**
     * Clips the gradients by global L2 norm to the given clipNorm threshold.
     *
     * @param gradients INDArray of gradients
     * @param clipNorm  maximum allowed L2 norm
     */
    public static void clipGradients(INDArray gradients, double clipNorm) {
        double globalNorm = gradients.norm2Number().doubleValue();

        if (globalNorm > clipNorm) {
            double scale = clipNorm / (globalNorm + 1e-8); // safer epsilon
            gradients.muli(scale); // in-place scaling
        }
    }

    /**
     * Computes gradients of KL divergence with respect to mu and logvar.
     *
     * KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
     * ∂KL/∂mu = mu
     * ∂KL/∂logvar = 0.5 * (exp(logvar) - 1)
     *
     * @param mu      latent mean
     * @param logvar  latent log-variance
     * @return array of gradients: [d_mu, d_logvar]
     */
    public static INDArray[] klGrad(INDArray mu, INDArray logvar) {
        INDArray dMu = mu.dup();  // ∂KL/∂mu = mu

        // ∂KL/∂logvar = 0.5 * (exp(logvar) - 1)
        INDArray dLogvar = Transforms.exp(logvar, true);  // in-place exp
        dLogvar.subi(1.0).muli(0.5);                       // (exp - 1) * 0.5

        return new INDArray[]{dMu, dLogvar};
    }
    public static void plot(String title, List<Double> mseHistory) {
        int numIterations = mseHistory.size();
        double[] iterations = new double[numIterations];
        double[] mseValues = new double[numIterations];

        for (int i = 0; i < numIterations; i++) {
            iterations[i] = i;
            mseValues[i] = mseHistory.get(i);
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(title)
                .xAxisTitle("Iteration")
                .yAxisTitle("MSE")
                .build();

        chart.addSeries("MSE", iterations, mseValues);
        new SwingWrapper<>(chart).displayChart();
    }

}
