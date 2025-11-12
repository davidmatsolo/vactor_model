package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.actor.ParameterShardActor;
import org.actor.LayerActor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.*;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

// === Metrics collector dependency ===
import org.actor.Extras.MetricsCollectorActor;

public class LatentActor extends LayerActor {

    public static class Reparametize implements Command {
        private final INDArray zMean;
        private final INDArray zLogVar;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public Reparametize(INDArray zMean, INDArray zLogVar, List<INDArray> weights, List<INDArray> biases) {
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.weights = weights;
            this.biases = biases;
        }
        public INDArray getZMean() { return zMean; }
        public INDArray getZLogVar() { return zLogVar; }
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
    }
    private final ActorRef<MetricsCollectorActor.Command> metricsCollector;
    private final ActorRef<Command> decoder;
    private final double beta;
    INDArray std;
    INDArray epsilon;
    INDArray zSampled;
    INDArray zMean;
    INDArray zLogVar;

    // === Counters ===
    private long reparamCount = 0;
    private long backwardCount = 0;
    private long validateCount = 0;
    private long testCount = 0;

    public static Behavior<Command> create(
            ActorRef<Command> decoder,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<MetricsCollectorActor.Command> metricsCollector,
            double beta
    ) {
        return Behaviors.setup(ctx -> new LatentActor(ctx, parameterShard, decoder, metricsCollector, beta));
    }
    private LatentActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<Command> decoder,
            ActorRef<MetricsCollectorActor.Command> metricsCollector,
            double beta
    ) {
        super(context, parameterShard);
        this.decoder = decoder;
        this.metricsCollector = metricsCollector;
        this.beta = beta;
        context.getLog().info("Latent Actor {} Created.", context.getSelf().path());
    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(Reparametize.class, this::onReparametize)
                .onMessage(Backward.class, this::onBackward)
                .onMessage(TestLayer.class, this::onTestLayer)
                .onMessage(ValidateLayer.class, this::onValidate)
                .build();
    }
    //==actor states==
    private Behavior<Command> onReparametize(Reparametize msg) {
        long start = System.nanoTime();
        try {
            reparamCount++;
            zMean = msg.getZMean();
            zLogVar = msg.getZLogVar();

            // == reparametrization trick ==
            INDArray safeLogVar = Transforms.min(Transforms.max(zLogVar.dup(), Nd4j.scalar(-5)), Nd4j.scalar(2));

            std = Transforms.exp(safeLogVar.mul(0.5));
            epsilon = Nd4j.randn(std.shape());
            zSampled = zMean.dup().add(epsilon.mul(std));

            decoder.tell(new DecoderLayerActor.Decode(zSampled.dup(), zMean.dup(), zLogVar.dup(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in LatentActor onReparametize: {}", e.getMessage());
            return Behaviors.stopped();
        } finally {
            long elapsed = (System.nanoTime() - start) / 1000;
            metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                    getContext().getSelf().path().name(), "Reparametize", elapsed));
            metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                    getContext().getSelf().path().name(), reparamCount));
        }
    }
    private Behavior<Command> onValidate(ValidateLayer msg) {
        try {
            INDArray safeLogVar = Transforms.min(Transforms.max(msg.getzLogVar().dup(), Nd4j.scalar(-5)), Nd4j.scalar(2));
            std = Transforms.exp(safeLogVar.mul(0.5));
            epsilon = Nd4j.randn(std.shape());
            zSampled = msg.getzMean().dup().add(epsilon.mul(std));

            msg.setInputData(zSampled.dup());
            decoder.tell(msg);
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in LatentActor onValidate: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onTestLayer(TestLayer msg) {
        try {
            INDArray safeLogVar = Transforms.min(Transforms.max(msg.getzLogVar().dup(), Nd4j.scalar(-5)), Nd4j.scalar(2));
            std = Transforms.exp(safeLogVar.mul(0.5));
            epsilon = Nd4j.randn(std.shape());
            zSampled = msg.getzMean().dup().add(epsilon.mul(std));

            msg.setInputData(zSampled.dup());
            decoder.tell(msg);
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in LatentActor onTestLayer: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onBackward(Backward msg) {
        long start = System.nanoTime();
        try {
                backwardCount++;

                // dequeue single reconstruction gradient from the chain (from decoder)
                INDArray dL_dzSampled = msg.dequeueDelta();

                // Reconstruction path contributions to d/dmu and d/dlogvar:
                // d zSampled / d mu = 1  => dL/dmu gets dL_dzSampled
                INDArray dL_zMean_recon = dL_dzSampled.dup(); // elementwise

                // d zSampled / d logvar = 0.5 * std * epsilon  => reconstruction component
                INDArray dL_zLogVar_recon = dL_dzSampled.dup().mul(std).mul(epsilon).mul(0.5);

                // KL term gradients (for beta-VAE scaling)
                // KL = 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar )
                // dKL/dmu = mu
                INDArray dL_zMean_kl = zMean.mul(beta);

                // dKL/dlogvar = 0.5 * (exp(logvar) - 1)
                INDArray dL_zLogVar_kl = Transforms.exp(zLogVar).sub(1.0).mul(0.5 * beta);

                // total gradients
                INDArray dL_zMean = dL_zMean_recon.dup().add(dL_zMean_kl);
                INDArray dL_zLogVar = dL_zLogVar_recon.dup().add(dL_zLogVar_kl);

                // propagate back to encoder hidden (through the linear layers that produced mu/logvar)
                INDArray dh_enc_zMean = msg.getWeights().get(1).dup().transpose().mmul(dL_zMean.dup());
                INDArray dh_enc_zLogVar = msg.getWeights().get(2).dup().transpose().mmul(dL_zLogVar.dup());
                INDArray deltaToEncoder = dh_enc_zMean.dup().add(dh_enc_zLogVar);

                // Enqueue in the order Encoder expects: dh_enc, d_zMean, d_zLogVar
                msg.enqueueDelta(deltaToEncoder);
                msg.enqueueDelta(dL_zMean);
                msg.enqueueDelta(dL_zLogVar);

                // Forward the Backward message to the encoder (sendTo points to encoder)
                msg.getSendTo().tell(msg);
                return this;
        } catch (Exception ex) {
            getContext().getLog().error("Error in LatentActor onBackward: {}", ex.getMessage());
            return Behaviors.stopped();
        } finally {
            long elapsed = (System.nanoTime() - start) / 1000;
            metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                    getContext().getSelf().path().name(), "Backward", elapsed));
            metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                    getContext().getSelf().path().name(), backwardCount));
        }
    }
}
