package org.actor;

import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;
import akka.actor.typed.javadsl.*;

import org.actor.Extras.MetricsCollectorActor;
import org.actor.Extras.Operations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ParameterShardActor {

    // === Commands ===
    public interface Command {}
    public static class Initialize implements Command {}
    public static class Gradient implements Command {
        private final List<INDArray> weightGradients;
        private final List<INDArray> biasGradients;
        private final int remainingEpochs;
        private ActorRef<DataShardActor.Command> replyTo;
        public Gradient(List<INDArray> weightGradients, List<INDArray> biasGradients, int remainingEpochs, ActorRef<DataShardActor.Command> replyTo) {
            this.weightGradients = weightGradients;
            this.biasGradients = biasGradients;
            this.remainingEpochs = remainingEpochs;
            this.replyTo = replyTo;
        }

        public int getRemainingEpochs() {
            return remainingEpochs;
        }
        public ActorRef<DataShardActor.Command> getReplyTo() {
            return replyTo;
        }
        public List<INDArray> getWeightGradients() {
            return weightGradients;
        }

        public List<INDArray> getBiasGradients() {
            return biasGradients;
        }
    }
    public static class FetchLatest implements Command {
        public final ActorRef<DataShardActor.Command> replyTo;
        public ActorRef<DataShardActor.Command> getReplyTo() { return replyTo; }
        public FetchLatest(ActorRef<DataShardActor.Command> replyTo) { this.replyTo = replyTo; }
    }
    public static class FetchForTest implements Command {
        public final ActorRef<DataShardActor.Command> replyTo;
        public ActorRef<DataShardActor.Command> getReplyTo() { return replyTo; }
        public FetchForTest(ActorRef<DataShardActor.Command> replyTo) { this.replyTo = replyTo; }
    }
    // === Factory Method === && === Behavior Implementation ===
    public static Behavior<Command> create(
            ActorRef<CoordinatorActor.Command> parent,
            ActorRef<MetricsCollectorActor.Command> metricsCollector,
            int inputDim,
            int layerDim,
            int latentDim,
            double learningRate,
            int epochs
    ) {
        return Behaviors.setup(ctx ->
                new ParameterShardBehavior(ctx, parent, metricsCollector, inputDim, layerDim, latentDim, learningRate, epochs));
    }
    static class ParameterShardBehavior extends AbstractBehavior<Command> {
        private final ActorRef<MetricsCollectorActor.Command> metricsCollector;
        private final List<INDArray> vWeights = new ArrayList<>();
        private final List<INDArray> vBiases = new ArrayList<>();
        private final List<INDArray> weights = new ArrayList<>() ;
        private final List<INDArray> biases = new ArrayList<>();
        final ActorRef<CoordinatorActor.Command> parent;
        private final double momentum = 0.9;
        private final double learningRate;
        private long messageCount = 0;
        private final int latentDim;
        private final int inputDim;
        private final int layerDim;
        private final int epochs;
        //====
        public ParameterShardBehavior(
                ActorContext<Command> context,
                ActorRef<CoordinatorActor.Command> parent,
                ActorRef<MetricsCollectorActor.Command> metricsCollector,
                int inputDim,
                int layerDim,
                int latentDim,
                double learningRate,
                int epochs
        ) {
            super(context);
            this.metricsCollector = metricsCollector;
            this.learningRate = learningRate;
            this.latentDim = latentDim;
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.parent = parent;
            this.epochs = epochs;

            // Add encoder weights and bias
            weights.add(Nd4j.zeros(this.layerDim, this.inputDim));     // Input to Hidden
            biases.add(Nd4j.zeros(this.layerDim));                     // Hidden bias

            // Add latent weights and bias
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden to Latent Mean
            biases.add(Nd4j.zeros(this.latentDim));                    // Latent mean bias

            // Add latent weights and bias
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden to Latent LogVar
            biases.add(Nd4j.zeros(this.latentDim));                    // Latent logvar bias

            // Add decoder weights and bias
            weights.add(Nd4j.zeros(this.layerDim, this.latentDim));    // Latent to Hidden
            biases.add(Nd4j.zeros(this.layerDim));                     // Decoder hidden bias

            // Add reconstruction weights and bias
            weights.add(Nd4j.zeros(this.inputDim, this.layerDim));     // Hidden to Reconstructed Input
            biases.add(Nd4j.zeros(this.inputDim));                     // Reconstructed input bias

            // Initialize momentum velocities as zeros
            for (INDArray w : weights) vWeights.add(Nd4j.zerosLike(w));
            for (INDArray b : biases) vBiases.add(Nd4j.zerosLike(b));

            context.getLog().info("ParameterShardActor {} Created.", context.getSelf().path().name());
        }

        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(Initialize.class, this::onInitialize)
                    .onMessage(Gradient.class, this::onGradient)
                    .onMessage(FetchLatest.class, this::onFetchLatest)
                    .onMessage(FetchForTest.class, this::onFetchForTest)
                    .build();
        }

        //==He and Xavier Initialization==
        private INDArray heInit(long[] shape) {
            double std = Math.sqrt(2.0 / shape[1]);
            return Nd4j.randn(shape).muli(std);
        }
        private INDArray xavierInit(long[] shape) {
            double std = Math.sqrt(2.0 / (shape[0] + shape[1]));
            return Nd4j.randn(shape).muli(std);
        }
        private void applyGradientDescent(
                List<INDArray> parameters,
                List<INDArray> gradients,
                double learningRate
        ) {
            double maxNorm = 1.0; // Maximum gradient norm
            for (int i = 0; i < parameters.size(); i++) {
                INDArray gradient = gradients.get(i);
                //Operations.clipGradients(gradient, gradient.norm2Number().doubleValue());
                // Compute L2 norm of the gradient
                double norm = gradient.norm2Number().doubleValue();
                //if (norm > maxNorm) {
                //    gradient.muli(maxNorm / norm); // Scale gradient to maxNorm
                //}
                INDArray update = gradient.mul(learningRate);
                parameters.set(i, parameters.get(i).add(update));
            }
        }
        // === Momentum Update ===
        private void applySGDMomentum(List<INDArray> parameters,
                                      List<INDArray> gradients,
                                      List<INDArray> velocities,
                                      double momentum,
                                      double lr
        ) {
            for (int i = 0; i < parameters.size(); i++) {
                INDArray grad = gradients.get(i);
                //INDArray weightDecayTerm = grad.dup().mul(1e-4);
                //gradient.addi(weightDecayTerm);

                //Operations.clipGradients(grad, 0.05);  // Clip to user-defined threshold

                INDArray velocity = velocities.get(i);
                velocity.muli(momentum).addi(grad.dup().mul(lr));    // in-place update velocity
                velocities.set(i, velocity);                   // store updated velocity

                parameters.get(i).addi(velocity);             // in-place parameter update
            }
        }
        //==actor states==
        private Behavior<Command> onGradient(Gradient msg) {
            long start = System.nanoTime();
            messageCount++;
            try {
                //applyGradientDescent(this.weights, msg.getWeightGradients(), this.learningRate);
                //applyGradientDescent(this.biases, msg.getBiasGradients(), this.learningRate);
                applySGDMomentum(this.weights, msg.getWeightGradients(), this.vWeights,this.momentum, this.learningRate);
                applySGDMomentum(this.biases, msg.getBiasGradients(), this.vBiases, this.momentum, this.learningRate);

                List<INDArray> weightsCopy = this.weights.stream().map(INDArray::dup).collect(Collectors.toList());
                List<INDArray> biasesCopy = this.biases.stream().map(INDArray::dup).collect(Collectors.toList());

                if (epochs >= msg.getRemainingEpochs()) {
                    msg.getReplyTo().tell(new DataShardActor.ParametersReceived(weightsCopy,biasesCopy, epochs));
                } else {
                    msg.getReplyTo().tell(new DataShardActor.CompleteTraining(
                            this.weights.stream().map(INDArray::dup).collect(Collectors.toList()),
                            this.biases.stream().map(INDArray::dup).collect(Collectors.toList())));
                }
                return this;
            }
            catch (Exception ex){
                return  this;
            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        "ParameterShardActor", "Gradient", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        "ParameterShardActor", messageCount
                ));
            }
        }
        private Behavior<Command> onInitialize(Initialize msg) {

            try {
                for (int i = 0; i < this.weights.size(); i++) {
                    long[] shape = this.weights.get(i).shape();
                    INDArray initializedWeights =   heInit(shape); // xavierInit(shape); //
                    this.weights.set(i, initializedWeights);
                }
                for (int i = 0; i < this.biases.size(); i++) {
                    long[] shape = this.biases.get(i).shape();
                    this.biases.set(i, Nd4j.zeros(shape));
                }
                this.parent.tell(new CoordinatorActor.Initialized());
                getContext().getLog().info("All weights initialized and biases set to zero.");
                return this;
            } catch (Exception ex){
                return  this;
            }
        }
        private Behavior<Command> onFetchLatest(FetchLatest msg) {
            long start = System.nanoTime();
            messageCount++;

            try {
                List<INDArray> weightsCopy = this.weights.stream().map(INDArray::dup).collect(Collectors.toList());
                List<INDArray> biasesCopy = this.biases.stream().map(INDArray::dup).collect(Collectors.toList());
                int epochsCopy = this.epochs;

                msg.replyTo.tell(new DataShardActor.ParametersReceived(
                        weightsCopy, biasesCopy, epochsCopy));
                return this;
            } finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        "ParameterShardActor", "FetchLatest", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        "ParameterShardActor", messageCount
                ));
            }
        }
        private Behavior<Command> onFetchForTest(FetchForTest msg) {

            try {
                msg.replyTo.tell(new DataShardActor.TestParameters(
                        this.weights.stream().map(INDArray::dup).collect(Collectors.toList()),
                        this.biases.stream().map(INDArray::dup).collect(Collectors.toList())));
                return this;
            }catch (Exception ex){
                return this;
            }
        }
    }
}
