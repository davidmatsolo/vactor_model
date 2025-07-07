package org.actor;

import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;
import akka.actor.typed.javadsl.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

        public Gradient(List<INDArray> weightGradients, List<INDArray> biasGradients) {
            this.weightGradients = weightGradients;
            this.biasGradients = biasGradients;
        }

        public List<INDArray> getWeightGradients() {
            return weightGradients;
        }

        public List<INDArray> getBiasGradients() {
            return biasGradients;
        }
    }
    public static class FetchLatest implements Command {
        public final ActorRef<ParameterResponse> replyTo;

        public ActorRef<ParameterResponse> getReplyTo() {
            return replyTo;
        }

        public FetchLatest(ActorRef<ParameterResponse> replyTo) {
            this.replyTo = replyTo;
        }
    }
    public static class ParameterResponse {
        public final List<INDArray> weights;
        public final List<INDArray> biases;
        public final int epochs;

        public ParameterResponse(List<INDArray> weights, List<INDArray> biases, int epochs) {
            this.weights = weights;
            this.biases = biases;
            this.epochs = epochs;
        }
    }

    // === Factory Method === && === Behavior Implementation ===
    public static Behavior<Command> create(int inputDim, int layerDim, int latentDim, double learningRate, int epochs) {
        return Behaviors.setup(ctx ->
                new ParameterShardBehavior(ctx, inputDim, layerDim, latentDim, learningRate, epochs));
    }
    static class ParameterShardBehavior extends AbstractBehavior<Command> {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final int inputDim;
        private final int layerDim;
        private final int latentDim;
        private final double learningRate;
        private final int epochs;

        //
        public ParameterShardBehavior(ActorContext<Command> context, int inputDim, int layerDim, int latentDim,double learningRate, int epochs) {
            super(context);
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.latentDim = latentDim;
            this.learningRate = learningRate;
            this.epochs = epochs;

            this.weights = new ArrayList<>();
            this.biases = new ArrayList<>();

            // Add encoder weights
            weights.add(Nd4j.zeros(this.layerDim, this.inputDim));     // Input → Hidden
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden → Latent Mean
            // Add encoder biases
            biases.add(Nd4j.zeros(this.layerDim));                // Hidden bias
            biases.add(Nd4j.zeros(this.latentDim));               // Latent mean bias

            //Add latent weights
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden → Latent LogVar
            //Add latent biases
            biases.add(Nd4j.zeros(this.latentDim));               // Latent logvar bias

            // Add decoder weights
            weights.add(Nd4j.zeros(this.layerDim, this.latentDim)); // Latent → Hidden
            weights.add(Nd4j.zeros(this.inputDim, this.layerDim));  // Hidden → Reconstructed Input
            // Add decoder biases
            biases.add(Nd4j.zeros(this.layerDim));             // Decoder hidden bias
            biases.add(Nd4j.zeros(this.inputDim));             // Reconstructed input bias


            context.getLog().info("ParameterShardActor {} Created.", context.getSelf().path().name());
        }
        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(Initialize.class, this::onInitialize)
                    .onMessage(Gradient.class, this::onGradient)
                    .onMessage(FetchLatest.class, this::onFetchLatest)
                    .build();
        }

        // === States ===
        private Behavior<Command> onInitialize(Initialize msg) {
            getContext().getLog().info("Initializing weights and biases...");

            for (int i = 0; i < weights.size(); i++) {
                weights.set(i, Nd4j.rand(weights.get(i).shape()).mul(2).sub(1)); // values in [-1, 1]
            }

            for (int i = 0; i < biases.size(); i++) {
                biases.set(i, Nd4j.rand(biases.get(i).shape()).mul(2).sub(1));
            }

            getContext().getLog().info("Initialized weights and biases.");
            return this;
        }
        private Behavior<Command> onFetchLatest(FetchLatest msg) {
            List<INDArray> weightsCopy = weights.stream().map(INDArray::dup).collect(Collectors.toList());
            List<INDArray> biasesCopy = biases.stream().map(INDArray::dup).collect(Collectors.toList());
            int epochsCopy = epochs;
            msg.replyTo.tell(new ParameterResponse(weightsCopy, biasesCopy, epochsCopy));
            return this;
        }
        private Behavior<Command> onGradient(Gradient msg) {
            for (int i = 0; i < weights.size(); i++) {
                weights.set(i, weights.get(i).sub(msg.getWeightGradients().get(i).mul(learningRate)));
            }

            for (int i = 0; i < biases.size(); i++) {
                biases.set(i, biases.get(i).sub(msg.getBiasGradients().get(i).mul(learningRate)));
            }

            getContext().getLog().info("Updated weights and biases using gradients.");
            return this;
        }

    }
}