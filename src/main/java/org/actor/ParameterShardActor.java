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

        public ActorRef<DataShardActor.Command> getReplyTo() {
            return replyTo;
        }

        public FetchLatest(ActorRef<DataShardActor.Command> replyTo) {
            this.replyTo = replyTo;
        }
    }

    // === Factory Method === && === Behavior Implementation ===
    public static Behavior<Command> create(ActorRef<MasterActor.Command> parent, int inputDim, int layerDim, int latentDim, double learningRate, int epochs) {
        return Behaviors.setup(ctx ->
                new ParameterShardBehavior(ctx, parent, inputDim, layerDim, latentDim, learningRate, epochs));
    }
    static class ParameterShardBehavior extends AbstractBehavior<Command> {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        final ActorRef<MasterActor.Command> parent;
        private final int inputDim;
        private final int layerDim;
        private final int latentDim;
        private final double learningRate;
        private final int epochs;

        //====
        public ParameterShardBehavior(ActorContext<Command> context,ActorRef<MasterActor.Command> parent, int inputDim, int layerDim, int latentDim,double learningRate, int epochs) {
            super(context);
            this.learningRate = learningRate;
            this.latentDim = latentDim;
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.parent=parent;
            this.epochs = epochs;

            this.weights = new ArrayList<>();
            this.biases = new ArrayList<>();

            // Add encoder weights and bias
            weights.add(Nd4j.zeros(this.layerDim, this.inputDim));     // Input to Hidden
            biases.add(Nd4j.zeros(this.layerDim));                // Hidden bias

            //Add latent weights and bias
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden to Latent Mean
            biases.add(Nd4j.zeros(this.latentDim));               // Latent mean bias

            //Add latent weights and bias
            weights.add(Nd4j.zeros(this.latentDim, this.layerDim));     // Hidden to Latent LogVar
            biases.add(Nd4j.zeros(this.latentDim));               // Latent logvar bias

            // Add decoder weights and bias
            weights.add(Nd4j.zeros(this.layerDim, this.latentDim)); // Latent to Hidden
            biases.add(Nd4j.zeros(this.layerDim));             // Decoder hidden bias

            //Add reconstruction weights and bias
            weights.add(Nd4j.zeros(this.inputDim, this.layerDim));  // Hidden to Reconstructed Input
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
        //==He and Xavier Initialization==
        private INDArray heInit(long[] shape) {
            double std = Math.sqrt(2.0 / shape[1]);
            getContext().getLog().info("He Init: shape={}, std={}", shape, std);
            return Nd4j.randn(shape).muli(std);
        }
        private INDArray xavierInit(long[] shape) {
            double std = Math.sqrt(2.0 / (shape[0] + shape[1]));
            getContext().getLog().info("Xavier Init: shape={}, std={}", shape, std);
            return Nd4j.randn(shape).muli(std);
        }
        private void applyGradientDescent(List<INDArray> parameters, List<INDArray> gradients, double learningRate) {
            double maxNorm = 1.0; // Maximum gradient norm
            for (int i = 0; i < parameters.size(); i++) {
                INDArray gradient = gradients.get(i);
                // Compute L2 norm of the gradient
                double norm = gradient.norm2Number().doubleValue();
                if (norm > maxNorm) {
                    gradient.muli(maxNorm / norm); // Scale gradient to maxNorm
                }
                INDArray varOcg = gradient.mul(learningRate);
                parameters.set(i, parameters.get(i).sub(varOcg));
            }
        }

        // === States ===
        private Behavior<Command> onInitialize(Initialize msg) {
            for (int i = 0; i < this.weights.size(); i++) {
                long[] shape = this.weights.get(i).shape();

                // Choose Xavier or He initialization here
                INDArray initializedWeights = heInit(shape); //xavierInit(shape);//

                this.weights.set(i, initializedWeights);
                getContext().getLog().info("Weight[{}] shape: {}", i, initializedWeights.shapeInfoToString());
            }

            for (int i = 0; i < this.biases.size(); i++) {
                long[] shape = this.biases.get(i).shape();
                this.biases.set(i, Nd4j.zeros(shape));
                getContext().getLog().info("Bias[{}] shape: {}", i, this.biases.get(i).shapeInfoToString());
            }
            //getContext().getLog().info("weights {} .\n", weights);
            //getContext().getLog().info("biases {} Created.\n\n", biases);

            this.parent.tell(new MasterActor.Initialized());
            getContext().getLog().info("All weights initialized and biases set to zero.");
            return this;
        }
        private Behavior<Command> onFetchLatest(FetchLatest msg) {
            List<INDArray> weightsCopy = this.weights.stream().map(INDArray::dup).collect(Collectors.toList());
            List<INDArray> biasesCopy = this.biases.stream().map(INDArray::dup).collect(Collectors.toList());
            int epochsCopy = this.epochs;

            msg.replyTo.tell(new DataShardActor.ParametersReceived(weightsCopy, biasesCopy, epochsCopy));
            return this;
        }
        private Behavior<Command> onGradient(Gradient msg) {
            applyGradientDescent(this.weights, msg.getWeightGradients(), this.learningRate);
            applyGradientDescent(this.biases, msg.getBiasGradients(), this.learningRate);
            //getContext().getLog().info("this.weights after gradient apply= {}", this.weights);
            if(epochs >= msg.getRemainingEpochs()){
                msg.getReplyTo().tell(new DataShardActor.ParametersReceived(weights, biases, epochs));
            }
            else {
                msg.getReplyTo().tell(new DataShardActor.CompleteTraining());
            }
            return this;
        }
    }
}