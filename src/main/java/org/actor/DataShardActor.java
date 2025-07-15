package org.actor;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.actor.Layers.DecoderLayerActor;
import org.actor.Layers.EncoderLayerActor;
import org.actor.Layers.LatentActor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.LinkedList;
import java.util.Queue;
import java.util.List;

public class DataShardActor {

    //==commands state messages==
    public interface Command {}
    public static class InitialParametersReceived implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final int epochs;

        public InitialParametersReceived(List<INDArray> weights, List<INDArray> biases, int epochs) {
            this.weights = weights;
            this.biases = biases;
            this.epochs = epochs;
        }

        public List<INDArray> getBiases() {
            return biases;
        }

        public List<INDArray> getWeights() {
            return weights;
        }

        public int getEpochs() {
            return epochs;
        }
    }
    public static class ComputeLossWithReply implements Command {
        private final INDArray reconstruction;
        private final INDArray zMean;
        private final INDArray zLogVar;
        private final ActorRef<DecoderLayerActor.Command> replyTo;

        public ComputeLossWithReply(INDArray reconstruction, INDArray zMean, INDArray zLogVar, ActorRef<LayerActor.Command> replyTo) {
            this.reconstruction = reconstruction;
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.replyTo = replyTo;
        }
        public INDArray getReconstruction() {
            return reconstruction;
        }
        public INDArray getZMean() {
            return zMean;
        }
        public INDArray getZLogVar() {
            return zLogVar;
        }
        public ActorRef<LayerActor.Command> getReplyTo() {
            return replyTo;
        }
    }
    public static class ParametersReceived implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final int epochs;

        public ParametersReceived(List<INDArray> weights, List<INDArray> biases, int epochs) {
            this.weights = weights;
            this.biases = biases;
            this.epochs = epochs;
        }

        public List<INDArray> getBiases() {
            return biases;
        }

        public List<INDArray> getWeights() {
            return weights;
        }

        public int getEpochs() {
            return epochs;
        }
    }
    public static class DataPointProcessed implements  Command{
        private List<INDArray> weightGradients;
        private List<INDArray> biasesGradients;

        public  DataPointProcessed(List<INDArray> weightGrads, List<INDArray> biasesGrads){
            this.weightGradients = weightGrads;
            this.biasesGradients = biasesGrads;
        }

        public List<INDArray> getWeightGradients() {
            return weightGradients;
        }

        public List<INDArray> getBiasesGradients() {
            return biasesGradients;
        }
    }
    public static class CompleteTraining implements Command {

    }
    public static class ReadyToProcess implements Command {}
    public static class CreateLayers implements Command {}
    //====
    public static Behavior<Command> create(ActorRef<MasterActor.Command> parent, ActorRef<ParameterShardActor.Command> parameterShard, List<DataPoint> dataPoints, double testRatio, double beta) {
        return Behaviors.setup(ctx -> new DataShardBehavior(ctx, parent, parameterShard, dataPoints, testRatio, beta));
    }
    static class DataShardBehavior extends AbstractBehavior<Command> {
        private List<ActorRef<LayerActor.Command>> layers = new LinkedList<>();
        private final ActorRef<ParameterShardActor.Command> parameterShard;
        final ActorRef<MasterActor.Command> parent;
        private List<DataPoint> validationData;
        private List<DataPoint> trainingData;
        private DataPoint currentDataPoint;
        private int remainingEpochs;
        private final double beta;
        int dataPointsProcessed;
        //====
        public DataShardBehavior(ActorContext<Command> context,ActorRef<MasterActor.Command> parent, ActorRef<ParameterShardActor.Command> parameterShard, List<DataPoint> dataPoints, double validationRatio, double beta) {
            super(context);
            this.splitData(dataPoints, validationRatio);
            this.parameterShard = parameterShard;
            this.dataPointsProcessed =0;
            this.remainingEpochs =0;
            this.parent = parent;
            this.beta = beta;

            getContext().getLog().info("DataShard {} Created.\n", getContext().getSelf().path());
        }
        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(CreateLayers.class, this::createLayers)
                    .onMessage(ReadyToProcess.class, this::onProcess)
                    .onMessage(InitialParametersReceived.class, this::initializeTrainModel)
                    .onMessage(ParametersReceived.class, this::trainModel)
                    .onMessage(ComputeLossWithReply.class, this::computeLossWithReply)
                    .onMessage(DataPointProcessed.class, this::onDataPointProcessed)
                    .onMessage(CompleteTraining.class, this::onDone)
                    .build();
        }
        // ===states===
        private Behavior<Command> initializeTrainModel(InitialParametersReceived msg) {
            this.currentDataPoint = trainingData.get(dataPointsProcessed);
            dataPointsProcessed =dataPointsProcessed + 1;
            remainingEpochs=remainingEpochs+1;

            //getContext().getLog().info("target = {}", currentDataPoint.getFeatures());
            layers.get(layers.size() - 1).tell(new LayerActor.Forward(this.currentDataPoint.getFeatures(), msg.getWeights(), msg.getBiases()));
            return this;
        }
        private void splitData(List<DataPoint> dataPoints, double validationRatio) {
            if (dataPoints == null || dataPoints.isEmpty()) {
                getContext().getLog().error("Empty Data points.");
                return;
            }

            trainingData = new LinkedList<>();
            validationData = new LinkedList<>();
            int totalSize = dataPoints.size();
            int validationSize = (int) (totalSize * validationRatio);

            for (int i = 0; i < totalSize; i++) {
                DataPoint dp = dataPoints.get(i);
                if (dp == null) continue;
                if (i < validationSize) validationData.add(dp);
                else trainingData.add(dp);
            }

            getContext().getLog().info("{} Training data size: {}", getContext().getSelf().path(), trainingData.size());
            getContext().getLog().info("{} Validation data size: {}", getContext().getSelf().path(), validationData.size());
        }
        private Behavior<Command> computeLossWithReply(ComputeLossWithReply msg) {
            if (msg.getReconstruction() == null || msg.getZMean() == null || msg.getZLogVar() == null) {
                getContext().getLog().error("Received null values in ComputeLossWithReply.");
                return this;
            }
            try{
                INDArray reconstruction = msg.getReconstruction().reshape(msg.getReconstruction().length());

                // === Compute MSE ===
                INDArray diff = reconstruction.sub(currentDataPoint.getFeatures());
                double mse = diff.mul(diff).meanNumber().doubleValue();

                // === Compute KL Divergence ===
                INDArray var = Transforms.exp(msg.getZLogVar());
                INDArray klLoss = var.add(msg.getZMean().mul(msg.getZMean())).sub(msg.getZLogVar()).sub(1).mul(0.5);
                double kl = klLoss.meanNumber().doubleValue();

                // === Combine Loss ==
                double total = mse + beta * kl;
                getContext().getLog().info("output = {}", reconstruction);
                getContext().getLog().info("target = {}",currentDataPoint.getFeatures());
                getContext().getLog().info("Received Loss: total={}, mse={}, kl={}", total, mse, kl);

                msg.getReplyTo().tell(new DecoderLayerActor.LossResponse(mse, kl, total,currentDataPoint.getFeatures(), layers.get(1), layers.get(2)));
                return this;
            }catch (Exception ex){
                getContext().getLog().error("Error in computeLossWithReply: {}", ex.getMessage());
                return this;
            }
        }
        private Behavior<Command> onDataPointProcessed(DataPointProcessed msg) {
            if (dataPointsProcessed >= trainingData.size()) {
                dataPointsProcessed = 0;
                remainingEpochs++;
            }
            parameterShard.tell(new ParameterShardActor.Gradient(msg.getWeightGradients(), msg.getBiasesGradients(),remainingEpochs, getContext().getSelf()));
            return this;
        }
        private Behavior<Command> trainModel(ParametersReceived msg) {
            this.currentDataPoint = trainingData.get(dataPointsProcessed);
            dataPointsProcessed =dataPointsProcessed + 1;
            layers.get(layers.size() - 1).tell(new LayerActor.Forward(this.currentDataPoint.getFeatures(), msg.getWeights(), msg.getBiases()));
            return this;
        }
        private Behavior<Command> createLayers(CreateLayers msg) {
            try {
                ActorRef<LayerActor.Command> decoder = getContext().spawn(
                        DecoderLayerActor.create(parameterShard, getContext().getSelf()),
                        "decoder-" + getContext().getSelf().path().name()
                );
                layers.add(decoder);

                ActorRef<LayerActor.Command> latent = getContext().spawn(
                        LatentActor.create(decoder, parameterShard),
                        "latent-" + getContext().getSelf().path().name()
                );
                layers.add(latent);

                ActorRef<LayerActor.Command> encoder = getContext().spawn(
                        EncoderLayerActor.create(latent, parameterShard, getContext().getSelf()),
                        "encoder-" + getContext().getSelf().path().name()
                );
                layers.add(encoder);
                getContext().getSelf().tell(new DataShardActor.ReadyToProcess());

            } catch (Exception ex) {
                getContext().getLog().error("Matrix operation failed: {}", ex.getMessage());
            }
            return this;
        }
        private Behavior<Command> onProcess(ReadyToProcess msg) {
            parameterShard.tell(new ParameterShardActor.FetchLatest(getContext().getSelf()));
            return this;
        }
        private Behavior<Command> onDone(CompleteTraining msg){
            parent.tell(new MasterActor.Done());
            return this;
        }
    }
}
