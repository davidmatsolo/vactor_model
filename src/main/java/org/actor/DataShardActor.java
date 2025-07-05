package org.actor;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import org.actor.Layers.DecoderLayerActor;
import org.actor.Layers.EncoderLayerActor;
import akka.actor.typed.javadsl.Receive;
import org.actor.Layers.LatentActor;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedList;
import java.util.Queue;
import java.util.List;

public class DataShardActor {

    public interface Command {}
    public static class Initialize implements Command {}
    public static class Done implements Command {}
    public static class ComputeLoss implements Command {
        private final INDArray reconstruction;
        public ComputeLoss(INDArray reconstruction){
            this.reconstruction = reconstruction;
        }
        public INDArray getReconstruction() {
            return reconstruction;
        }
    }
    public static class InitialParametersReceived implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final int epochs;
        public InitialParametersReceived(List<INDArray> weights, List<INDArray> biases, int epochs) {
            this.weights = weights;
            this.biases = biases;
            this.epochs =epochs;
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
    public static class ReadyToProcess implements Command {}
    public static Behavior<Command> create(
            ActorRef<ParameterShardActor.Command> parameterShard,
            Queue<DataPoint> dataPoints,
            double testRatio
    ) {
        return Behaviors.setup(ctx -> new DataShardBehavior(ctx, parameterShard, dataPoints, testRatio));
    }
    static class DataShardBehavior extends AbstractBehavior<Command> {

        private   final ActorRef<ParameterShardActor.Command> parameterShard;
        private List<ActorRef<LayerActor.Command>> layers = new LinkedList<>();
        private  Queue<DataPoint> trainingData;
        private  Queue<DataPoint> validationData;
        //
        private int remainingEpochs = 0;
        private  DataPoint currentDataPoint;
        public List<ActorRef<LayerActor.Command>> getLayers() {
            return layers;
        }

        public DataShardBehavior(
                ActorContext<Command> context,
                ActorRef<ParameterShardActor.Command> parameterShard,
                Queue<DataPoint> dataPoints,
                double validationRatio
        ) {

            super(context);
            this.parameterShard = parameterShard;

            this.splitData(dataPoints, validationRatio);
            getContext().getLog().info("DataShard {} Created.", getContext().getSelf().path());
        }
        @Override
        public Receive<Command> createReceive() {

            return newReceiveBuilder()
                    .onMessage(ReadyToProcess.class, this::Process)
                    .onMessage(Initialize.class, this::onInitialize)
                    .onMessage(InitialParametersReceived.class, this::startTraining)
                    .onMessage(ComputeLoss.class, this::computeLoss)
                    .build();
        }
        //before training
        private Behavior<Command> Process(ReadyToProcess msg) {

            ActorRef<ParameterShardActor.ParameterResponse> adapter =
                    getContext().messageAdapter(ParameterShardActor.ParameterResponse.class, response ->
                            new InitialParametersReceived(response.weights, response.biases, response.epochs));

            parameterShard.tell(new ParameterShardActor.FetchLatest(adapter));
            return this;
        }
        private Behavior<Command> onInitialize(Initialize msg) {

            getContext().getLog().info("DataShard {} received initialized.", getContext().getSelf().path());

            ActorRef<LayerActor.Command> decoder = getContext().spawn(
                    DecoderLayerActor.create(parameterShard, getContext().getSelf()), "decoder-" + getContext().getSelf().path().name());
            layers.add(decoder);

            ActorRef<LayerActor.Command> latent = getContext().spawn(
                    LatentActor.create(decoder, parameterShard), "latent-" + getContext().getSelf().path().name());
            layers.add(latent);

            ActorRef<LayerActor.Command> encoder = getContext().spawn(
                    EncoderLayerActor.create(latent, parameterShard, getContext().getSelf()), "encoder-" + getContext().getSelf().path().name());
            layers.add(encoder);

            getContext().getLog().info("DataShard {} initialized.", getContext().getSelf().path());
            return this;
        }
        private void splitData(Queue<DataPoint> dataPoints, double validationRatio) {
            if (dataPoints == null || dataPoints.isEmpty()) {
                getContext().getLog().error("Empty Data points.");
                return;
            }
            this.trainingData = new LinkedList<>();
            this.validationData = new LinkedList<>();

            trainingData.clear();
            validationData.clear();

            int totalSize = dataPoints.size();
            int validationSize = (int) (totalSize * validationRatio);

            for (int i = 0; i < totalSize; i++) {
                DataPoint dp = dataPoints.poll();
                if (dp == null){
                    getContext().getLog().error("Null DataPoint encountered during data split.");
                    continue;}

                if (i < validationSize) {
                    validationData.add(dp);
                } else {
                    trainingData.add(dp);
                }
            }

            getContext().getLog().info("{} Training data size: {}", getContext().getSelf().path(), trainingData.size());
            getContext().getLog().info("{} Validation data size: {}", getContext().getSelf().path(), validationData.size());
        }
        //training operations
        private Behavior<Command> startTraining(InitialParametersReceived msg){

            currentDataPoint = trainingData.poll();
            remainingEpochs = msg.getEpochs();

            layers.get(layers.size() - 1).tell(new LayerActor.Forward(currentDataPoint.getFeatures(), msg.getWeights(), msg.getBiases()));
            return this;
        }
        private Behavior<Command> computeLoss(ComputeLoss msg){
            remainingEpochs--;
            // Compute loss: reconstruction error (e.g., MSE)
            INDArray diff = msg.getReconstruction().sub(currentDataPoint.getFeatures());
            double mse = diff.mul(diff).meanNumber().doubleValue();

            getContext().getLog().info("Reconstruction MSE: {}", mse);
            if(remainingEpochs == 0){

            }
            else {

            }
            return this;
        }
    }
}