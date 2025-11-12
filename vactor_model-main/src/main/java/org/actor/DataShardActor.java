package org.actor;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.actor.Extras.MetricsCollectorActor;
import org.actor.Extras.Operations;
import org.actor.Layers.DecoderLayerActor;
import org.actor.Layers.EncoderLayerActor;
import org.actor.Layers.LatentActor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.List;
import java.time.format.DateTimeFormatter;

public class DataShardActor {

    //==commands state messages==
    public interface Command {}
    public static class EvaluateValidationResult implements Command {
        private final INDArray reconstruction;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray zSampled;
        private final INDArray zLogVar;
        private final INDArray zMean;
        //===
        public EvaluateValidationResult(INDArray reconstruction, INDArray zSampled, INDArray zMean, INDArray zLogVar, List<INDArray> weights, List<INDArray> biases) {
            this.reconstruction = reconstruction;
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.weights = weights;
            this.biases = biases;
            this.zSampled = zSampled;
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
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
        public INDArray getzSampled() { return zSampled;}
    }
    public static class ComputeLossWithReply implements Command {
        private final ActorRef<DecoderLayerActor.Command> replyTo;
        private final INDArray reconstruction;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray zSampled;
        private final INDArray zLogVar;
        private final INDArray zMean;
        //===
        public ComputeLossWithReply(INDArray reconstruction, INDArray zSampled, INDArray zMean, INDArray zLogVar, ActorRef<LayerActor.Command> replyTo, List<INDArray> weights, List<INDArray> biases) {
            this.reconstruction = reconstruction;
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.replyTo = replyTo;
            this.weights = weights;
            this.biases = biases;
            this.zSampled = zSampled;
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
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
        public INDArray getzSampled() { return zSampled;}
    }
    public static class EvaluateTestResult implements Command {
        private final INDArray reconstruction;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray zSampled;
        private final INDArray zLogVar;
        private final INDArray zMean;
        //===
        public EvaluateTestResult(INDArray reconstruction, INDArray zSampled, INDArray zMean, INDArray zLogVar, List<INDArray> weights, List<INDArray> biases) {
            this.reconstruction = reconstruction;
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.weights = weights;
            this.biases = biases;
            this.zSampled = zSampled;
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
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
        public INDArray getzSampled() { return zSampled;}
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
    public static class ValidateParameters implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        public ValidateParameters(List<INDArray> weights, List<INDArray> biases) {
            this.weights = weights;
            this.biases = biases;
        }
        public List<INDArray> getBiases() {
            return biases;
        }
        public List<INDArray> getWeights() {
            return weights;
        }

    }
    public static class CompleteTraining implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        public CompleteTraining(List<INDArray> weights, List<INDArray> biases) {
            this.weights = weights;
            this.biases = biases;
        }
        public List<INDArray> getBiases() {
            return biases;
        }
        public List<INDArray> getWeights() {
            return weights;
        }

    }
    public static class TestParameters implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        public TestParameters(List<INDArray> weights, List<INDArray> biases) {
            this.weights = weights;
            this.biases = biases;
        }

        public List<INDArray> getBiases() {
            return biases;
        }
        public List<INDArray> getWeights() {
            return weights;
        }

    }
    public static class ReadyToProcess implements Command {}
    public static class StartTest implements Command {
        List<DataPoint> testingData;
        public StartTest(List<DataPoint> testingData) {
            this.testingData = testingData;
        }

        public List<DataPoint> getTestingData() {
            return testingData;
        }
    }
    public static class CreateLayers implements Command {}
    //====
    public static Behavior<Command> create(
            ActorRef<CoordinatorActor.Command> parent,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<MetricsCollectorActor.Command> metricsLogger,
            List<DataPoint> dataPoints,
            double testRatio,
            double beta,
            int batchSize
    ) {
        return Behaviors.setup(ctx -> new DataShardBehavior(ctx, parent, parameterShard, metricsLogger, dataPoints, testRatio, beta, batchSize));
    }
    static class DataShardBehavior extends AbstractBehavior<Command> {
        private List<ActorRef<LayerActor.Command>> layers = new LinkedList<>();
        private final ActorRef<ParameterShardActor.Command> parameterShard;
        private final ActorRef<MetricsCollectorActor.Command> metricsCollector;
        final ActorRef<CoordinatorActor.Command> parent;
        private final List<DataPoint> validationData;
        private final List<DataPoint> trainingData;
        private List<DataPoint> testingData;
        private DataPoint currentDataPoint;
        private int remainingEpochs;
        private final double beta;
        int dataPointsProcessed;
        List<Double> mseHistory;
        private final int batchSize;
        Queue<DataPoint> batch;
        LocalDateTime dateTime;
        private long messageCount;
        //====
        public DataShardBehavior(
                ActorContext<Command> context,
                ActorRef<CoordinatorActor.Command> parent,
                ActorRef<ParameterShardActor.Command> parameterShard,
                ActorRef<MetricsCollectorActor.Command> metricsShard,
                List<DataPoint> dataPoints,
                double validationRatio,
                double beta,
                int batchSize
        ) {
            super(context);

            trainingData = new LinkedList<>();
            validationData = new LinkedList<>();

            this.mseHistory = new ArrayList<>();
            this.batch = null;//new ArrayList<>();
            this.splitData(dataPoints, validationRatio);
            //this.trainingData = dataPoints;
            this.parameterShard = parameterShard;
            this.metricsCollector = metricsShard;
            this.dataPointsProcessed =0;
            this.remainingEpochs =1;
            this.parent = parent;
            this.beta = beta;
            this.batchSize = batchSize;
            this.dateTime = LocalDateTime.now();
            this.messageCount =0;

            getContext().getLog().info("DataShard {} Created.\n", getContext().getSelf().path());
        }
        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(CreateLayers.class, this::createLayers)
                    .onMessage(ReadyToProcess.class, this::onProcess)
                    .onMessage(ParametersReceived.class, this::trainModel)
                    .onMessage(ComputeLossWithReply.class, this::computeLossWithReply)
                    .onMessage(DataPointProcessed.class, this::onDataPointProcessed)
                    .onMessage(EvaluateTestResult.class, this::evaluateTestResult)
                    .onMessage(EvaluateValidationResult.class, this::evaluateValidationResult)
                    .onMessage(ValidateParameters.class, this::validateModel)
                    .onMessage(CompleteTraining.class, this::onDone)
                    .onMessage(TestParameters.class, this::testModel)
                    .onMessage(StartTest.class, this::onStartTest)
                    .build();
        }

        //======
        private Queue<DataPoint> getRandomBatch() {
            Queue<DataPoint> batch = new LinkedList<>();

            if (trainingData.isEmpty()) {
                return batch; // empty
            }

            int maxStart = Math.max(0, trainingData.size() - batchSize);
            int start = new java.util.Random().nextInt(maxStart + 1);

            /*for (int i = start; i < start + batchSize && i < trainingData.size(); i++) {
                batch.add(trainingData.get(i));
            }*/
            for (DataPoint dp: trainingData) {
                batch.add(dp);

            }

            return batch;
        }
        private void splitData(
                List<DataPoint> dataPoints,
                double validationRatio
        ) {
            if (dataPoints == null || dataPoints.isEmpty()) {
                getContext().getLog().error("Empty Data points.");
                return;
            }


            int totalSize = dataPoints.size();
            int validationSize = Math.max(1, (int) (totalSize * validationRatio));
            validationSize = Math.min(validationSize, totalSize - 1);

            for (int i = 0; i < totalSize; i++) {
                DataPoint dp = dataPoints.get(i);
                if (dp == null) continue;
                if (i < validationSize) validationData.add(dp);
                else trainingData.add(dp);
            }

            getContext().getLog().info("{} Training data size: {}", getContext().getSelf().path(), trainingData.size());
            getContext().getLog().info("{} Validation data size: {}", getContext().getSelf().path(), validationData.size());
/*
            // Inside DataShardBehavior constructor, after splitData call
            getContext().getLog().info("=== First 5 Training DataPoints After Sharding ===");
            for (int idx = 0; idx < Math.min(5, trainingData.size()); idx++) {
                DataPoint dp = trainingData.get(idx);
                getContext().getLog().info("DataPoint {}: features={}, label={}", idx, dp.getFeatures(), dp.getLabel());
            }

            getContext().getLog().info("=== First 5 Validation DataPoints After Sharding ===");
            for (int idx = 0; idx < Math.min(5, validationData.size()); idx++) {
                DataPoint dp = validationData.get(idx);
                getContext().getLog().info("DataPoint {}: features={}, label={}", idx, dp.getFeatures(), dp.getLabel());
            }*/
        }

        // ===states===
        private Behavior<Command> evaluateValidationResult(EvaluateValidationResult msg) {

            try {
                if (msg.getReconstruction() == null || msg.getZMean() == null || msg.getZLogVar() == null) {
                    getContext().getLog().error("Received null values in evaluateValidationResult.");
                    return this;
                }
                INDArray reconstruction = msg.getReconstruction();

                // === Compute MSE ===
                INDArray original = currentDataPoint.getFeatures().dup();
                INDArray diff = reconstruction.dup().sub(original);
                double mse = diff.mul(diff).meanNumber().doubleValue();

                // === Compute KL Divergence ===
                INDArray var = Transforms.exp(msg.getZLogVar());
                INDArray klLoss = var.add(msg.getZMean().mul(msg.getZMean())).sub(msg.getZLogVar()).sub(1).mul(0.5);
                double kl = klLoss.meanNumber().doubleValue();

                // === Combine Loss ==
                double total = mse + beta * kl;
                getContext().getLog().info("Validation Loss: total={}, mse={}, kl={}", total, mse, kl);

                //===Save to File===
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
                String filePath = "D:/MSc_RESEARCH/prototype/code/analysis/validation_data_output/valating_data_results_" + getContext().getSelf().path().name() + dateTime.format(formatter) + ".csv";
                File file = new File(filePath);
                boolean append = file.exists();

                Operations.saveTestingOrginalReconLatentData(filePath, currentDataPoint.getFeatures(), reconstruction, msg.getzSampled(), total, kl, mse, currentDataPoint.getLabel(), append);

                //====
                getContext().getSelf().tell(new DataShardActor.ValidateParameters(msg.getWeights(), msg.getBiases()));
                return this;
            } catch (Exception ex) {
                getContext().getLog().error("Error in evaluateValidationResult: {}", ex.getMessage());
                return this;
            }
        }
        private Behavior<Command> computeLossWithReply(ComputeLossWithReply msg) {
            long start = System.nanoTime();
            messageCount++;

            try{
                if (msg.getReconstruction() == null || msg.getZMean() == null || msg.getZLogVar() == null) {
                    getContext().getLog().error("Received null values in ComputeLossWithReply.");
                    return this;
                }

                INDArray reconstruction = msg.getReconstruction();

                // === Compute MSE ===
                INDArray original = currentDataPoint.getFeatures().dup();
                INDArray diff = original.sub(reconstruction);
                double mse = diff.mul(diff).meanNumber().doubleValue();

                // === Compute KL Divergence ===
                INDArray var = Transforms.exp(msg.getZLogVar());
                INDArray klLoss = var.add(msg.getZMean().mul(msg.getZMean())).sub(msg.getZLogVar()).sub(1).mul(0.5);
                double kl = klLoss.meanNumber().doubleValue();

                // === Combine Loss ==
                double total = mse + beta * kl;
                //getContext().getLog().info("klLoss = {}", klLoss);
                getContext().getLog().info("output = {}", reconstruction);
                getContext().getLog().info("target after calculation = {}",currentDataPoint.getFeatures());
                //getContext().getLog().info("diff = {}\n",diff);
                getContext().getLog().info("Received Loss: total={}, mse={}, kl={}", total, mse, kl);

                //===Save to File===
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
                String filePath = "D:/MSc_RESEARCH/prototype/code/analysis/training_data_output/training_data_results_"
                        + getContext().getSelf().path().name() + dateTime.format(formatter) + ".csv";
                File file = new File(filePath);
                boolean append = file.exists();

                Operations.saveOrginalReconLatentData( filePath, currentDataPoint.getFeatures().dup(), reconstruction, msg.getzSampled(),total, kl, mse, append );

                // === Send Loss Response ===
                mseHistory.add(mse);
                msg.getReplyTo().tell(new DecoderLayerActor.LossResponse(beta, currentDataPoint.getFeatures(), layers.get(1),layers.get(2),
                        msg.getWeights(), msg.getBiases()));
                return this;
            }catch (Exception ex){
                getContext().getLog().error("Error in computeLossWithReply: {}", ex.getMessage());
                return this;
            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        getContext().getSelf().path().name(), "ComputeLossWithReply", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        getContext().getSelf().path().name(), messageCount
                ));
            }
        }
        private Behavior<Command> onDataPointProcessed(DataPointProcessed msg) {

            long start = System.nanoTime();
            messageCount++;
            try {

                dataPointsProcessed++;
                if (dataPointsProcessed >= trainingData.size()) {
                    dataPointsProcessed = 0;
                    remainingEpochs++;
                }
                parameterShard.tell(new ParameterShardActor.Gradient(msg.getWeightGradients(), msg.getBiasesGradients(), remainingEpochs, getContext().getSelf()));
            }
            catch (Exception ex){

            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        getContext().getSelf().path().name(), "DataPointProcessed", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        getContext().getSelf().path().name(), messageCount
                ));
            }
            return this;
        }
        private Behavior<Command> evaluateTestResult(EvaluateTestResult msg) {
            try{

                if (msg.getReconstruction() == null || msg.getZMean() == null || msg.getZLogVar() == null) {
                    getContext().getLog().error("Received null values in ComputeLossWithReply.");
                    return this;
                }
                INDArray reconstruction = msg.getReconstruction();

                // === Compute MSE ===
                INDArray diff = reconstruction.dup().sub(currentDataPoint.getFeatures().dup());
                double mse = diff.mul(diff).meanNumber().doubleValue();

                // === Compute KL Divergence ===
                INDArray var = Transforms.exp(msg.getZLogVar().dup());
                INDArray klLoss = var.add(msg.getZMean().mul(msg.getZMean())).sub(msg.getZLogVar()).sub(1).mul(0.5);
                double kl = klLoss.meanNumber().doubleValue();

                // === Combine Loss ==
                double total = mse + beta * kl;
                getContext().getLog().info("Testing Loss: total={}, mse={}, kl={}", total, mse, kl);

                //===Save to File===
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
                String filePath = "D:/MSc_RESEARCH/prototype/code/analysis/testing_data_output/testing_data_results_"+ getContext().getSelf().path().name() + dateTime.format(formatter) + ".csv";
                File file = new File(filePath);
                boolean append = file.exists();

                Operations.saveTestingOrginalReconLatentData( filePath, currentDataPoint.getFeatures().dup(), reconstruction, msg.getzSampled(),total, kl, mse,currentDataPoint.getLabel(), append );

                //====
                getContext().getSelf().tell(new DataShardActor.TestParameters(msg.getWeights(), msg.getBiases()));
                return this;
            }catch (Exception ex){
                getContext().getLog().error("Error in computeLossWithReply: {}", ex.getMessage());
                return this;
            }
        }
        private Behavior<Command> validateModel(ValidateParameters msg) {
            if (validationData.size() == dataPointsProcessed) {
                dataPointsProcessed =0;
                parent.tell(new CoordinatorActor.DoneTraining());
                return this;
            }

            currentDataPoint = validationData.get(dataPointsProcessed);
            dataPointsProcessed++;

            layers.get(layers.size() - 1).tell(
                    new LayerActor.ValidateLayer(currentDataPoint.getFeatures().dup(), msg.getWeights(), msg.getBiases())
            );
            return this;
        }
        private Behavior<Command> trainModel(ParametersReceived msg) {
            long start = System.nanoTime();
            messageCount++;
            try {
                if (batch == null || batch.isEmpty()) {
                    batch = getRandomBatch();
                }

                currentDataPoint = batch.poll();

                layers.get(layers.size() - 1).tell(
                        new LayerActor.ForwardPass(currentDataPoint.getFeatures().dup(), msg.getWeights(), msg.getBiases())
                );
            }
            catch (Exception ex){
                getContext().getLog().error("Error in trainModel: {}", ex.getMessage());
            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        getContext().getSelf().path().name(), "ParameterReceived", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        getContext().getSelf().path().name(), messageCount
                ));
            }
            return this;
        }
        private Behavior<Command> createLayers(CreateLayers msg) {
            try {
                ActorRef<LayerActor.Command> decoder = getContext().spawn(
                        DecoderLayerActor.create(parameterShard, getContext().getSelf(), this.metricsCollector),
                        "decoder-" + getContext().getSelf().path().name()
                );
                layers.add(decoder);

                ActorRef<LayerActor.Command> latent = getContext().spawn(
                        LatentActor.create(decoder, parameterShard, this.metricsCollector, beta ),
                        "latent-" + getContext().getSelf().path().name()
                );
                layers.add(latent);

                ActorRef<LayerActor.Command> encoder = getContext().spawn(
                        EncoderLayerActor.create(latent, parameterShard, getContext().getSelf(), this.metricsCollector),
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
            long start = System.nanoTime();
            messageCount++;
            try {

            parameterShard.tell(new ParameterShardActor.FetchLatest(getContext().getSelf()));
            }
            catch (Exception ex){

            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        getContext().getSelf().path().name(), "ReadyToProcess", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        getContext().getSelf().path().name(), messageCount
                ));
            }
            return this;
        }
        private Behavior<Command> testModel(TestParameters msg) {

            if (testingData.size() == dataPointsProcessed) {
                parent.tell(new CoordinatorActor.DoneTesting());
                return this;
            }

            currentDataPoint = testingData.get(dataPointsProcessed);
            dataPointsProcessed++;

            layers.get(layers.size() - 1).tell(
                    new LayerActor.TestLayer(currentDataPoint.getFeatures().dup(), msg.getWeights(), msg.getBiases())
            );
            return this;
        }
        private Behavior<Command> onDone(CompleteTraining msg) {
            long start = System.nanoTime();
            messageCount++;

            try {
                String title = getContext().getSelf().path().name() + " Loss History";
                //Operations.plotEpochs(title, "mse",mseHistory, remainingEpochs-1, batchSize);
                Operations.plot(title, "mse", mseHistory);
                dataPointsProcessed = 0;

                getContext().getSelf().tell(new DataShardActor.ValidateParameters(msg.getWeights(), msg.getBiases()));

            }
            catch (Exception ex) {
            }
            finally {
                long elapsedMicros = (System.nanoTime() - start) / 1000;
                metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(
                        getContext().getSelf().path().name(), "onDone", elapsedMicros
                ));
                metricsCollector.tell(new MetricsCollectorActor.LoadEvent(
                        getContext().getSelf().path().name(), messageCount
                ));
            }
            return this;
        }
        private Behavior<Command> onStartTest(StartTest msg){
            this.testingData = msg.getTestingData();

            if (testingData.isEmpty()) {
                parent.tell(new CoordinatorActor.DoneTesting());
                return this;
            }
            dataPointsProcessed=0;

            this.parameterShard.tell(new ParameterShardActor.FetchForTest(getContext().getSelf()));
            getContext().getLog().info("Coordinator Actor {} initialized testing.", getContext().getSelf().path());

            return this;
        }
    }
}
