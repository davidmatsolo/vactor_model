package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.actor.Extras.Operations;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.actor.Extras.MetricsCollectorActor;
import org.actor.Extras.ActivationFunctions;

import org.actor.ParameterShardActor;
import org.actor.DataShardActor;
import org.actor.LayerActor;

import java.util.List;

public class DecoderLayerActor extends LayerActor {

    // == commands ==
    public static class LossResponse implements Command {
        private final ActorRef<LayerActor.Command> forwardTo;
        private final ActorRef<LayerActor.Command> sendTo;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray datapoint;
        final double beta;

        public LossResponse(double beta, INDArray datapoint,
                            ActorRef<LayerActor.Command> sendTo,
                            ActorRef<LayerActor.Command> forwardTo,
                            List<INDArray> weights, List<INDArray> biases) {
            this.forwardTo = forwardTo;
            this.beta = beta;
            this.sendTo = sendTo;
            this.datapoint = datapoint;
            this.weights = weights;
            this.biases = biases;
        }
        public ActorRef<Command> getForwardTo() { return forwardTo; }
        public ActorRef<Command> getSendTo() { return sendTo; }
        public INDArray getDataPoint() { return datapoint; }
        public double getBeta() { return this.beta; }
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
    }
    public static class Decode implements Command {
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray zLogVar;
        private final INDArray zSampled;
        private final INDArray zMean;

        public Decode(INDArray zSampled, INDArray zMean, INDArray zLogVar,
                      List<INDArray> weights, List<INDArray> biases) {
            this.zSampled = zSampled;
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.weights = weights;
            this.biases = biases;
        }
        public INDArray getzSampled() { return zSampled; }
        public INDArray getZMean() { return zMean; }
        public INDArray getZLogVar() { return zLogVar; }
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
    }
    // == fields ==
    private final ActorRef<MetricsCollectorActor.Command> metricsCollector;
    private final ActorRef<DataShardActor.Command> parent;
    private INDArray decoHiddenLayer_a;
    private INDArray decoHiddenLayer;
    private INDArray reconstruction;
    private INDArray zSampled;
    private long messageCount;

    //== create ==
    public static Behavior<Command> create(
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent,
            ActorRef<MetricsCollectorActor.Command> metricsCollector
    ) {
        return Behaviors.setup(ctx -> new DecoderLayerActor(ctx, parameterShard, parent, metricsCollector));
    }
    private DecoderLayerActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent,
            ActorRef<MetricsCollectorActor.Command> metricsCollector
    ) {
        super(context, parameterShard);
        messageCount = 0;
        this.parent = parent;
        this.metricsCollector = metricsCollector;
        context.getLog().info("Decoder Actor {} Created.", context.getSelf().path());
    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(Decode.class, this::onDecode)
                .onMessage(LossResponse.class, this::onLossResponse)
                .onMessage(TestLayer.class, this::onTestLayer)
                .onMessage(ValidateLayer.class, this::onValidateLayer)
                .build();
    }
    private void reportMetrics(String op, long startTime) {
        long elapsedMicros = (System.nanoTime() - startTime) / 1000;
        metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(getContext().getSelf().path().name(), op, elapsedMicros));
        metricsCollector.tell(new MetricsCollectorActor.LoadEvent(getContext().getSelf().path().name(), messageCount));
    }
    //==actor states==
    private Behavior<Command> onDecode(Decode msg) {
        long start = System.nanoTime();
        messageCount++;
        try {
            zSampled = msg.getzSampled();
            decoHiddenLayer = ActivationFunctions.relu(
                    msg.getWeights().get(3).dup().mmul(zSampled.dup()).addColumnVector(msg.getBiases().get(3).dup()), false);

            decoHiddenLayer_a = ActivationFunctions.sigmoid(
                    msg.getWeights().get(4).dup().mmul(decoHiddenLayer.dup()).addColumnVector(msg.getBiases().get(4).dup()), false);

            reconstruction = decoHiddenLayer_a.dup().reshape(decoHiddenLayer_a.dup().length());

            parent.tell(new DataShardActor.ComputeLossWithReply(
                    reconstruction.dup(), msg.getzSampled().dup(), msg.getZMean().dup(), msg.getZLogVar().dup(),
                    getContext().getSelf(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception ex) {
            getContext().getLog().error("Error in DecoderLayerActor onDecode: {}", ex);
            return Behaviors.stopped();
        } finally {
            reportMetrics("Decode", start);
        }
    }
    private Behavior<Command> onLossResponse(LossResponse msg) {
        long start = System.nanoTime();
        messageCount++;
        try {
            // Always use the zSampled carried by the message (do not rely on actor field)
            INDArray zSampledLocal = zSampled.dup();

            INDArray diff = msg.getDataPoint().dup().sub(reconstruction.dup());
            //getContext().getLog().info("target after calculation = {}", msg.getDataPoint());
            //getContext().getLog().info("diff2 = {}\n",diff);
            INDArray deltaOut = diff.reshape(diff.length(),1);

            INDArray sigmoidPrime = ActivationFunctions.sigmoid(decoHiddenLayer_a.dup(), true);
            INDArray dZ2 = deltaOut.mul(sigmoidPrime);

            INDArray dW2 = dZ2.dup().mmul(decoHiddenLayer.dup().transpose());
            INDArray db2 = dZ2.dup().sum(1);

            Operations.clipGradients(dW2, 0.1);
            Operations.clipGradients(db2, 0.1);

            INDArray tanhPrime = ActivationFunctions.relu(decoHiddenLayer.dup(), true);
            INDArray dHidden = msg.getWeights().get(4).dup().transpose().mmul(dZ2).mul(tanhPrime);

            INDArray dW1 = dHidden.dup().mmul(zSampledLocal.dup().transpose());
            INDArray db1 = dHidden.sum(1);

            Operations.clipGradients(dW1, 0.1);
            Operations.clipGradients(db1, 0.1);

            // gradient of loss wrt zSampled (reconstruction part)
            INDArray dL_dzSampled = msg.getWeights().get(3).dup().transpose().mmul(dHidden);

            LayerActor.Backward backwardMsg = new LayerActor.Backward(
                    dW2.dup(), db2.dup(), dL_dzSampled.dup(), msg.getForwardTo(), msg.getWeights(), msg.getBiases());

            // add decoder's encoder-facing gradients (weights / biases for decoder layer before latent)
            backwardMsg.addGradients(dW1.dup(), db1.dup());

            msg.getSendTo().tell(backwardMsg);
            return this;
        } catch (Exception ex) {
            getContext().getLog().error("Error in DecoderLayerActor onLossResponse: {}", ex);
            return Behaviors.stopped();
        } finally {
            reportMetrics("LossResponse", start);
        }
    }
    private Behavior<Command> onValidateLayer(ValidateLayer msg) {

        try {
            decoHiddenLayer = ActivationFunctions.relu(
                    msg.getWeights().get(3).dup().mmul(msg.getInput().dup()).addColumnVector(msg.getBiases().get(3).dup()), false);

            decoHiddenLayer_a = ActivationFunctions.sigmoid(
                    msg.getWeights().get(4).dup().mmul(decoHiddenLayer.dup()).addColumnVector(msg.getBiases().get(4).dup()), false);

            reconstruction = decoHiddenLayer_a.dup().reshape(decoHiddenLayer_a.length());

            parent.tell(new DataShardActor.EvaluateValidationResult(
                    reconstruction.dup(), msg.getInput().dup(), msg.getzMean().dup(), msg.getzLogVar().dup(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in DecoderLayerActor onValidateLayer: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onTestLayer(TestLayer msg) {
        try {
            decoHiddenLayer = ActivationFunctions.relu(
                    msg.getWeights().get(3).dup().mmul(msg.getInput().dup()).addColumnVector(msg.getBiases().get(3).dup()), false);

            decoHiddenLayer_a = ActivationFunctions.sigmoid(
                    msg.getWeights().get(4).dup().mmul(decoHiddenLayer.dup()).addColumnVector(msg.getBiases().get(4).dup()), false);

            reconstruction = decoHiddenLayer_a.dup().reshape(decoHiddenLayer_a.length());
            parent.tell(new DataShardActor.EvaluateTestResult(
                    reconstruction.dup(),msg.getInput().dup(), msg.getzMean().dup(), msg.getzLogVar().dup(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception ex) {
            getContext().getLog().error("Error in DecoderLayerActor onTestLayer: {}", ex);
            return Behaviors.stopped();
        }
    }
}