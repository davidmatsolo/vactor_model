package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.api.ndarray.INDArray;

import org.actor.Extras.ActivationFunctions;
import org.actor.ParameterShardActor;
import org.actor.DataShardActor;
import org.actor.LayerActor;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;

public class DecoderLayerActor extends LayerActor {
    //==commands state messages==
    public static class Decode implements Command {

        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray zLogVar;
        private final INDArray zSampled;
        private final INDArray zMean;

        public Decode(INDArray zSampled, INDArray zMean, INDArray zLogVar, List<INDArray> weights, List<INDArray> biases) {
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
    public static class LossResponse implements Command {
        private final ActorRef<LayerActor.Command> forwardTo;
        private final ActorRef<LayerActor.Command> sendTo;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private final INDArray diff;
        final double beta;

        //====
        public LossResponse( double beta, INDArray diff, ActorRef<LayerActor.Command> sendTo,
                             ActorRef<LayerActor.Command> forwardTo, List<INDArray> weights, List<INDArray> biases) {
            this.forwardTo = forwardTo;
            this.beta = beta;
            this.sendTo = sendTo;
            this.diff = diff;
            this.weights = weights;
            this.biases = biases;
        }
        public ActorRef<Command> getForwardTo() {
            return forwardTo;
        }
        public ActorRef<Command> getSendTo() {
            return sendTo;
        }
        public INDArray getDiff() {
            return diff;
        }
        public double getBeta() {
            return this.beta;
        }
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
    }
    //====
    private final ActorRef<DataShardActor.Command> parent;
    private INDArray decoHiddenLayer;
    private INDArray reconstruction;
    private INDArray zSampled;

    //====
    public static Behavior<Command> create( ActorRef<ParameterShardActor.Command> parameterShard, ActorRef<DataShardActor.Command> parent) {
        return Behaviors.setup(ctx -> new DecoderLayerActor(ctx, parameterShard, parent));
    }
    private DecoderLayerActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard, ActorRef<DataShardActor.Command> parent) {
        super(context, parameterShard);
        this.parent = parent;
        context.getLog().info("Decoder Actor {} Created.", context.getSelf().path());
    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(Decode.class, this::onDecode)
                .onMessage(LossResponse.class, this::onLossResponse)
                .build();
    }

    //==states==
    private Behavior<Command> onDecode(Decode msg) {
        zSampled = msg.getzSampled();
        try {
            // Decoder hidden layer
            decoHiddenLayer = ActivationFunctions.leakyRelu(msg.getWeights().get(3).mmul(zSampled).addColumnVector(msg.getBiases().get(3)), 0.1);
            //getContext().getLog().info("decoHiddenLayer = {}\n\n",decoHiddenLayer);
            // Reconstruct input be positive
            reconstruction = ActivationFunctions.sigmoid(msg.getWeights().get(4).mmul(decoHiddenLayer).addColumnVector(msg.getBiases().get(4)));

            //getContext().getLog().info("reconstruction = {}",reconstruction);
            parent.tell(new DataShardActor.ComputeLossWithReply(reconstruction, msg.getZMean(), msg.getZLogVar(),
                        getContext().getSelf(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in DecoderLayerActor onDecode: {} {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onLossResponse(LossResponse msg) {
        try {
            INDArray deltaOut = msg.getDiff().reshape(msg.getDiff().length(), 1); // shape: [features, batch]
            //getContext().getLog().info("deltaOut shape = {}\n\n", deltaOut.shapeInfoToString());

            // === Step 1: Output layer backward ===
            INDArray sigmoidOutput = reconstruction;
            INDArray sigmoidPrime = sigmoidOutput.mul(sigmoidOutput.rsub(1));  // sigmoid' * loss
            INDArray dZ2 = deltaOut.mul(sigmoidPrime);
            INDArray dW2 = dZ2.mmul(decoHiddenLayer.transpose());
            INDArray db2 = dZ2.sum(1);                                              // bias2 gradient

            // === Hidden decoder layer backward ===
            INDArray Z1 = msg.getWeights().get(3).mmul(zSampled).addColumnVector(msg.getBiases().get(3)); // pre-activation
            INDArray reluPrime = Z1.gt(0);
            INDArray dHidden = msg.getWeights().get(4).transpose().mmul(dZ2).mul(reluPrime);  // backprop through ReLU
            INDArray dW1 = dHidden.mmul(zSampled.transpose());                      // W1 gradient
            INDArray db1 = dHidden.sum(1);                                          // bias1 gradient

            // === Gradient wrt zSampled (for KL and encoder backprop) ===
            INDArray dL_dzSampled = msg.getWeights().get(3).transpose().mmul(dHidden);

            // === update gradients  ===
            LayerActor.Backward backwardMsg = new LayerActor.Backward(dW2, db2, dL_dzSampled, msg.getForwardTo(),
                    msg.getWeights(), msg.getBiases(), msg.getBeta());
            backwardMsg.addGradients(dW1, db1);  // Add hidden layer gradients
            //==send msg==
            msg.getSendTo().tell(backwardMsg);

            return this;

        } catch (Exception ex) {
            getContext().getLog().error("Error in DecoderLayerActor onLossResponse: {}", ex.getMessage());
            return Behaviors.stopped();
        }
    }
}