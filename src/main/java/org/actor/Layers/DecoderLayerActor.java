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

import java.util.List;

public class DecoderLayerActor extends LayerActor {
    //==commands state messages==
    public static class Decode implements Command {
        private final INDArray zSampled;
        private final INDArray zMean;
        private final INDArray zLogVar;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

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

        private final ActorRef<LayerActor.Command> replyT;
        private final ActorRef<LayerActor.Command> sendTo;
        private final INDArray originalInput;
        final double total;
        final double mse;
        final double kl;
        //====
        public LossResponse(double mse, double kl, double total,INDArray originalInput, ActorRef<LayerActor.Command> replyTo, ActorRef<LayerActor.Command> sendTo) {
            this.mse = mse;
            this.kl = kl;
            this.total = total;
            this.replyT = replyTo;
            this.sendTo = sendTo;
            this.originalInput = originalInput;
        }
        public ActorRef<Command> getSendTo() {
            return sendTo;
        }
        public INDArray getOriginalInput() {
            return originalInput;
        }
        public ActorRef<LayerActor.Command> getReplyTo() {
            return replyT;
        }
    }
    //====
    private final ActorRef<DataShardActor.Command> parent;
    private INDArray decoHiddenLayer;
    private INDArray reconstruction;
    private INDArray zSampled;
    //====
    private INDArray decoHidCurrentWeights;
    private INDArray decoHidCurrentBiases;
    private INDArray reConCurrentWeights;
    INDArray reConCurrentBiases;
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

        decoHidCurrentWeights = msg.getWeights().get(3);
        decoHidCurrentBiases = msg.getBiases().get(3);

        reConCurrentWeights = msg.getWeights().get(4);
        reConCurrentBiases = msg.getBiases().get(4);

        try {
            // Decoder hidden layer
            decoHiddenLayer = ActivationFunctions.relu(decoHidCurrentWeights.mmul(zSampled).addColumnVector(decoHidCurrentBiases));
            // Reconstruct input
            reconstruction = reConCurrentWeights.mmul(decoHiddenLayer).addColumnVector(reConCurrentBiases);

            parent.tell(new DataShardActor.ComputeLossWithReply(reconstruction, msg.getZMean(), msg.getZLogVar(), getContext().getSelf()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onLossResponse(LossResponse msg) {
        try{
            INDArray OriginalInput = msg.getOriginalInput().reshape((int) msg.getOriginalInput().length(),1);
            INDArray deltaOut = reconstruction.sub(OriginalInput);

            // === Step 3: d1 = ReLU'(z1) * (W2 * d0) ===
            INDArray dW_out = deltaOut.mmul(decoHiddenLayer.transpose());
            INDArray db_out = deltaOut.sum(1);

            // === Step 3: d1 = ReLU'(z1) * (W2 * d0) ===
            INDArray z1 = decoHidCurrentWeights.mmul(zSampled).addColumnVector(decoHidCurrentBiases);
            INDArray reluPrime = z1.gt(0); // ReLU derivative (1 where z1 > 0, else 0)
            INDArray deltaHidden = reConCurrentWeights.transpose().mmul(deltaOut).mul(reluPrime);

            // === Step 4: Gradients for hidden layer weights and biases ===
            INDArray dW_dec = deltaHidden.mmul(zSampled.transpose());
            INDArray db_dec = deltaHidden.sum(1);

            INDArray dL_dzSampled = decoHidCurrentWeights.transpose().mmul(deltaHidden);

            double clipNorm = 1.0;
            double norm = dL_dzSampled.norm2Number().doubleValue();
            if (norm > clipNorm) {
                double scale = clipNorm / (norm+0.001);
                dL_dzSampled.muli(scale);
            }

            LayerActor.Backward backwardMsg = new LayerActor.Backward(dW_out, db_out, dL_dzSampled, msg.getSendTo());
            backwardMsg.addGradients(dW_dec, db_dec);

            msg.getReplyTo().tell(backwardMsg);
            return this;

    } catch (Exception ex){
            getContext().getLog().error("Error in LatentActor onForward: {}", ex.getMessage());
            return Behaviors.stopped();
        }
    }
}