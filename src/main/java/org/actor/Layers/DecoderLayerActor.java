package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.actor.DataPoint;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.actor.Extras.ActivationFunctions;
import org.actor.ParameterShardActor;
import org.actor.DataShardActor;
import org.actor.LayerActor;

import java.util.List;

public class DecoderLayerActor extends LayerActor {

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
    // Local message to receive loss result
    public static class LossResponse implements Command {
        private final double mse;
        private final double kl;
        private final double total;
        private final List<ActorRef<LayerActor.Command>> layers;

        public LossResponse(double mse, double kl, double total, List<ActorRef<LayerActor.Command>> layers) {
            this.mse = mse;
            this.kl = kl;
            this.total = total;
            this.layers = layers;
        }
        public double getKl() {
            return kl;
        }
        public double getMse() {
            return mse;
        }
        public double getTotal() {
            return total;
        }
        public List<ActorRef<LayerActor.Command>> getLayers() {
            return layers;
        }
    }

    private final ActorRef<DataShardActor.Command> parent;
    private INDArray decoHiddenLayer;
    private INDArray reconstruction;
    private List<INDArray> currentWeights;
    private List<INDArray> currentBiases;

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
                //.onMessage(LossResponse.class, this::onLossResponse)
                .build();
    }
    private Behavior<Command> onDecode(Decode msg) {
        try {
            currentWeights.add(msg.getWeights().get(3));
            currentBiases.add(msg.getBiases().get(3));

            currentWeights.add(msg.getWeights().get(4));
            currentBiases.add(msg.getBiases().get(4));

            INDArray zSampled = msg.getzSampled();
            getContext().getLog().info("currentWeights shape: {}", currentWeights.get(0).shapeInfoToString());
            getContext().getLog().info("DecoderActor zSampled shape: {}", zSampled.shapeInfoToString());

            // Decoder hidden layer
            decoHiddenLayer = ActivationFunctions.relu(msg.getWeights().get(0).mmul(msg.getzSampled()).addColumnVector(msg.getBiases().get(0)));
            getContext().getLog().info("decoHiddenLayer  shape: {}", decoHiddenLayer.shapeInfoToString());

            // Reconstruct input
            reconstruction = msg.getWeights().get(1).mmul(decoHiddenLayer).addColumnVector(msg.getBiases().get(1));
            getContext().getLog().info("reconstruction shape {}\n", reconstruction.shapeInfoToString());

            parent.tell(new DataShardActor.ComputeLossWithReply(reconstruction, msg.getZMean(), msg.getZLogVar(), getContext().getSelf()));


        } catch (IllegalArgumentException e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }

        return this;
    }
    private Behavior<Command> onLossResponse(LossResponse msg) {
        try{
            double mse = msg.getMse();
            double kl = msg.getKl();

            INDArray W4 = currentWeights.get(0);  // decoder output weights: (hidden_dim, input_dim)
            INDArray W3 = currentWeights.get(1);  // decoder hidden weights: (latent_dim, hidden_dim)

            /*// === Output layer gradients ===
            INDArray dL_dxhat = reconstruction.sub(features).mul(2);  // dL/dx: (batch, input_dim)
            INDArray dW_out = decoHiddenLayer.transpose().mmul(dL_dxhat).div(batchSize);  // (hidden_dim, input_dim)
            INDArray db_out = dL_dxhat.sum(0).div(batchSize);  // (input_dim)

            // === Backprop to hidden layer ===
            INDArray dh_dec = dL_dxhat.mmul(W4);  // (batch, hidden_dim)
            dh_dec = dh_dec.mul(decoHiddenLayer.gt(0));  // ReLU derivative

            // === Hidden layer gradients ===
            INDArray dW_dec = zSampled.transpose().mmul(dh_dec).div(batchSize);  // (latent_dim, hidden_dim)
            INDArray db_dec = dh_dec.sum(0).div(batchSize);  // (hidden_dim)

            // === Optional: gradient w.r.t. zSampled (for encoder) ===
            INDArray dz = dh_dec.mmul(W3);  // (batch, latent_dim)*/

            // === Send gradients backward ===
            /*Backward backwardMsg = new LayerActor.Backward(dW_out, db_out, dz,msg.getLayers().get(2));
            backwardMsg.addGradients(dW_dec, db_dec);
            getContext().getLog().info("\n\n\n");
            getContext().getLog().info("dW_out shape = {}", java.util.Arrays.toString(dW_out.shape()));
            getContext().getLog().info("db_out shape = {}", java.util.Arrays.toString(db_out.shape()));
            getContext().getLog().info("dW_dec shape = {}", java.util.Arrays.toString(dW_dec.shape()));
            getContext().getLog().info("db_dec shape = {}", java.util.Arrays.toString(db_dec.shape()));
            getContext().getLog().info("\n\n\n\n");*/

            // Send to previous layer (e.g., encoder)
            //msg.getLayers().get(1).tell(backwardMsg);


    } catch (Exception ex){

        }
        return this;
    }
}