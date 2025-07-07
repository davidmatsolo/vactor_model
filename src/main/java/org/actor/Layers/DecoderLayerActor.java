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
        private final DataPoint currentDataPoint;
        private final List<ActorRef<LayerActor.Command>> layers;

        public LossResponse(double mse, double kl, double total, List<ActorRef<LayerActor.Command>> layers, DataPoint currentDataPoint) {
            this.mse = mse;
            this.kl = kl;
            this.total = total;
            this.layers = layers;
            this.currentDataPoint = currentDataPoint;
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
        public DataPoint getCurrentDataPoint() {
            return currentDataPoint;
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
    private INDArray zSampled;

    public static Behavior<Command> create(
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent
    ) {
        return Behaviors.setup(ctx -> new DecoderLayerActor(ctx, parameterShard, parent));
    }
    private DecoderLayerActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent
    ) {
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


    private Behavior<Command> onDecode(Decode msg) {
        try {
            currentWeights= msg.getWeights();
            currentBiases = msg.getBiases();
            zSampled = msg.getzSampled();

            // Decoder hidden layer
            decoHiddenLayer = ActivationFunctions.relu(msg.getzSampled().mmul(msg.getWeights().get(3).transpose()).addRowVector(msg.getBiases().get(3)));
            //getContext().getLog().info("DecoderActor activated shape: {}", decoHiddenLayer.shapeInfoToString());

            // Reconstruct input
            reconstruction = decoHiddenLayer.mmul(msg.getWeights().get(4).transpose()).addRowVector(msg.getBiases().get(4));
            getContext().getLog().info("DecoderActor reconstructed data {}", reconstruction.shapeInfoToString());

            parent.tell(new DataShardActor.ComputeLossWithReply(reconstruction, msg.getZMean(), msg.getZLogVar(), getContext().getSelf()));


        } catch (IllegalArgumentException e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }

        return this;
    }
    private Behavior<Command> onLossResponse(LossResponse msg) {
        getContext().getLog().info("Received Loss: total={}, mse={}, kl={}", msg.getTotal(), msg.getMse(), msg.getKl());

        INDArray W4 = currentBiases.get(4);  // decoder output weights: (hidden_dim, input_dim)
        INDArray W3 = currentWeights.get(3); // decoder hidden weights: (latent_dim, hidden_dim)

        INDArray dL_dxhat = reconstruction.sub(msg.currentDataPoint.getFeatures()).mul(2);  // (batch, input_dim)

        INDArray dW_out = decoHiddenLayer.transpose().mmul(dL_dxhat).div(msg.currentDataPoint.getFeatures().size(0));  // (hidden_dim, input_dim)

        INDArray dh_dec = dL_dxhat.mmul(W4);  // (batch, hidden_dim)
        dh_dec = dh_dec.mul(decoHiddenLayer.gt(0));  // ReLU backprop

        INDArray dW_dec = zSampled.transpose().mmul(dh_dec);  // (latent_dim, hidden_dim)
        INDArray dz = dh_dec.mmul(W3);  // (batch, latent_dim)

        return this;
    }
}