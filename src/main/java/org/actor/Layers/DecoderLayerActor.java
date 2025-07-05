
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

    public static class Decode implements Command {
        private final INDArray input_data;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public Decode(INDArray data, List<INDArray> weights, List<INDArray> biases) {
            this.input_data = data;
            this.weights = weights;
            this.biases = biases;
        }
        public INDArray getInput_data() {
            return input_data;
        }
        public List<INDArray> getWeights() {
            return weights;
        }
        public List<INDArray> getBiases() {
            return biases;
        }
    }
    private final ActorRef<DataShardActor.Command> parent;
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
                .build();
    }
    private Behavior<Command> onDecode(Decode msg) {
        try {
            // Decoder hidden layer
            INDArray h = ActivationFunctions.relu(msg.getInput_data().mmul(msg.getWeights().get(3).transpose()).addRowVector(msg.getBiases().get(3)));
            getContext().getLog().info("DecoderActor activated shape: {}", h.shapeInfoToString());

            // Reconstruct input
            INDArray reconstruction = h.mmul(msg.getWeights().get(4).transpose()).addRowVector(msg.getBiases().get(4));
            getContext().getLog().info("DecoderActor reconstructed data to shape: {}", reconstruction.shapeInfoToString());
            getContext().getLog().debug("Reconstructed data: {}", reconstruction);

            parent.tell(new DataShardActor.ComputeLoss(reconstruction));
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }

        return this;
    }
}