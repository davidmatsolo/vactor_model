package org.actor.Layers;

import org.actor.Extras.ActivationFunctions;
import org.actor.ParameterShardActor;
import org.actor.DataShardActor;
import org.actor.LayerActor;
import org.actor.DataPoint;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class EncoderLayerActor extends LayerActor {

    private final ActorRef<Command> latent;
    private final ActorRef<DataShardActor.Command> parent;

    public static Behavior<Command> create(
            ActorRef<Command> latent,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent
    ) {
        return Behaviors.setup(ctx -> new EncoderLayerActor(ctx, parameterShard, latent, parent));
    }
    private EncoderLayerActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<Command> latent,
            ActorRef<DataShardActor.Command> parent

    ) {
        super(context, parameterShard);
        this.latent = latent;
        this.parent = parent;
        context.getLog().info("EncoderActor {} Created.", context.getSelf().path());

        parent.tell(new DataShardActor.ReadyToProcess());
    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(Forward.class, this::onForward)
                .build();
    }

    private Behavior<Command> onForward(Forward msg) {
        INDArray input = msg.getInput();
        if (input.rank() == 1) {
            input = input.reshape(1, input.size(0)); // Ensure it's 2D
        }

        try {
            INDArray z1 = input.mmul(msg.getWeights().get(0).transpose())
                    .addRowVector(msg.getBiases().get(0));
            INDArray activated = ActivationFunctions.relu(z1);
            getContext().getLog().info("activated data: {}", activated);

            latent.tell(new Forward(activated, msg.getWeights(), msg.getBiases()));
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }

        return this;
    }
}