package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.ops.transforms.Transforms;
import org.actor.Extras.ActivationFunctions;
import org.actor.ParameterShardActor;
import org.actor.LayerActor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LatentActor extends LayerActor {

    private final ActorRef<Command> decoder;

    public static Behavior<Command> create(
            ActorRef<Command> decoder,
            ActorRef<ParameterShardActor.Command> parameterShard
    ) {
        return Behaviors.setup(ctx -> new LatentActor(ctx, parameterShard, decoder));
    }
    private LatentActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<Command> decoder
    ) {
        super(context, parameterShard);
        this.decoder = decoder;

        context.getLog().info("Latent Actor {} Created.", context.getSelf().path());
    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(EncoderLayerActor.Forward.class, this::onForward)
                .build();
    }
    private Behavior<Command> onForward(EncoderLayerActor.Forward msg) {
        // Compute mean and log variance
        INDArray zMean = ActivationFunctions.relu(msg.getInput().mmul(msg.getWeights().get(1).transpose()).addRowVector(msg.getBiases().get(1)));
        INDArray zLogVar = ActivationFunctions.relu(msg.getInput().mmul(msg.getWeights().get(2).transpose()).addRowVector(msg.getBiases().get(2)));

        // Sample
        INDArray std = Transforms.exp(zLogVar.mul(0.001));
        INDArray epsilon = Nd4j.randn(std.shape());
        INDArray zSampled = zMean.add(epsilon.mul(std));
        getContext().getLog().info("latent space {} ", zSampled);

        decoder.tell(new DecoderLayerActor.Decode(zSampled, zMean, zLogVar, msg.getWeights(), msg.getBiases()));

        getContext().getLog().info("LatentActor processing latent variables.");

        return this;
    }
}

