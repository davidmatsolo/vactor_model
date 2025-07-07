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
    private INDArray zMean;
    private INDArray zLogVar;
    private INDArray zSampled;
    private INDArray epsilon;
    private INDArray std;

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
        zMean = ActivationFunctions.relu(msg.getInput().mmul(msg.getWeights().get(1).transpose()).addRowVector(msg.getBiases().get(1)));
        zLogVar = ActivationFunctions.relu(msg.getInput().mmul(msg.getWeights().get(2).transpose()).addRowVector(msg.getBiases().get(2)));

        // reparamiterization trick
        std = Transforms.exp(zLogVar.mul(0.5));
        epsilon = Nd4j.randn(std.shape());
        zSampled = zMean.add(epsilon.mul(std));
        getContext().getLog().info("latent space {} ", zSampled);

        decoder.tell(new DecoderLayerActor.Decode(zSampled, zMean, zLogVar, msg.getWeights(), msg.getBiases()));

        return this;
    }

    private Behavior<Command> onBackward(DecoderLayerActor.Backward msg){
        getContext().getLog().error("LatentActor received unexpected message: {}", msg);
        INDArray dmu_kl = zMean.dup();
        INDArray dz = msg.getGradients().get(0);

        INDArray dlogvar_kl = Transforms.exp(zLogVar).sub(1).mul(0.5);

        INDArray dsigma = dz.mul(epsilon);
        INDArray dlogvar = dsigma.mul(std).mul(0.5);  // dsigma/dlogvar = 0.5 * exp(0.5 * logvar)
        dlogvar.addi(dlogvar_kl);

        INDArray dmu = dz.add(dmu_kl);  // Add KL gradient to dz/dmu


        return this;
    }
}

