package org.actor.Layers;

import org.actor.Extras.ActivationFunctions;
import org.actor.ParameterShardActor;
import org.actor.DataShardActor;
import org.actor.LayerActor;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.api.ndarray.INDArray;

public class EncoderLayerActor extends LayerActor {

    final ActorRef<DataShardActor.Command> parent;
    private final ActorRef<Command> latent;
    INDArray inputStored;
    private INDArray activated;
    INDArray zLogVar;
    INDArray zMean;

    public static Behavior<Command> create(ActorRef<Command> latent, ActorRef<ParameterShardActor.Command> parameterShard, ActorRef<DataShardActor.Command> parent) {
        return Behaviors.setup(ctx -> new EncoderLayerActor(ctx, parameterShard, latent, parent));
    }
    private EncoderLayerActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard, ActorRef<Command> latent, ActorRef<DataShardActor.Command> parent) {
        super(context, parameterShard);
        this.latent = latent;
        this.parent = parent;
        context.getLog().info("EncoderActor {} Created.\n", context.getSelf().path());

    }
    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(ForwardPass.class, this::onEncode)
                .onMessage(Backward.class, this::onBackward)
                .build();
    }
    //==states==
    private Behavior<Command> onEncode(ForwardPass msg) {
        if (msg.getInput() == null || msg.getInput().isEmpty()) {
            getContext().getLog().error("Received empty input in EncoderLayerActor.");
            return Behaviors.stopped();
        }

        try {
            inputStored = msg.getInput().reshape(1, (int) msg.getInput().length());

            activated = ActivationFunctions.leakyRelu(msg.getWeights().get(0).mmul(inputStored.transpose()).addColumnVector(msg.getBiases().get(0)), 0.1);

            // Compute mean and log variance
            zMean = msg.getWeights().get(1).mmul(activated).addColumnVector(msg.getBiases().get(1));
            zLogVar = msg.getWeights().get(2).mmul(activated).addColumnVector(msg.getBiases().get(2));

            latent.tell(new LatentActor.Reparametize(zMean, zLogVar, msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onEncode failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onBackward(Backward msg) {
        if (!msg.hasDelta()) {
            getContext().getLog().error("Encoder onBackward: Received null delta.");
            return Behaviors.stopped();
        }

        try {
            INDArray dh_enc = msg.dequeueDelta();
            INDArray d_zMean = msg.dequeueDelta();
            INDArray d_zLogVar = msg.dequeueDelta();

            //
            INDArray dW_logvar = d_zLogVar.mmul(activated.transpose());
            INDArray db_logvar = d_zLogVar.sum(1);
            msg.addGradients(dW_logvar, db_logvar);

            INDArray dW_mu = d_zMean.mmul(activated.transpose());
            INDArray db_mu = d_zMean.sum(1);
            msg.addGradients(dW_mu, db_mu);

            INDArray dW_enc = dh_enc.mmul(inputStored);
            INDArray db_enc = dh_enc.sum(1);
            msg.addGradients(dW_enc, db_enc);

            parent.tell(new DataShardActor.DataPointProcessed(msg.getWeightGradients(), msg.getBiasesGradients()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onBackward failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
}