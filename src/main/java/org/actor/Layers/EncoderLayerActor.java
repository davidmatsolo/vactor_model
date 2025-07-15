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
    private INDArray activated;
    INDArray currentWeights;
    INDArray currentBiases;
    INDArray inputStored;
    private INDArray z1;

    public static Behavior<Command> create(
            ActorRef<Command> latent,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent
    ) {
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
                .onMessage(Forward.class, this::onForward)
                .onMessage(Backward.class, this::onBackward)
                .build();
    }
    //==states==
    private Behavior<Command> onForward(Forward msg) {
        inputStored = msg.getInput();
        if (inputStored == null || inputStored.isEmpty()) {
            getContext().getLog().error("Received empty input in EncoderLayerActor.");
            return Behaviors.stopped();
        }

        currentWeights= msg.getWeights().get(0);
        currentBiases = msg.getBiases().get(0);

        try {
            inputStored = inputStored.reshape(1, (int) inputStored.length());

            z1 = currentWeights.mmul(inputStored.transpose()).addColumnVector(currentBiases);
            activated = ActivationFunctions.relu(z1);

            latent.tell(new Forward(activated, msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onBackward(Backward msg) {
        INDArray delta = msg.getDelta();
        if (delta == null) {
            getContext().getLog().error("Encoder onBackward: Received null delta.");
            return Behaviors.stopped();
        }

        try {
            // ==ReLU Derivative==
            INDArray reluPrime = activated.gt(0);
            INDArray deltaActivated = delta.mul(reluPrime); // element-wise multiply
            // ==Compute Gradients==
            INDArray dW = deltaActivated.mmul(inputStored);
            INDArray db = deltaActivated.sum(1);  // sum across batch dimension
            //==Update Message==
            msg.addGradients(dW, db);

            parent.tell(new DataShardActor.DataPointProcessed(msg.getWeightGradients(), msg.getBiasesGradients()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onBackward failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
}