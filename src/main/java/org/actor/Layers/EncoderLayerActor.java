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

import java.util.List;


public class EncoderLayerActor extends LayerActor {

    private final ActorRef<Command> latent;
    private final ActorRef<DataShardActor.Command> parent;
    private INDArray z1;

    private List<INDArray> currentWeights;
    private List<INDArray> currentBiases;
    private INDArray inputStored;

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
        inputStored = msg.getInput();
        getContext().getLog().info("input data before reshape: {}", inputStored);

        if (inputStored.rank() == 1) {
            inputStored = inputStored.reshape(1, inputStored.size(0));
        }
        getContext().getLog().info("input data after reshape: {}", inputStored);

        try {
            currentWeights= msg.getWeights();
            currentBiases = msg.getBiases();

            z1 = inputStored.mmul(currentWeights.get(0).transpose())
                    .addRowVector(currentBiases.get(0));
            INDArray activated = ActivationFunctions.relu(z1);
            getContext().getLog().info("activated data: {}", activated);

            latent.tell(new Forward(activated, msg.getWeights(), msg.getBiases()));
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }

        return this;
    }

    private Behavior<Command> onBackward(Backward msg) {
        INDArray dlogvar = msg.getGradients().get(0);  // dL/dlogvar
        INDArray dmu = msg.getGradients().get(1);      // dL/dmu

        // Extract weights from msg
        INDArray W_mu = currentWeights.get(1);        // weights to mu layer
        INDArray W_logvar = currentWeights.get(2);    // weights to logvar layer

        // Compute ∂L/∂h_enc = dmu · W_muᵀ + dlogvar · W_logvarᵀ
        INDArray dh_mu = dmu.mmul(W_mu.transpose());
        INDArray dh_logvar = dlogvar.mmul(W_logvar.transpose());
        INDArray dh_enc = dh_mu.add(dh_logvar);

        // ReLU backprop
        INDArray reluMask = z1.gt(0); // z1 is pre-activation
        dh_enc = dh_enc.mul(reluMask);

        // Gradient w.r.t encoder weights
        INDArray dW_enc = inputStored.transpose().mmul(dh_enc);
        INDArray db_enc = dh_enc.sum(0);

        // Send to parameter shard or log it
        //parameterShard.tell(new ParameterShardActor.UpdateGradients(List.of(dW_enc), List.of(db_enc)));

        return this;
    }
}