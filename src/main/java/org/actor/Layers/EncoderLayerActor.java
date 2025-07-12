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
    private INDArray activated;
    private List<INDArray> currentWeights;
    private List<INDArray> currentBiases;

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
                .onMessage(Backward.class, this::onBackward)
                .build();
    }

    private Behavior<Command> onForward(Forward msg) {
        INDArray inputStored = msg.getInput();
        //getContext().getLog().info("input data before reshape: {}", inputStored);

        long[] shape = inputStored.shape();

        getContext().getLog().info("Input shape before reshape: {}", java.util.Arrays.toString(shape));
        inputStored = inputStored.reshape(1, (int) inputStored.length());
        getContext().getLog().info("Input shape after reshape: {}", java.util.Arrays.toString(inputStored.shape()));

        try {
            currentWeights= msg.getWeights();
            currentBiases = msg.getBiases();
            getContext().getLog().info("currentWeights.get(0) shape = {}", java.util.Arrays.toString(currentWeights.get(0).shape()));
            getContext().getLog().info("currentBiases.get(0) shape = {}", java.util.Arrays.toString(currentBiases.get(0).shape()));

            /*
            z1 = inputStored.mmul(currentWeights.get(0).transpose())
                    .addRowVector(currentBiases.get(0));
            */
            z1 = currentWeights.get(0).mmul(inputStored.transpose()).addColumnVector(currentBiases.get(0));
            activated = ActivationFunctions.relu(z1);
            getContext().getLog().info("z1 shape = {}\n", java.util.Arrays.toString(z1.shape()));

            latent.tell(new Forward(activated, msg.getWeights(), msg.getBiases()));
        } catch (Exception e) {
            getContext().getLog().error("Matrix operation failed: {}", e.getMessage());
        }
        return this;
    }
    private Behavior<Command> onBackward(Backward msg) {

        INDArray dlogvar = msg.getWeightGradients().get(1);
        INDArray dmu = msg.getWeightGradients().get(0);

        getContext().getLog().info("dmu shape = {}", java.util.Arrays.toString(dmu.shape()));
        getContext().getLog().info("z1 shape = {}", java.util.Arrays.toString(z1.shape()));
        getContext().getLog().info("activated shape = {}", java.util.Arrays.toString(activated.shape()));

/*
        INDArray W_mu = currentWeights.get(1);       // (latent_dim, hidden_dim)
        INDArray W_logvar = currentWeights.get(2);   // (latent_dim, hidden_dim)

        // Step 1: Compute upstream gradient for weights
        getContext().getLog().info("W_mu shape = {}", java.util.Arrays.toString(W_mu.shape()));

        INDArray dh_mu = dmu.mmul(W_mu);               // (batch, hidden_dim)

        getContext().getLog().info("dh_mu shape = {}\n\n", java.util.Arrays.toString(dh_mu.shape()));
        getContext().getLog().info("dlogvar shape = {}", java.util.Arrays.toString(dlogvar.shape()));

        INDArray dh_logvar = dlogvar.mmul(W_logvar);   // (batch, hidden_dim)
        getContext().getLog().info("dh_logvar shape = {}\n\n", java.util.Arrays.toString(dh_logvar.shape()));

        INDArray dh_enc = dh_mu.add(dh_logvar);
        getContext().getLog().info("before dh_enc shape = {}", java.util.Arrays.toString(dh_enc.shape()));

        // Step 2: Apply ReLU derivative
        INDArray reluMask = activated.gt(0);
        dh_enc = dh_enc.mul(reluMask);
        getContext().getLog().info("after dh_enc shape = {}\n\n", java.util.Arrays.toString(dh_enc.shape()));

        // Step 3: Compute gradients for weights and biases
        long batchSize = inputStored.size(0);
        getContext().getLog().info("inputStored shape = {}", java.util.Arrays.toString(inputStored.shape()));

        INDArray dW_enc = inputStored.transpose().mmul(dh_enc).div(batchSize);
        getContext().getLog().info("dW_enc shape = {}", java.util.Arrays.toString(dW_enc.shape()));

        INDArray db_mu = msg.getBiasesGradients().get(0);
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(db_mu.shape()));

        INDArray db_logvar = msg.getBiasesGradients().get(1);
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(db_logvar.shape()));

        INDArray summed_db_mu = db_mu.sum(0);        // shape: [1, latent_dim] (e.g. [1,4])
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(summed_db_mu.shape()));

        INDArray grad_b_enc_from_mu = summed_db_mu.mmul(W_mu);  // Now [1,4] x [4,12] -> [1,12]
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(summed_db_mu.shape()));

        INDArray summed_db_logvar = db_logvar.sum(0);
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(grad_b_enc_from_mu.shape()));

        INDArray grad_b_enc_from_logvar = summed_db_logvar.mmul(W_logvar);
        getContext().getLog().info("shape = {}", java.util.Arrays.toString(summed_db_mu.shape()));

        INDArray db_enc = grad_b_enc_from_mu.add(grad_b_enc_from_logvar);

        msg.addGradients(dW_enc, db_enc);

        getContext().getLog().info("dW_enc= {}", dW_enc);
        getContext().getLog().info("db_enc= {}", db_enc);

        // Step 4: Send Backward message onward
        msg.getSendTo().tell(msg);*/

        return this;
    }
}