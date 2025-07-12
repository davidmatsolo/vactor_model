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

import java.util.ArrayList;
import java.util.List;

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
                .onMessage(Backward.class, this::onBackward)
                .build();
    }
    private Behavior<Command> onForward(EncoderLayerActor.Forward msg) {
        // Compute mean and log variance
        INDArray inputStored = msg.getInput();
        //getContext().getLog().info("z input shape = {}", java.util.Arrays.toString(inputStored.shape()));
        //getContext().getLog().info("z msg.getWeights().get(1) shape = {}", java.util.Arrays.toString(msg.getWeights().get(1).shape()));

        zMean = msg.getWeights().get(1).mmul(inputStored).addColumnVector(msg.getBiases().get(1));
        zLogVar = msg.getWeights().get(1).mmul(inputStored).addColumnVector(msg.getBiases().get(2));
        /*
        zMean = inputStored.mmul(msg.getWeights().get(1).transpose()).addRowVector(msg.getBiases().get(1));
        zLogVar = inputStored.mmul(msg.getWeights().get(2).transpose()).addRowVector(msg.getBiases().get(2));*/
        getContext().getLog().info("zMean shape = {}", java.util.Arrays.toString(zMean.shape()));
        getContext().getLog().info("zLogVar shape = {}\n", java.util.Arrays.toString(zLogVar.shape()));

        // ==reparamiterization trick==
        std = Transforms.exp(zLogVar.mul(0.5));
        epsilon = Nd4j.randn(std.shape());
        zSampled = zMean.add(epsilon.mul(std));

        decoder.tell(new DecoderLayerActor.Decode(zSampled, zMean, zLogVar, msg.getWeights(), msg.getBiases()));

        return this;
    }

    private Behavior<Command> onBackward(Backward msg) {

        INDArray dz = msg.getDz();  // Gradient from decoder wrt z
        INDArray dmu_kl = zMean.dup();  // KL derivative wrt mean
        INDArray dlogvar_kl = Transforms.exp(zLogVar).sub(1).mul(0.5);  // KL derivative wrt logvar

        INDArray dsigma = dz.mul(epsilon);
        INDArray dlogvar = dsigma.mul(std).mul(0.5);  // chain rule: dz/dlogvar via std
        dlogvar.addi(dlogvar_kl);  // add KL divergence gradient

        INDArray dmu = dz.add(dmu_kl);  // add KL divergence gradient to dz/dmu

        // === Compute weight and bias gradients for encoder's mean path ===
        INDArray input = zMean;  // input is the same as the input used to generate zMean/zLogVar originally
        long batchSize = input.size(0);

        INDArray dW_mu = input.transpose().mmul(dmu).div(batchSize);   // (input_dim, latent_dim)
        INDArray db_mu = dmu.sum(0).div(batchSize);                    // (latent_dim)

        INDArray dW_logvar = input.transpose().mmul(dlogvar).div(batchSize);  // (input_dim, latent_dim)
        INDArray db_logvar = dlogvar.sum(0).div(batchSize);                   // (latent_dim)

        msg.addGradients(dW_logvar, db_logvar);
        msg.addGradients(dW_mu, db_mu);
        getContext().getLog().info("\n\n\n\n\n\n");
        getContext().getLog().info("dW_mu= {}", dW_mu);
        getContext().getLog().info("db_mu= {}", db_mu);
        getContext().getLog().info("dW_logvar= {}", dW_logvar);
        getContext().getLog().info("db_logvar= {}", db_logvar);
        getContext().getLog().info("\n\n\n\n\n\n");
        // === Send Backward message to encoder ===
        msg.getSendTo().tell(msg);  // assuming encoder is next in the list

        return this;
    }

}

