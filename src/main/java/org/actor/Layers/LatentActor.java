package org.actor.Layers;

import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.ops.transforms.*;
import org.actor.ParameterShardActor;
import org.actor.LayerActor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LatentActor extends LayerActor {

    private final ActorRef<Command> decoder;
    private INDArray zMean;
    private INDArray zLogVar;
    private INDArray zSampled;
    private INDArray inputStored;
    private INDArray std;
    private INDArray epsilon;
    //
    INDArray zMeanCurrentWeights;
    INDArray zMeanCurrentBiases;
    INDArray zLogVarCurrentWeights;
    INDArray zLogVarCurrentBiases;


    public static Behavior<Command> create(
            ActorRef<Command> decoder,
            ActorRef<ParameterShardActor.Command> parameterShard
    ) {
        return Behaviors.setup(ctx -> new LatentActor(ctx, parameterShard, decoder));
    }

    private LatentActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard, ActorRef<Command> decoder) {
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

        zMeanCurrentWeights = msg.getWeights().get(1);
        zMeanCurrentBiases = msg.getBiases().get(1);

        zLogVarCurrentWeights = msg.getWeights().get(2);
        zLogVarCurrentBiases = msg.getBiases().get(2);

        try {

            // Compute mean and log variance
            inputStored = msg.getInput();
            zMean = zMeanCurrentWeights.mmul(inputStored).addColumnVector(zMeanCurrentBiases);
            zLogVar = zLogVarCurrentWeights.mmul(inputStored).addColumnVector(zLogVarCurrentBiases);
            // ==reparamiterization trick==
            std = Transforms.exp(zLogVar.mul(0.5));
            epsilon = Nd4j.randn(std.shape());
            zSampled = zMean.add(epsilon.mul(std));

            decoder.tell(new DecoderLayerActor.Decode(zSampled, zMean, zLogVar, msg.getWeights(), msg.getBiases()));

            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in LatentActor onForward: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onBackward(Backward msg) {
        try {
            INDArray dL_dzSampled = msg.getDelta();

            INDArray dL_dMean = dL_dzSampled.add(zMean.neg());
            INDArray dL_dLogVar_recon = dL_dzSampled.mul(std).mul(epsilon).mul(0.5);
            INDArray dL_dLogVar_kl = Transforms.exp(zLogVar).rsub(1.0).mul(0.5);
            INDArray dL_dLogVar = dL_dLogVar_recon.add(dL_dLogVar_kl);

            // === Weights ===
            INDArray dW_mean = dL_dMean.mmul(inputStored.transpose());
            INDArray dW_logvar = dL_dLogVar.mmul(inputStored.transpose());

            // === Biases ===
            INDArray db_mean = dL_dMean.sum(1);       // sum across batch
            INDArray db_logvar = dL_dLogVar.sum(1);   // sum across batch
            //==Deltas==
            INDArray deltaToEncoder = zMeanCurrentWeights.transpose().mmul(dL_dMean)
                    .add(zLogVarCurrentWeights.transpose().mmul(dL_dLogVar));

            // pseudo-code for clipping in Java/ND4J
            double clipNorm = 1.0;
            double norm = deltaToEncoder.norm2Number().doubleValue();

            if (norm > clipNorm) {
                double scale = clipNorm / (norm + 0.001);
                deltaToEncoder.muli(scale);
            }

            //==Update msg==
            msg.addGradients(dW_mean, db_mean);
            msg.addGradients(dW_logvar, db_logvar);
            msg.setDelta(deltaToEncoder);

            getContext().getLog().info("Latent layer Grad norm = {}\n\n" ,deltaToEncoder.norm2Number());

            msg.getSendTo().tell(msg);
            return this;
        }catch (Exception ex) {
            getContext().getLog().error("LatentActor onBackward failed: {}", ex.getMessage());
            return Behaviors.stopped();
        }
    }
}

