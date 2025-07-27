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

import java.util.List;

public class LatentActor extends LayerActor {

    public static class Reparametize implements Command {
        private final INDArray zMean;
        private final INDArray zLogVar;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public Reparametize(INDArray zMean, INDArray zLogVar, List<INDArray> weights, List<INDArray> biases) {
            this.zMean = zMean;
            this.zLogVar = zLogVar;
            this.weights = weights;
            this.biases = biases;
        }
        public INDArray getZMean() { return zMean; }
        public INDArray getZLogVar() { return zLogVar; }
        public List<INDArray> getWeights() { return weights; }
        public List<INDArray> getBiases() { return biases; }
    }
    private final ActorRef<Command> decoder;
    INDArray std;
    INDArray epsilon;
    INDArray zSampled;
    INDArray zMean;
    INDArray zLogVar;
    public static Behavior<Command> create(ActorRef<Command> decoder, ActorRef<ParameterShardActor.Command> parameterShard) {
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
                .onMessage(Reparametize.class, this::onReparametize)
                .onMessage(Backward.class, this::onBackward)
                .build();
    }
    private Behavior<Command> onReparametize(Reparametize msg) {
        zMean = msg.getZMean();
        zLogVar =msg.getZLogVar();
        try {
            // ==reparamiterization trick==
            INDArray safeLogVar = Transforms.min(Transforms.max(zLogVar, Nd4j.scalar(-10)), Nd4j.scalar(10));
            std = Transforms.exp(safeLogVar.mul(0.5));
            epsilon = Nd4j.randn(std.shape());
            zSampled = msg.getZMean().add(epsilon.mul(std));

            decoder.tell(new DecoderLayerActor.Decode(zSampled,zMean ,zLogVar , msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("Error in LatentActor onForward: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onBackward(Backward msg) {

        try {
            INDArray dL_dzSampled = msg.dequeueDelta();

            //==Deltas zMean==
            INDArray dL_zMean = dL_dzSampled.add(zMean.neg()); //d_mu += dz

            //==Deltas LogVar==
            INDArray dL_zLogVar_recon = dL_dzSampled.mul(std).mul(epsilon).mul(0.5);//d_std = dz * eps
            INDArray dL_zLogVar_kl = Transforms.exp(zLogVar).rsub(1.0).mul(0.5 * msg.getBeta());
            INDArray dL_zLogVar = dL_zLogVar_recon.add(dL_zLogVar_kl);

            // === Deltas ecoder hidden ===
            INDArray dh_enc_zMean = msg.getWeights().get(1).transpose().mmul(dL_zMean);
            INDArray dh_enc_zLogVar = msg.getWeights().get(2).transpose().mmul(dL_zLogVar);
            INDArray deltaToEncoder = dh_enc_zMean.add(dh_enc_zLogVar);

            //==Update msg==
            msg.enqueueDelta(deltaToEncoder);
            msg.enqueueDelta(dL_zMean);
            msg.enqueueDelta(dL_zLogVar);

            //==send msg==
            msg.getSendTo().tell(msg);
            return  this;
        }catch (Exception ex){
            getContext().getLog().error("Error in LatentActor onBackward: {}", ex.getMessage());
            return Behaviors.stopped();
        }
    }
}

