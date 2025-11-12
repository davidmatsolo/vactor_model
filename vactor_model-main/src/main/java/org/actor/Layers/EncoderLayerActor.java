package org.actor.Layers;

import org.actor.Extras.MetricsCollectorActor;
import org.actor.Extras.ActivationFunctions;
import org.actor.Extras.Operations;

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
    private final ActorRef<MetricsCollectorActor.Command> metricsCollector;
    private final ActorRef<DataShardActor.Command> parent;
    private final ActorRef<Command> latent;
    private long messageCount = 0;
    private INDArray activated;
    INDArray inputStored;
    INDArray zLogVar;
    INDArray zMean;

    public static Behavior<Command> create(
            ActorRef<Command> latent,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<DataShardActor.Command> parent,
            ActorRef<MetricsCollectorActor.Command> metricsCollector
    ) {
        return Behaviors.setup(ctx -> new EncoderLayerActor(ctx, parameterShard, latent, parent, metricsCollector));
    }

    private EncoderLayerActor(
            ActorContext<Command> context,
            ActorRef<ParameterShardActor.Command> parameterShard,
            ActorRef<Command> latent,
            ActorRef<DataShardActor.Command> parent,
            ActorRef<MetricsCollectorActor.Command> metricsCollector
    ) {
        super(context, parameterShard);
        this.latent = latent;
        this.parent = parent;
        this.metricsCollector = metricsCollector;
        context.getLog().info("EncoderActor {} Created.\n", context.getSelf().path());
    }

    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(ForwardPass.class, this::onEncode)
                .onMessage(Backward.class, this::onBackward)
                .onMessage(TestLayer.class, this::onTestLayer)
                .onMessage(ValidateLayer.class, this::onValidateLayer)
                .build();
    }
    // == helper ==
    private void reportMetrics(String op, long startTime) {
        long elapsedMicros = (System.nanoTime() - startTime) / 1000;
        metricsCollector.tell(new MetricsCollectorActor.LatencyEvent(getContext().getSelf().path().name(), op, elapsedMicros));
        metricsCollector.tell(new MetricsCollectorActor.LoadEvent(getContext().getSelf().path().name(), messageCount));
    }
    //==actor states==
    private Behavior<Command> onValidateLayer(ValidateLayer msg) {
        try {
            inputStored = msg.getInput().dup().reshape(1, (int) msg.getInput().length());
            activated = ActivationFunctions.relu(
                    msg.getWeights().get(0).dup().mmul(inputStored.dup().transpose()).addColumnVector(msg.getBiases().get(0).dup()), false);

            zMean = msg.getWeights().get(1).dup().mmul(activated.dup()).addColumnVector(msg.getBiases().get(1).dup());
            zLogVar = msg.getWeights().get(2).dup().mmul(activated.dup()).addColumnVector(msg.getBiases().get(2).dup());

            msg.setzMean(zMean.dup());
            msg.setzLogVar(zLogVar.dup());
            latent.tell(msg);
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onValidateLayer failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onTestLayer(TestLayer msg) {
        try {
            inputStored = msg.getInput().dup().reshape(1, (int) msg.getInput().length());
            activated = ActivationFunctions.relu(
                    msg.getWeights().get(0).dup().mmul(inputStored.dup().transpose()).addColumnVector(msg.getBiases().get(0).dup()), false);

            zMean = msg.getWeights().get(1).dup().mmul(activated.dup()).addColumnVector(msg.getBiases().get(1).dup());
            zLogVar = msg.getWeights().get(2).dup().mmul(activated.dup()).addColumnVector(msg.getBiases().get(2).dup());

            msg.setzMean(zMean.dup());
            msg.setzLogVar(zLogVar.dup());
            latent.tell(msg);
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onTestLayer failed: {}", e.getMessage());
            return Behaviors.stopped();
        }
    }
    private Behavior<Command> onEncode(ForwardPass msg) {
        long start = System.nanoTime();
        messageCount++;
        try {
            //getContext().getLog().info("target before calculation = {}",msg.getInput());
            inputStored = msg.getInput().reshape(1, (int) msg.getInput().length());
            activated = ActivationFunctions.relu(
                    msg.getWeights().get(0).dup().mmul(inputStored.transpose()).addColumnVector(msg.getBiases().get(0).dup()), false);

            zMean = msg.getWeights().get(1).dup().mmul(activated).addColumnVector(msg.getBiases().get(1).dup());
            zLogVar = msg.getWeights().get(2).dup().mmul(activated).addColumnVector(msg.getBiases().get(2).dup());

            //Operations.clipGradients(zLogVar, 0.1);
            latent.tell(new LatentActor.Reparametize(zMean.dup(), zLogVar.dup(), msg.getWeights(), msg.getBiases()));
            return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onEncode failed: {}", e);
            return Behaviors.stopped();
        } finally {
            reportMetrics("ForwardPass", start);
        }
    }
    private Behavior<Command> onBackward(Backward msg) {
        long start = System.nanoTime();
        messageCount++;
        try {
                INDArray dh_enc = msg.dequeueDelta();
                INDArray d_zMean = msg.dequeueDelta();
                INDArray d_zLogVar = msg.dequeueDelta();

                // derivative of tanh(activated) w.r.t pre-activation
                INDArray tanhPrime = ActivationFunctions.relu(activated.dup(), true);

                // Gradients w.r.t. params that created zMean and zLogVar:
                INDArray dW_logvar =   d_zLogVar.dup().mmul(tanhPrime.dup().transpose());
                INDArray db_logvar = d_zLogVar.dup().sum(1);

                Operations.clipGradients(dW_logvar, 0.1);
                Operations.clipGradients(db_logvar, 0.1);
                msg.addGradients(dW_logvar, db_logvar);

                INDArray dW_mu = d_zMean.dup().mmul(tanhPrime.dup().transpose());
                INDArray db_mu = d_zMean.dup().sum(1);

                Operations.clipGradients(dW_mu, 0.1);
                Operations.clipGradients(db_mu, 0.1);
                msg.addGradients(dW_mu, db_mu);

                // Gradients for encoder input->hidden weights (the encoder that produced 'activated'):
                INDArray dW_enc = dh_enc.dup().mmul(inputStored.dup());
                INDArray db_enc = dh_enc.dup().sum(1);

                Operations.clipGradients(dW_enc, 0.1);
                Operations.clipGradients(db_enc, 0.1);
                msg.addGradients(dW_enc, db_enc);

                parent.tell(new DataShardActor.DataPointProcessed(msg.getWeightGradients(), msg.getBiasesGradients()));
                return this;
        } catch (Exception e) {
            getContext().getLog().error("EncoderLayerActor onBackward failed: {}", e.getMessage());
            return Behaviors.stopped();
        } finally {
            reportMetrics("Backward", start);
        }
    }
}
