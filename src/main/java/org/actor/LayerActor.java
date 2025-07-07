package org.actor;

import org.nd4j.linalg.api.ndarray.INDArray;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.ActorRef;

import java.util.List;

public abstract class LayerActor extends AbstractBehavior<LayerActor.Command> {
    public interface Command {}

    public static class Forward implements Command {
        private final INDArray input_data;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public Forward(INDArray features, List<INDArray> weights, List<INDArray> biases) {
            this.input_data = features;
            this.weights = weights;
            this.biases = biases;
        }
        public INDArray getInput() {
            return input_data;
        }
        public List<INDArray> getWeights() {
            return weights;
        }

        public List<INDArray> getBiases() {
            return biases;
        }
    }
    public static class Backward implements Command {
        private List<INDArray> gradients;
        private final List<ActorRef<LayerActor.Command>> layers;
        private final ActorRef<LayerActor.Command> replyto;

        public Backward(List<INDArray> grads, List<ActorRef<LayerActor.Command>> layers, ActorRef<LayerActor.Command> replyto) {
            this.gradients = grads;
            this.layers = layers;
            this.replyto = replyto;
        }
        public List<INDArray> getGradients() {
            return gradients;
        }
        public List<ActorRef<LayerActor.Command>> getLayers() {
            return layers;
        }
        public ActorRef<LayerActor.Command> getReplyto() {
            return replyto;
        }
    }
    protected ActorRef<ParameterShardActor.Command> parameterShard;
    protected LayerActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard) {
        super(context);
        this.parameterShard = parameterShard;
    }
}