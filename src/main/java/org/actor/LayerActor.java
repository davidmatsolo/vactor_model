package org.actor;

import org.nd4j.linalg.api.ndarray.INDArray;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.ActorRef;

import java.util.ArrayList;
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
        private List<INDArray> weightGradients;
        private List<INDArray> biasesGradients;
        private INDArray dz;

        private final ActorRef<LayerActor.Command> sendto;

        public Backward(INDArray weightGrads,INDArray biasesGrads,INDArray dz, ActorRef<LayerActor.Command> sendto) {
            this.sendto = sendto;
            this.dz = dz;
            this.weightGradients = new ArrayList<>();
            this.biasesGradients = new ArrayList<>();

            addGradients(weightGrads, biasesGrads);

        }

        public INDArray getDz() {
            return dz;
        }
        public List<INDArray> getBiasesGradients() {
            return biasesGradients;
        }
        public List<INDArray> getWeightGradients() {
            return weightGradients;
        }
        public ActorRef<LayerActor.Command> getSendTo() {
            return sendto;
        }
        public void addGradients(INDArray weightGrads,INDArray biasesGrads){
            this.weightGradients.add(0,weightGrads);
            this.biasesGradients.add(0, biasesGrads);
        }

    }
    protected ActorRef<ParameterShardActor.Command> parameterShard;
    protected LayerActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard) {
        super(context);
        this.parameterShard = parameterShard;
    }
}