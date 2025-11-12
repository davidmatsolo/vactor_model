package org.actor;

import org.nd4j.linalg.api.ndarray.INDArray;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.ActorRef;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public abstract class LayerActor extends AbstractBehavior<LayerActor.Command> {
    //==commands state messages==
    public interface Command {}
    public static class ForwardPass implements Command {
        private final INDArray input_data;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public ForwardPass(INDArray features, List<INDArray> weights, List<INDArray> biases) {
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
    public static class ValidateLayer implements Command {
        private  INDArray inputData;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private   INDArray zMean;
        private   INDArray zLogVar;

        public ValidateLayer(INDArray features, List<INDArray> weights, List<INDArray> biases) {
            this.inputData =features;
            this.weights = weights;
            this.biases = biases;
        }

        public INDArray getzLogVar() {
            return zLogVar;
        }
        public INDArray getzMean() {
            return zMean;
        }

        public void setInputData(INDArray inputData) {
            this.inputData = inputData;
        }
        public void setzLogVar(INDArray zLogVar) {
            this.zLogVar = zLogVar;
        }
        public void setzMean(INDArray zMean) {
            this.zMean = zMean;
        }

        public INDArray getInput() {
            return this.inputData;
        }
        public List<INDArray> getWeights() {
            return weights;
        }
        public List<INDArray> getBiases() {
            return biases;
        }
    }
    public static class TestLayer implements Command {
        private  INDArray inputData;
        private final List<INDArray> weights;
        private final List<INDArray> biases;
        private   INDArray zMean;
        private   INDArray zLogVar;

        public TestLayer(INDArray features, List<INDArray> weights, List<INDArray> biases) {
            this.inputData =features;
            this.weights = weights;
            this.biases = biases;
        }

        public INDArray getzLogVar() {
            return zLogVar;
        }
        public INDArray getzMean() {
            return zMean;
        }

        public void setInputData(INDArray inputData) {
            this.inputData = inputData;
        }
        public void setzLogVar(INDArray zLogVar) {
            this.zLogVar = zLogVar;
        }
        public void setzMean(INDArray zMean) {
            this.zMean = zMean;
        }

        public INDArray getInput() {
            return this.inputData;
        }
        public List<INDArray> getWeights() {
            return weights;
        }
        public List<INDArray> getBiases() {
            return biases;
        }
    }
    public static class Backward implements Command {
        private final ActorRef<LayerActor.Command> sendto;
        private final List<INDArray> weightGradients;
        private final List<INDArray> biasesGradients;
        private  Queue<INDArray> deltaQueue;
        private final List<INDArray> weights;
        private final List<INDArray> biases;

        public Backward(INDArray weightGrads, INDArray biasesGrads, INDArray delta,
                        ActorRef<LayerActor.Command> sendto, List<INDArray> weights, List<INDArray> biases)
        {
            this.deltaQueue = new LinkedList<>();
            this.weightGradients = new ArrayList<>();
            this.biasesGradients = new ArrayList<>();

            this.sendto = sendto;
            this.deltaQueue.add(delta);
            this.weights = weights;
            this.biases = biases;

            addGradients(weightGrads, biasesGrads);
        }

        public void enqueueDelta(INDArray delta) {
            deltaQueue.add(delta);
        }

        public INDArray dequeueDelta() {
            return deltaQueue.poll(); // Returns null if empty
        }

        public List<INDArray> getWeights() {
            return weights;
        }

        public List<INDArray> getBiases() {
            return biases;
        }

        public boolean hasDelta() {
            return !deltaQueue.isEmpty();
        }

        public ActorRef<LayerActor.Command> getSendTo() {
            return sendto;
        }

        public List<INDArray> getBiasesGradients() {
            return biasesGradients;
        }

        public List<INDArray> getWeightGradients() {
            return weightGradients;
        }

        public void addGradients(INDArray weightGrads, INDArray biasesGrads){
            this.weightGradients.add(0, weightGrads);
            this.biasesGradients.add(0, biasesGrads);
        }
    }
    //====
    protected ActorRef<ParameterShardActor.Command> parameterShard;
    //====
    protected LayerActor(ActorContext<Command> context, ActorRef<ParameterShardActor.Command> parameterShard) {
        super(context);
        this.parameterShard = parameterShard;
    }
}