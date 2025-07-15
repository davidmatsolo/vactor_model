package org.actor;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

public class MasterActor {

    //==commands state messages==
    public interface Command { }
    public static class Initialize implements Command { }
    public static class Initialized implements Command { }
    public static class Done implements Command {
    }
    public static Behavior<Command> create(Queue<DataPoint> data, int numOfShards, int inputDim, int layerDim, int latentDim, double learningRate, int epochs, double beta) {
        return Behaviors.setup(ctx ->
                new MasterActorBehavior(ctx, data, numOfShards, inputDim, layerDim, latentDim, learningRate, epochs, beta));
    }
    static class MasterActorBehavior extends AbstractBehavior<Command> {
        private final ActorRef<ParameterShardActor.Command> parameterShard;
        private final List<ActorRef<DataShardActor.Command>> dataShards;
        private List<List<DataPoint>> Data;
        private final double learningRate;
        private final int numOfShards;
        private  int completedShards;
        private final int latentDim;
        private final int inputDim;
        private final int layerDim;
        private final double beta;
        private final int epochs;
        //
        public MasterActorBehavior(ActorContext<Command> context, Queue<DataPoint> data, int numOfShards, int inputDim, int layerDim, int latentDim, double learningRate, int epochs, double beta) {
            super(context);
            this.learningRate = learningRate;
            this.numOfShards = numOfShards;
            this.Data = shardData(data);
            this.latentDim = latentDim;
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.completedShards =0;
            this.epochs = epochs;
            this.beta = beta;
            //====
            this.parameterShard =
                    getContext().spawn(ParameterShardActor.create( getContext().getSelf(),this.inputDim, this.layerDim, this.latentDim, this.learningRate, this.epochs), "parameterShard");
            //====
            this.dataShards = new ArrayList<>();
            for (int i = 0; i < this.numOfShards; i++) {
                ActorRef<DataShardActor.Command> dataShard =
                        getContext().spawn(DataShardActor.create(context.getSelf(), this.parameterShard, Data.get(i), 0.2, this.beta), "dataShard-" + i);
                this.dataShards.add(dataShard);
            }
            getContext().getLog().info("Master Actor {} Created.", getContext().getSelf().path());
        }
        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(Initialize.class, this::onInitialize)
                    .onMessage(Initialized.class, this::onInitTraining)
                    .onMessage(Done.class, this::shardTrainingCompleted)
                    .build();
        }
        private Behavior<Command> onInitialize(Initialize msg) {

            this.parameterShard.tell(new ParameterShardActor.Initialize());
            getContext().getLog().info("Master Actor {} initialized.", getContext().getSelf().path());
            return this;
        }
        private Behavior<Command> onInitTraining(Initialized msg){
            for(ActorRef<DataShardActor.Command> datashard : this.dataShards){
                datashard.tell(new DataShardActor.CreateLayers());
            }
            return this;
        }
        private Behavior<Command> shardTrainingCompleted(Done msg){

            return this;
        }
        private List<List<DataPoint>> shardData(Queue<DataPoint> data) {
            List<List<DataPoint>> shards = new ArrayList<>();
            for (int i = 0; i < this.numOfShards; i++) {
                shards.add(new ArrayList<>());
            }

            int originalSize = data.size();
            int index = 0;
            for (DataPoint point : data) {
                shards.get(index % this.numOfShards).add(point);
                index++;
            }

            getContext().getLog().info("Data of size {} sharded to size {}", originalSize, shards.get(0).size());
            return shards;
        }
    }
}