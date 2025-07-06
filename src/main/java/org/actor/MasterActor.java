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

    public interface Command { }

    public static class Initialize implements Command { }

    public static class Done implements Command {
    }

    public static Behavior<Command> create(Queue<DataPoint> data, int numOfShards, int inputDim, int layerDim, int latentDim, double learningRate, int epochs) {
        return Behaviors.setup(ctx ->
                new MasterActorBehavior(ctx, data, numOfShards, inputDim, layerDim, latentDim, learningRate, epochs));
    }

    static class MasterActorBehavior extends AbstractBehavior<Command> {
        private final List<ActorRef<ParameterShardActor.Command>> parameterShards;
        private final List<ActorRef<DataShardActor.Command>> dataShards;
        private List<Queue<DataPoint>> Data;
        private final double learningRate;
        private final int numOfShards;
        private final int latentDim;
        private final int inputDim;
        private final int layerDim;
        private final int epochs;


        public MasterActorBehavior(ActorContext<Command> context, Queue<DataPoint> data, int numOfShards, int inputDim, int layerDim, int latentDim, double learningRate, int epochs) {
            super(context);
            this.numOfShards = numOfShards;
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.latentDim = latentDim;
            this.learningRate = learningRate;
            this.epochs = epochs;
            this.Data = shardData(data);

            this.parameterShards = new ArrayList<>();
            this.dataShards = new ArrayList<>();

            for (int i = 0; i < this.numOfShards; i++) {

                ActorRef<ParameterShardActor.Command> parameterShard =
                        getContext().spawn(ParameterShardActor.create(this.inputDim, this.layerDim, this.latentDim, this.learningRate, this.epochs), "parameterShard-" + i);

                ActorRef<DataShardActor.Command> dataShard =
                        getContext().spawn(DataShardActor.create(parameterShard, Data.get(i), 0.2, 0.1f), "dataShard-" + i);

                this.parameterShards.add(parameterShard);
                this.dataShards.add(dataShard);
            }

            getContext().getLog().info("Master Actor {} Created.", getContext().getSelf().path());
        }

        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(Initialize.class, this::onInitialize)
                    .build();
        }

        private Behavior<Command> onInitialize(Initialize msg) {
            for (int i = 0; i < numOfShards; i++) {
                this.parameterShards.get(i).tell(new ParameterShardActor.Initialize());
                this.dataShards.get(i).tell(new DataShardActor.Initialize());
            }

            getContext().getLog().info("Master Actor {} initialized.", getContext().getSelf().path());
            return this;
        }

        private List<Queue<DataPoint>> shardData(Queue<DataPoint> data) {
            List<Queue<DataPoint>> shards = new ArrayList<>();
            for (int i = 0; i < numOfShards; i++) {
                shards.add(new LinkedList<>());
            }

            int originalSize = data.size();
            int index = 0;
            while (!data.isEmpty()) {
                shards.get(index % numOfShards).add(data.poll());
                index++;
            }

            getContext().getLog().info("Data of size {} sharded to size {}", originalSize, shards.get(0).size());
            return shards;
        }
    }
}