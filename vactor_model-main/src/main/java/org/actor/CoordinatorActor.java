package org.actor;

import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import akka.actor.typed.Behavior;
import akka.actor.typed.ActorRef;
import org.actor.Extras.MetricsCollectorActor;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class CoordinatorActor {

    //==commands state messages==
    public interface Command { }
    public static class Initialize implements Command { }
    public static class Initialized implements Command { }
    public static class DoneTraining implements Command { }
    public static class DoneTesting implements Command { }
    public static Behavior<Command> create(Queue<DataPoint> traindata, Queue<DataPoint> testdata, int numOfShards, int inputDim, int layerDim, int latentDim, double learningRate, int epochs, double beta, int batchSize) {
        return Behaviors.setup(ctx ->
                new CoordinatorActorActorBehavior(ctx, traindata, testdata, numOfShards, inputDim, layerDim, latentDim, learningRate, epochs, beta, batchSize));
    }
    static class CoordinatorActorActorBehavior extends AbstractBehavior<Command> {
        private final ActorRef<ParameterShardActor.Command> parameterShard;
        private final List<ActorRef<DataShardActor.Command>> dataShards;
        private final ActorRef<MetricsCollectorActor.Command> metricsLogger;

        private List<List<DataPoint>> trainData;
        private List<DataPoint> testData;
        private final double learningRate;
        private final int numOfShards;
        private  int completedShards;
        private final int latentDim;
        private final int inputDim;
        private final int layerDim;
        private final double beta;
        private final int epochs;
        private final int batchSize;
        //
        public CoordinatorActorActorBehavior(
                ActorContext<Command> context,
                Queue<DataPoint> traindata,
                Queue<DataPoint> testdata,
                int numOfShards,
                int inputDim,
                int layerDim,
                int latentDim,
                double learningRate,
                int epochs,
                double beta,
                int batchSize
        ) {
            super(context);
            this.learningRate = learningRate;
            this.numOfShards = numOfShards;
            this.trainData = shardData(traindata);
            this.testData = organiseTestData(testdata);
            this.latentDim = latentDim;
            this.inputDim = inputDim;
            this.layerDim = layerDim;
            this.completedShards =0;
            this.epochs = epochs;
            this.beta = beta;
            this.batchSize = batchSize;
            //====

            String filePath = "D:/MSc_RESEARCH/prototype/code/analysis/metrics/metrics_data_results_"
                    + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".csv";
            this.metricsLogger =
                    getContext().spawn(MetricsCollectorActor.create(getContext().getSelf(), filePath), "metricsLogger");

            this.parameterShard =
                    getContext().spawn(ParameterShardActor.create( getContext().getSelf(), this.metricsLogger,
                                                                this.inputDim, this.layerDim, this.latentDim,
                                                                this.learningRate, this.epochs), "parameterShard");
            //====
            this.dataShards = new ArrayList<>();
            for (int i = 0; i < this.numOfShards; i++) {
                ActorRef<DataShardActor.Command> dataShard =
                        getContext().spawn(DataShardActor.create(context.getSelf(), this.parameterShard,this.metricsLogger,
                                                                    this.trainData.get(i), 0.3, this.beta, this.batchSize), "dataShard-" + i);
                this.dataShards.add(dataShard);
            }

            getContext().getLog().info("Master Actor {} Created.", getContext().getSelf().path());
        }
        @Override
        public Receive<Command> createReceive() {
            return newReceiveBuilder()
                    .onMessage(Initialize.class, this::onInitialize)
                    .onMessage(Initialized.class, this::onInitTraining)
                    .onMessage(DoneTraining.class, this::shardTrainingCompleted)
                    .onMessage(DoneTesting.class, this::shardTestingCompleted)
                    .build();
        }

        //==helper functions==
        private List<List<DataPoint>> shardData(Queue<DataPoint> data) {
            List<List<DataPoint>> shards = new ArrayList<>();
            for (int i = 0; i < this.numOfShards; i++) {
                shards.add(new ArrayList<>());
            }

            int total = data.size();
            int baseSize = total / this.numOfShards; // minimum examples per shard
            int remainder = total % this.numOfShards; // leftover examples

            int shardIndex = 0;
            int countInShard = 0;
            int maxInShard = baseSize + (remainder > 0 ? 1 : 0);

            while (!data.isEmpty()) {
                shards.get(shardIndex).add(data.poll());
                countInShard++;

                if (countInShard >= maxInShard) {
                    shardIndex++;
                    countInShard = 0;
                    if (remainder > 0) remainder--; // reduce remainder as we give extra example
                    maxInShard = baseSize + (remainder > 0 ? 1 : 0);
                }
            }

            getContext().getLog().info("Data of size {} sharded into {} shards size {}",
                    total, this.numOfShards,
                    shards.size());

            return shards;
        }

        private List<DataPoint> organiseTestData(Queue<DataPoint> data) {
            List<DataPoint> test_data = new ArrayList<>();

            for (DataPoint point : data) {
                test_data.add(point);
            }
            getContext().getLog().info("test_data of size  {}",  test_data.size());
            return test_data;
        }

        //==states===
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
        private Behavior<Command> shardTestingCompleted(DoneTesting msg){

            return this;
        }
        private Behavior<Command> shardTrainingCompleted(DoneTraining msg){
            this.completedShards++;
            if (this.completedShards == this.numOfShards){
                this.dataShards.get(0).tell(new DataShardActor.StartTest(this.testData));
                getContext().getLog().info("Coordinator Actor {} send test message.", getContext().getSelf().path());
            }
            return this;
        }
    }
}