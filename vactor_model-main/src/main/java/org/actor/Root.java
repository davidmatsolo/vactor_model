package org.actor;

import akka.actor.typed.ActorSystem;
import org.actor.Extras.Operations;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

import java.util.*;

public class Root {
    public static void main(String[] args) throws IOException {

        List<double[]> train_raw = Operations.readCSV("D:/MSc_RESEARCH/prototype/data/temp/train_data.csv");
        List<double[]> test_raw = Operations.readCSV("D:/MSc_RESEARCH/prototype/data/temp/test_data.csv");

        //System.out.println("test data size is "+test_raw.size());
        Queue<DataPoint> finaltrainData = Operations.toINDArrayFormat(train_raw, 1000);
        Queue<DataPoint> finaltestData = Operations.toINDArrayFormat(test_raw, test_raw.size());

        if (!finaltrainData.isEmpty()) {
            ActorSystem<CoordinatorActor.Command> system = ActorSystem.create(
                    CoordinatorActor.create(finaltrainData,finaltestData, 10, 9, 62, 4, 0.00015, 80, 0.001, 64), "VactorModel");

            system.tell(new CoordinatorActor.Initialize());
        }
        else{
            System.err.println("Final data is not found. Please check your data source.");
            return;
        }
    }
}
