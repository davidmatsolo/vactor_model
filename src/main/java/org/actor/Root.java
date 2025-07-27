package org.actor;

import akka.actor.typed.ActorSystem;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.util.*;

public class Root {

    public static void main(String[] args) throws IOException {

        List<double[]> raw = loadCSV("D:/MSc_RESEARCH/prototype/data/temp/normalized_data.csv");
        Queue<DataPoint> finalData = normalData(raw);

        if (!finalData.isEmpty()) {
            ActorSystem<MasterActor.Command> system = ActorSystem.create(
                    MasterActor.create(finalData, 4, 15, 12, 7, 0.09, 10, 2), "VactorModel");

            system.tell(new MasterActor.Initialize());
        }else{
            System.err.println("Final data is not found. Please check your data source.");
            return;
        }
    }
    //==Load CSV into list of double arrays==
    public static List<double[]> loadCSV(String filePath) throws IOException {
        List<double[]> rawData = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String header = reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] row = new double[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    row[i] = Double.parseDouble(tokens[i]);
                }
                rawData.add(row);
            }
        }

        return rawData;
    }
    //====
    public static Queue<DataPoint> normalData(List<double[]> cleanedData) {
        Queue<DataPoint> dataPoints = new LinkedList<>();

        for (double[] row : cleanedData) {
            float[] floatRow = new float[row.length];
            for (int j = 0; j < row.length; j++) {
                floatRow[j] = (float) row[j];
            }

            INDArray features = Nd4j.create(floatRow);  // FLOAT type

            //double tempCelsius = row[4];
            int label = 1;//(tempCelsius <= 28.9) ? 1 : 0;

            dataPoints.add(new DataPoint(features, label));
        }

        return dataPoints;
    }
}
