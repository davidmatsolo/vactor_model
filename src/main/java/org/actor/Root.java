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
        List<double[]> cleaned = cleanData(raw, true);
        //System.out.println("Cleaned data source size is "+ cleaned.size());
        Queue<DataPoint> finalData = normalData(cleaned);


        if (!finalData.isEmpty()) {
            //System.out.println("Cleaned data source size is "+ finalData.size());
            //saveNormalizedDataAsCSV(finalData, "D:/MSc_RESEARCH/prototype/data/temp/normalized_data.csv");
            ActorSystem<MasterActor.Command> system = ActorSystem.create(
                    MasterActor.create(finalData, 1, 15, 12, 6, 0.000000000001, 200, 0.9), "VactorModel");

            system.tell(new MasterActor.Initialize());
        }else{
            System.err.println("Final data is not found. Please check your data source.");
            return;
        }
    }
    // Load CSV into list of double arrays
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
    // Convert Kelvin to Celsius if needed
    public static List<double[]> cleanData(List<double[]> rawData, boolean convertFromKelvin) {
        List<double[]> cleaned = new ArrayList<>();

        for (double[] row : rawData) {
            double[] newRow = row.clone();
            /*if (convertFromKelvin) {
                newRow[4] = newRow[4] - 273.15;
            }*/
            cleaned.add(newRow);
        }

        return cleaned;
    }
    // Normalize features and assign labels, return DataPoints with INDArray features
    public static Queue<DataPoint> normalizeAndLabel(List<double[]> cleanedData) {
        int featureCount = cleanedData.get(0).length;

        // Convert List<double[]> to INDArray matrix (rows = samples, columns = features)
        int sampleCount = cleanedData.size();
        INDArray dataMatrix = Nd4j.create(sampleCount, featureCount);
        for (int i = 0; i < sampleCount; i++) {
            dataMatrix.putRow(i, Nd4j.create(cleanedData.get(i)));
        }

        // Calculate min and max per feature (column-wise)
        INDArray min = dataMatrix.min(0);
        INDArray max = dataMatrix.max(0);
        INDArray range = max.sub(min);

        // Normalize
        INDArray normalized = dataMatrix.subRowVector(min).divRowVector(range);

        Queue<DataPoint> dataPoints = new LinkedList<>();

        for (int i = 0; i < sampleCount; i++) {
            INDArray features = normalized.getRow(i).dup();

            // Assign label based on original temperature (not normalized)
            double tempCelsius = cleanedData.get(i)[4];
            int label = (tempCelsius <= 28.9) ? 1 : 0;

            dataPoints.add(new DataPoint(features, label));
        }
        return dataPoints;
    }
    //
    public static Queue<DataPoint> normalData(List<double[]> cleanedData) {
        Queue<DataPoint> dataPoints = new LinkedList<>();

        for (double[] row : cleanedData) {
            float[] floatRow = new float[row.length];
            for (int j = 0; j < row.length; j++) {
                floatRow[j] = (float) row[j];
            }

            INDArray features = Nd4j.create(floatRow);  // FLOAT type

            double tempCelsius = row[4];
            int label = (tempCelsius <= 28.9) ? 1 : 0;

            dataPoints.add(new DataPoint(features, label));
        }

        return dataPoints;
    }

    // Save normalized data points to CSV
    public static void saveNormalizedDataAsCSV(Queue<DataPoint> dataPoints, String outputPath) throws IOException {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputPath)) {
            while (!dataPoints.isEmpty()) {
                DataPoint dp = dataPoints.poll();
                INDArray features = dp.getFeatures();

                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < features.length(); i++) {
                    sb.append(features.getDouble(i)).append(",");
                }
                sb.append(dp.getLabel());
                writer.println(sb.toString());
            }
        }
    }
}
