package org.actor.Extras;

import org.actor.DataPoint;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.*;

public class Operations {
    public static INDArray[] klGrad(
            INDArray mu,
            INDArray logvar
    ) {
        INDArray dMu = mu.dup();  // ∂KL/∂mu = mu

        // ∂KL/∂logvar = 0.5 * (exp(logvar) - 1)
        INDArray dLogvar = Transforms.exp(logvar, true);  // in-place exp
        dLogvar.subi(1.0).muli(0.5);                       // (exp - 1) * 0.5

        return new INDArray[]{dMu, dLogvar};
    }
    //====
    public static void clipGradients(
            INDArray gradients,
            double clipNorm
    ) {
        // Compute L2 norm of the gradient
        double globalNorm = gradients.norm2Number().doubleValue();

        if (globalNorm > clipNorm) {
            double scale = clipNorm / globalNorm;
            gradients.muli(scale); // scale gradients in-place
        }
    }
    //====
    public static List<double[]> readCSV(String filePath) throws IOException {
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
    public static Queue<DataPoint> toINDArrayFormat(List<double[]> cleanedData, int size) {
        Queue<DataPoint> dataPoints = new ArrayDeque<>();
        int i = 0;

        for (double[] row : cleanedData) {
            // Label is last column value (Temp_Flag)
            int labelIndex = row.length - 1;
            int label = (int) row[labelIndex];

            // Features: keep all values as double, unchanged
            double[] featuresArray = Arrays.copyOfRange(row, 0, labelIndex);

            // Create INDArray using double values (dtype DOUBLE)
            //INDArray features = Nd4j.create(featuresArray, new long[]{1, featuresArray.length}, 'c');
            INDArray features = Nd4j.create(featuresArray, new long[]{1, featuresArray.length}, 'c').castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);

            // Add to queue
            dataPoints.add(new DataPoint(features, label));

            if (++i == size) break;
        }

        return dataPoints;
    }
    public static Queue<DataPoint> normalizeData(List<double[]> cleanedData, int size) {
        Queue<DataPoint> dataPoints = new ArrayDeque<>();
        int i = 0;

        for (double[] row : cleanedData) {
            // Label is last column value (Temp_Flag)
            int labelIndex = row.length - 1;
            int label = (int) row[labelIndex];

            // Features: keep all values as double, unchanged
            double[] featuresArray = Arrays.copyOfRange(row, 0, labelIndex);

            // Create INDArray using double values (dtype DOUBLE)
            INDArray features = Nd4j.create(featuresArray, new long[]{1, featuresArray.length}, Nd4j.defaultFloatingPointType());

            // Add to queue
            dataPoints.add(new DataPoint(features, label));

            if (++i == size) break;
        }

        return dataPoints;
    }


    /*public static Queue<DataPoint> normalizeData(
            List<double[]> cleanedData,
            int size
    ) {
        Queue<DataPoint> dataPoints = new ArrayDeque<>();

        if (cleanedData.isEmpty()) {
            return dataPoints;
        }

        int numFeatures = cleanedData.get(0).length - 1; // last column is label

        // Step 1: compute min/max per column
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];
        Arrays.fill(minValues, Double.POSITIVE_INFINITY);
        Arrays.fill(maxValues, Double.NEGATIVE_INFINITY);

        int i=0;
        for (double[] row : cleanedData) {
            for (int j = 0; j < numFeatures; j++) {
                minValues[j] = Math.min(minValues[j], row[j]);
                maxValues[j] = Math.max(maxValues[j], row[j]);
            }
            if(i==size)
                break;
            else
                i++;
        }

        i=0;
        // Step 2: normalize each row
        for (double[] row : cleanedData) {
            int label = (int) row[numFeatures]; // Temp_Flag

            float[] featuresArray = new float[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                double min = minValues[j];
                double max = maxValues[j];
                double value = row[j];

                // normalize to [0,1]
                double normalized = (max > min) ? (value - min) / (max - min) : 0.0;
                featuresArray[j] = (float) normalized;
            }

            INDArray features = Nd4j.create(featuresArray, new long[]{1, featuresArray.length});
            dataPoints.add(new DataPoint(features, label));
            if(i==size)
                break;
            else
                i++;
        }

        return dataPoints;
    }*/
    //====
    public static void plot(
            String title,
            String ytitle,
            List<Double> History
    ) {
        int numIterations = History.size();
        double[] iterations = new double[numIterations];
        double[] mseValues = new double[numIterations];

        for (int i = 0; i < numIterations; i++) {
            iterations[i] = i;
            mseValues[i] = History.get(i);
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(title)
                .xAxisTitle("Iteration")
                .yAxisTitle(ytitle)
                .build();

        chart.addSeries(ytitle, iterations, mseValues);
        new SwingWrapper<>(chart).displayChart();
    }
    //====
    public static void plotEpochs(
            String title,
            String ytitle,
            List<Double> History,
            int epochs,
            int iterationsPerEpoch
    ) {
        double[] epochAxis = new double[epochs];
        double[] epochAveragedMSE = new double[epochs];

        for (int e = 0; e < epochs; e++) {
            epochAxis[e] = e + 1;
            double sum = 0;
            int startIdx = e * iterationsPerEpoch;
            int endIdx = Math.min(startIdx + iterationsPerEpoch, History.size());

            for (int i = startIdx; i < endIdx; i++) {
                sum += History.get(i);
            }

            double avg = (endIdx > startIdx) ? sum / (endIdx - startIdx) : 0;
            epochAveragedMSE[e] = avg;
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(title)
                .xAxisTitle("Epoch")
                .yAxisTitle(ytitle)
                .build();

        chart.addSeries(ytitle, epochAxis, epochAveragedMSE);
        new SwingWrapper<>(chart).displayChart();
    }

    //====
    public static void saveModelDetails(
            String filePath,
            int numShards,
            int batchSize,
            double learningRate,
            double meanMse,
            boolean append
    ) {
        File file = new File(filePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, append))) {
            if (!append) {
                // Write header if file is newly created
                bw.write("numshards,batchsize,learningrate,meanmse");
                bw.newLine();
            }
            // Build the CSV line
            String line = numShards + "," + batchSize + "," + learningRate + "," + meanMse;
            bw.write(line);
            bw.newLine();
        } catch (IOException e) {
            //getContext().getLog().error("Error writing to CSV file: {}", e.getMessage());
        }
    }
    //====
    public static void saveOrginalReconLatentData(
            String filePath,
            INDArray original,
            INDArray reconstruction,
            INDArray latent_space,
            double elbo,
            double kl_loss,
            double mse,
            boolean append) {

        File file = new File(filePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, append))) {
            if (!append) {
                bw.write("original,reconstruction,latent_space,elbo,kl_loss,Mse");
                bw.newLine();
            }

            String originalStr = Arrays.toString(original.toDoubleVector());
            String reconStr    = Arrays.toString(reconstruction.toDoubleVector());
            String latentStr    = Arrays.toString(latent_space.toDoubleVector());

            // Build the CSV line
            String line = originalStr + "," + reconStr + ","+ latentStr + ","+ elbo +"," + kl_loss + "," + mse;
            bw.write(line);
            bw.newLine();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    //====
    public static void saveTestingOrginalReconLatentData(
            String filePath,
            INDArray original,
            INDArray reconstruction,
            INDArray latent_space,
            double elbo,
            double kl_loss,
            double mse,
            int original_label,
            boolean append
    ) {
        File file = new File(filePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, append))) {

            int Dimension = (int) reconstruction.length();

            // Write header only if file is new
            if (!append) {
                // latent_0, latent_1, ..., latent_{n-1}, kl_loss, mse, original_label
                StringBuilder header = new StringBuilder();
                for (int i = 0; i < Dimension; i++) {
                    header.append("point_").append(i);
                    if (i < Dimension - 1) header.append(",");
                }
                header.append(",elbo,kl_loss,Mse,original_label");
                bw.write(header.toString());
                bw.newLine();
            }

            // Write latent values
            double[] latentVals = reconstruction.toDoubleVector();
            StringBuilder row = new StringBuilder();
            for (int i = 0; i < latentVals.length; i++) {
                row.append(latentVals[i]);
                if (i < latentVals.length - 1) row.append(",");
            }

            // Append metrics
            row.append(",")
                    .append(elbo)
                    .append(",").append(kl_loss)
                    .append(",").append(mse)
                    .append(",").append(original_label);

            bw.write(row.toString());
            bw.newLine();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static String arrayToString(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(String.format("%.4f", arr[i]));
            if (i != arr.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }
}
