package org.actor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class DataPoint {
    private final INDArray features;
    private final int label;
    public DataPoint(INDArray features, int label) {
        this.features = features;
        this.label = label;
    }
    public INDArray getFeatures() {
        return features;
    }
    public int getLabel() {
        return label;
    }
}
