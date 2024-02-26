using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class TrainingSet
{
    public double[] input;
    public double output;
}

public class Perceptron : MonoBehaviour
{
    double[] weights;
    double bias;
    double learningRate = 0.1;

    void Start()
    {
        TrainOR();

        // Test the perceptron
        Debug.Log("OR 0 0: " + CalcOutput(new double[] { 0, 0 }));
        Debug.Log("OR 0 1: " + CalcOutput(new double[] { 0, 1 }));
        Debug.Log("OR 1 0: " + CalcOutput(new double[] { 1, 0 }));
        Debug.Log("OR 1 1: " + CalcOutput(new double[] { 1, 1 }));

        TrainAND();

        Debug.Log("AND 0 0: " + CalcOutput(new double[] { 0, 0 }));
        Debug.Log("AND 0 1: " + CalcOutput(new double[] { 0, 1 }));
        Debug.Log("AND 1 0: " + CalcOutput(new double[] { 1, 0 }));
        Debug.Log("AND 1 1: " + CalcOutput(new double[] { 1, 1 }));

        TrainNAND();

        Debug.Log("NAND 0 0: " + CalcOutput(new double[] { 0, 0 }));
        Debug.Log("NAND 0 1: " + CalcOutput(new double[] { 0, 1 }));
        Debug.Log("NAND 1 0: " + CalcOutput(new double[] { 1, 0 }));
        Debug.Log("NAND 1 1: " + CalcOutput(new double[] { 1, 1 }));

        TrainXOR();

        Debug.Log("XOR 0 0: " + CalcOutput(new double[] { 0, 0 }));
        Debug.Log("XOR 0 1: " + CalcOutput(new double[] { 0, 1 }));
        Debug.Log("XOR 1 0: " + CalcOutput(new double[] { 1, 0 }));
        Debug.Log("XOR 1 1: " + CalcOutput(new double[] { 1, 1 }));
    }

    void InitializeWeights(int inputSize)
    {
        weights = new double[inputSize];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = Random.Range(-1.0f, 1.0f);
        }
        bias = Random.Range(-1.0f, 1.0f);
    }

    double DotProductBias(double[] v1, double[] v2)
    {
        if (v1 == null || v2 == null)
            return -1;

        if (v1.Length != v2.Length)
            return -1;

        double d = 0;
        for (int x = 0; x < v1.Length; x++)
        {
            d += v1[x] * v2[x];
        }

        d += bias;

        return d;
    }

    double CalcOutput(double[] input)
    {
        double dp = DotProductBias(weights, input);
        return dp > 0 ? 1 : 0;
    }

    void TrainPerceptron(TrainingSet[] trainingSet, int epochs)
    {
        InitializeWeights(trainingSet[0].input.Length);

        for (int e = 0; e < epochs; e++)
        {
            foreach (TrainingSet ts in trainingSet)
            {
                double error = ts.output - CalcOutput(ts.input);
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] += learningRate * error * ts.input[i];
                }
                bias += learningRate * error;
            }
        }
    }

    void TrainOR()
    {
        TrainingSet[] orTrainingSet = {
            new TrainingSet() { input = new double[] {0, 0}, output = 0 },
            new TrainingSet() { input = new double[] {0, 1}, output = 1 },
            new TrainingSet() { input = new double[] {1, 0}, output = 1 },
            new TrainingSet() { input = new double[] {1, 1}, output = 1 }
        };
        TrainPerceptron(orTrainingSet, 1000);
    }

    void TrainAND()
    {
        TrainingSet[] andTrainingSet = {
            new TrainingSet() { input = new double[] {0, 0}, output = 0 },
            new TrainingSet() { input = new double[] {0, 1}, output = 0 },
            new TrainingSet() { input = new double[] {1, 0}, output = 0 },
            new TrainingSet() { input = new double[] {1, 1}, output = 1 }
        };
        TrainPerceptron(andTrainingSet, 1000);
    }

    void TrainNAND()
    {
        TrainingSet[] nandTrainingSet = {
            new TrainingSet() { input = new double[] {0, 0}, output = 1 },
            new TrainingSet() { input = new double[] {0, 1}, output = 1 },
            new TrainingSet() { input = new double[] {1, 0}, output = 1 },
            new TrainingSet() { input = new double[] {1, 1}, output = 0 }
        };
        TrainPerceptron(nandTrainingSet, 1000);
    }

    double CalcOutputXOR(double i1, double i2)
    {
        double[] input = new double[] { CalcOutput(new double[] { i1, i2 }), i1, i2 };
        return CalcOutput(input);
    }

    void TrainXOR()
    {
        TrainingSet[] xorTrainingSet = {
            new TrainingSet() { input = new double[] {0, 0}, output = 0 },
            new TrainingSet() { input = new double[] {0, 1}, output = 1 },
            new TrainingSet() { input = new double[] {1, 0}, output = 1 },
            new TrainingSet() { input = new double[] {1, 1}, output = 0 }
        };
        TrainPerceptron(xorTrainingSet, 10000);
    }
}
