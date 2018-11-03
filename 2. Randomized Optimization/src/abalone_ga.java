import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class abalone_ga {
    private static Instance[] instances = readWineData();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 1254);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 1254, 4177);

    private static DataSet set = new DataSet(train_set);

    private static int inputLayer = 11, hiddenLayer=31, outputLayer = 30;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");



    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }



    public static void main(String[] args) {

        String final_result = "";


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        int[] iterations = {500, 1000, 2500, 3000, 5000};
        int[] population = {10,20,50,100,200,500};
        int[] mate = {5,10,25,50,100,350};
        int[] mute = {2,5,10,10,20,50};

        for (int trainingIterations : iterations) {
            results = "";
            for (int q = 0; q < population.length; q++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                oa[0] = new StandardGeneticAlgorithm(population[q], mate[q], mute[q], nnop[0]);
                train(oa[0], networks[0], oaNames[0], trainingIterations);
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[0].getOptimal();
                networks[0].setWeights(optimalInstance.getData());

                // Calculate Training Set Statistics //

                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[0].setInputValues(train_set[j].getData());
                    networks[0].run();

                    Instance example = new Instance(networks[0].getOutputValues());
                    Instance output = train_set[j].getLabel();
                    double temp = example.getData().argMax() == output.getData().argMax()
                            ? correct++: incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTrain Results for GA:" + "," + population[q] + "," + mate[q] + "," + mute[q] + ","  + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                final_result = oaNames[0] + "," + trainingIterations + "," + population[q] + "," + mate[q] + "," + mute[q] + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime);
                write_output_to_file("Optimization_Results", "src/abalone_scaled.csv", final_result, true);

                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                for (int j = 0; j < test_set.length; j++) {
                    networks[0].setInputValues(test_set[j].getData());
                    networks[0].run();

                    Instance example = new Instance(networks[0].getOutputValues());
                    Instance output = test_set[j].getLabel();
                    double temp = example.getData().argMax() == output.getData().argMax()
                            ? correct++: incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for GA: " + "," + population[q] + "," + mate[q] + "," + mute[q] + ","  + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                final_result = oaNames[0] + "," + trainingIterations + "," + population[q] + "," + mate[q] + "," + mute[q] + "," + "testing accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime);
                write_output_to_file("Optimization_Results", "abalone_results_ga.csv", final_result, true);
            }
            System.out.println("results for iteration: " + trainingIterations + "---------------------------");
            System.out.println(results);
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        int trainingIterations = iteration;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));

                train_error += measure.value(output, example);
            }

        }
    }

    private static Instance[] readWineData() {
        double[][][] attributes = new double[4177][][];
        String line = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(
                    "abalone_scaled.csv"));
            int i = 0;
            br.readLine(); // skip header line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[11];
                attributes[i][1] = new double[1];

                for (int j = 0; j < 11; j++) {
                    attributes[i][0][j] = Double.parseDouble(values[j]);
                }
                attributes[i][1][0] = Double.parseDouble(values[11]);
                i++;

            }
            br.close();
        } catch (FileNotFoundException fe) {
            System.out.println("file not found");

        } catch (IOException e) {
            System.out.println("Error reading file");
        }
        Instance[] instances = new Instance[attributes.length];
        for (int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            int c = (int) attributes[i][1][0];
            double[] classes = new double[30];
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
        }
        return instances;

    }
}
