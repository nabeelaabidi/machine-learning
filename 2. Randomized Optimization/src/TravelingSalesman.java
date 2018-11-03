/**
 * Created by Rohith Krishnan on 3/9/18.
 */
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class TravelingSalesman {
    /** The n value */
    private static final int N = 30;
    private static int iterations = 30;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        FixedIterationTrainer fit = null;
        double total = 0.0;
        System.out.println("\n\nRandomized Hill Climbing\n\n");
        double rhcTotalTime = 0.0;
        List<Double> results = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, 200000);
            double start = System.nanoTime();
            fit.train();
            double end = System.nanoTime();
            double trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);
            rhcTotalTime += trainingTime;
            total += ef.value(rhc.getOptimal());
            results.add(ef.value(rhc.getOptimal()));

        }
        System.out.println("\n\nAverage Fitness: " + (total / (double) iterations));
        System.out.println("Total Time: " + rhcTotalTime);
        System.out.println("\nFitness Scores");
        for(int i = 0; i < iterations; i++) {
            System.out.println(results.get(i));
        }

        System.out.println("\n\nSimulated Annealing\n\n");
        double saTotalTime = 0.0;
        total = 0.0;
        results = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            double start = System.nanoTime();
            fit.train();
            double end = System.nanoTime();
            double trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            saTotalTime += trainingTime;
            total += ef.value(sa.getOptimal());
            results.add(ef.value(sa.getOptimal()));

        }
        System.out.println("\n\nAverage Fitness: " + (total/ (double) iterations));
        System.out.println("Total Time: " + saTotalTime);
        System.out.println("\nFitness Scores");
        for(int i = 0; i < iterations; i++) {
            System.out.println(results.get(i));
        }

        System.out.println("\n\nGenetic Algorithm\n\n");
        total = 0.0;
        double gaTotalTime = 0.0;
        results = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            double start = System.nanoTime();
            fit.train();
            double end = System.nanoTime();
            double trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);
            gaTotalTime += trainingTime;
            total += ef.value(ga.getOptimal());
            results.add(ef.value(ga.getOptimal()));
        }
        System.out.println("\n\nAverage Fitness: " + (total / (double) iterations));
        System.out.println("Total Time: " + gaTotalTime);
        System.out.println("\nFitness Scores");
        for(int i = 0; i < iterations; i++) {
            System.out.println(results.get(i));
        }


        // for mimic we use a sort encoding



        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("\n\nMIMIC\n\n");
        total = 0.0;
        double mimicTotalTime = 0.0;
        results = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            MIMIC mimic = new MIMIC(200, 5, pop);
            fit = new FixedIterationTrainer(mimic, 900);
            double start = System.nanoTime();
            fit.train();
            double end = System.nanoTime();
            double trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);
            mimicTotalTime += trainingTime;
            total += ef.value(mimic.getOptimal());
            results.add(ef.value(mimic.getOptimal()));
        }
        System.out.println("\n\nAverage Fitness: " + (total / (double) iterations));
        System.out.println("Total Time: " + mimicTotalTime);
        System.out.println("\nFitness Scores");
        for(int i = 0; i < iterations; i++) {
            System.out.println(results.get(i));
        }

    }
}
